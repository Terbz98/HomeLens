"""Turn a confirmed house location into the four HomeLens estimates."""
import logging
import math
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np

from core import furniture
from core.features import RENT_PRICE_LIMITS, SALE_PRICE_LIMITS, is_valid_price
from core.imagery import ExteriorImage, ImageryError, ImageryProvider

log = logging.getLogger("homelens")

LABELS = {
    "buy_unfurnished": "Buy, without furniture",
    "buy_furnished": "Buy, with furniture",
    "rent_unfurnished": "Rent, without furniture",
    "rent_furnished": "Rent, with furniture",
}

NO_RANGE_AREA = ("A reliable range is unavailable. This is an area-level figure and there are no "
                 "per-house prices for this city to check a range against.")
NO_RANGE_SCENARIO = "No range is shown because the furniture amount is an assumption, not a prediction."
NO_RANGE_FEW_ROWS = "A reliable range is unavailable because there were too few rows to calibrate one."


def _round_to(value: float, step: int) -> int:
    return int(round(value / step) * step)


def _default_embed(jpeg_bytes: bytes) -> np.ndarray:
    from core.embed import embed_image  # imported late: torch is slow to load
    return embed_image(jpeg_bytes)


class Estimator:
    """Holds the trained models and produces estimates.

    `bundle` is the dictionary saved by pipeline.train, or None when no
    models have been trained yet.
    """

    def __init__(self, bundle: Optional[Dict], provider: ImageryProvider, package: Dict, demo: bool,
                 embed_fn: Callable[[bytes], np.ndarray] = _default_embed):
        self.bundle = bundle
        self.provider = provider
        self.package = package
        self.demo = demo
        self._embed = embed_fn
        if bundle is not None and bool(bundle["manifest"]["synthetic_training_data"]) != bool(demo):
            # Never serve synthetic-data models as real, or real models as demo.
            raise ValueError("model bundle kind does not match the server mode")

    @classmethod
    def from_dir(cls, model_dir: Path, provider: ImageryProvider, package: Dict, demo: bool) -> "Estimator":
        path = Path(model_dir) / "bundle.joblib"
        bundle = joblib.load(path) if path.exists() else None
        return cls(bundle, provider, package, demo)

    # ------------------------------------------------------------ status helpers

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    @property
    def manifest(self) -> Optional[Dict]:
        return self.bundle["manifest"] if self.bundle else None

    @property
    def model_version(self) -> Optional[str]:
        return self.manifest["model_version"] if self.bundle else None

    def has_per_house_model(self, city: Dict) -> bool:
        return bool(self.bundle and city.get("per_house_sales") and self.bundle.get("seattle_sale"))

    def photo_model_served(self, city: Dict) -> bool:
        photo = self.bundle.get("seattle_photo") if self.bundle else None
        return bool(self.has_per_house_model(city) and photo and photo.get("serve"))

    # ------------------------------------------------------------ the photo step

    def _photo_adjustment(self, lat: float, lon: float, city: Dict, use_image: bool) -> Tuple[Dict, Optional[float]]:
        """Try to get a photo-based adjustment factor for the location-only price.

        Returns (image info for the response, factor or None).
        """
        if not use_image:
            return {"status": "not_requested", "message": "You asked for a location-only estimate."}, None
        photo = self.bundle.get("seattle_photo") if self.has_per_house_model(city) else None
        if photo is None:
            return {"status": "no_photo_model",
                    "message": "No photo model has been trained for this city, so the visible exterior was not analyzed."}, None
        if not photo.get("serve"):
            return {"status": "no_photo_model",
                    "message": "The photo model did not beat the location-only model in testing, so it is not used."}, None
        if not self.provider.configured:
            return {"status": "not_configured",
                    "message": "No licensed imagery source is set up on the server, so the visible exterior was not analyzed."}, None
        try:
            image = self.provider.find_best(lat, lon)
        except ImageryError as exc:
            log.warning("imagery search failed: %s", exc)
            return {"status": "error", "message": "The imagery source could not be reached. Location only was used."}, None
        if image is None:
            return {"status": "unavailable",
                    "message": "No licensed street photo was found close enough to this house."}, None
        try:
            embedding = self._embed(self.provider.download(image))
            factor = float(math.exp(float(photo["residual_model"].predict(np.asarray(embedding).reshape(1, -1))[0])))
        except Exception as exc:  # any failure here falls back to location only
            log.warning("photo step failed: %s", type(exc).__name__)
            return {"status": "error", "message": "The photo could not be analyzed. Location only was used."}, None
        if not (math.isfinite(factor) and 0.2 <= factor <= 5.0):
            return {"status": "error", "message": "The photo model gave an unusable result. Location only was used."}, None
        return self._image_info(image, "used", "This licensed street photo was analyzed for the estimate."), factor

    @staticmethod
    def _image_info(image: ExteriorImage, status: str, message: str) -> Dict:
        return {
            "status": status, "message": message, "provider": image.provider, "image_id": image.image_id,
            "thumb_url": image.thumb_url, "page_url": image.page_url, "creator": image.creator,
            "captured_at": image.captured_at, "license": image.license, "distance_m": image.distance_m,
        }

    # ------------------------------------------------------------ the estimate

    def _interval(self, amount: int, spec: Optional[Dict]) -> Optional[Dict]:
        if not spec:
            return None
        half = spec["relative_half_width"]
        return {
            "low_usd": _round_to(amount * (1 - half), 1000),
            "high_usd": _round_to(amount * (1 + half), 1000),
            "target_coverage": spec["target_coverage"],
            "measured_coverage": spec["measured_coverage_on_test"],
            "method": spec["method"],
        }

    def _card(self, key: str, amount: Optional[float], unit: str, kind: str, basis: str,
              explanation: str, interval: Optional[Dict] = None, interval_note: str = "") -> Dict:
        limits = SALE_PRICE_LIMITS if unit == "USD" else RENT_PRICE_LIMITS
        ok = amount is not None and is_valid_price(amount, *limits)
        return {
            "key": key,
            "label": LABELS[key],
            "status": "ok" if ok else "unavailable",
            "amount_usd": int(amount) if ok else None,
            "unit": unit,
            "kind": "demo" if self.demo else kind,
            "basis": basis,
            "interval": interval if ok else None,
            "interval_note": interval_note if ok else "No estimate could be made.",
            "explanation": explanation if ok else "The model did not produce a usable number for this location.",
        }

    def estimate(self, lat: float, lon: float, city: Dict, metro_id: Optional[str], use_image: bool = True) -> Dict:
        """Build the estimate block for a house in a supported city."""
        if not self.loaded:
            raise RuntimeError("no models are loaded")
        manifest = self.manifest
        period = manifest["data_period"]
        attributions = ["Home value and rent indexes: Zillow Research (ZHVI and ZORI).",
                        "ZIP center points: US Census Bureau Gazetteer."]
        limitations = [
            "These are estimates of current market prices. They are not appraisals, forecasts or guarantees.",
            "Nothing about the inside of the house is known or used: no bedrooms, bathrooms, floor area or condition.",
            "A photo of the outside cannot show furniture. The furnished numbers add one fixed, assumed furniture package.",
        ]

        # ---- buy price
        mode = "location_only"
        sale_period = "Zillow home values for %s" % period["area_sale_month"]
        if self.has_per_house_model(city):
            sale = self.bundle["seattle_sale"]
            base = float(sale["model"].predict(np.array([[lat, lon, sale["time_index"]]]))[0])
            image_info, factor = self._photo_adjustment(lat, lon, city, use_image)
            spec = sale["interval"]
            if factor is not None:
                base *= factor
                mode = "location_plus_image"
                spec = self.bundle["seattle_photo"]["interval"]
            buy = _round_to(base, 1000) if is_valid_price(base, *SALE_PRICE_LIMITS) else None
            buy_basis = "per_house_model"
            buy_text = ("Predicted for this house from real %s sale records. Inputs: its location%s."
                        % (city["name"], " and a licensed street photo" if factor is not None else ""))
            buy_interval = self._interval(buy, spec) if buy else None
            buy_note = ("About %.0f%% of test houses sold inside a range this wide." % (100 * spec["measured_coverage_on_test"])
                        if buy_interval else NO_RANGE_FEW_ROWS)
            sale_period = "%s sales from %s to %s" % (city["name"], period["seattle_sales_from"], period["seattle_sales_to"])
            attributions.append("Sale records: King County Department of Assessments.")
        else:
            area_sale = self.bundle.get("area_sale")
            base = float(area_sale.predict(np.array([[lat, lon]]))[0]) if area_sale is not None else float("nan")
            buy = _round_to(base, 1000) if is_valid_price(base, *SALE_PRICE_LIMITS) else None
            buy_basis = "area_level_model"
            buy_text = ("Typical single-family home value near this spot, from Zillow ZIP level values. "
                        "It is not specific to this house.")
            buy_interval = None
            buy_note = NO_RANGE_AREA
            if use_image:
                image_info = {"status": "no_photo_model",
                              "message": "No photo model is available for this city, so the visible exterior was not analyzed."}
            else:
                image_info = {"status": "not_requested", "message": "You asked for a location-only estimate."}
            limitations.append("For this city the buy price is an area-level figure, not a value for this specific house.")

        # ---- rent
        area_rent = self.bundle.get("area_rent")
        ratio_info = (manifest.get("metro_rent_ratio") or {}).get(metro_id or "")
        rent = None
        rent_text = "No rent model is available."
        if area_rent is not None:
            raw = float(area_rent.predict(np.array([[lat, lon]]))[0])
            if ratio_info:
                raw *= ratio_info["ratio"]
                rent_text = ("Area-level figure: the Zillow ZIP rent index near this spot (all home types), multiplied by "
                             "%.2f, the single-family to all-homes rent ratio for the %s metro area."
                             % (ratio_info["ratio"], ratio_info["metro"]))
            else:
                rent_text = "Area-level figure: the Zillow ZIP rent index near this spot (all home types)."
            rent = _round_to(raw, 10) if is_valid_price(raw, *RENT_PRICE_LIMITS) else None
        limitations.append("Rent is an area-level figure, not a rent for this specific house.")

        # ---- furnished scenario
        pack = furniture.summary(self.package)
        buy_f = int(round(furniture.furnished_purchase(buy, self.package))) if buy else None
        rent_f = int(round(furniture.furnished_rent(rent, self.package))) if rent else None

        cards = [
            self._card("buy_unfurnished", buy, "USD", "trained", buy_basis, buy_text, buy_interval, buy_note),
            self._card("buy_furnished", buy_f, "USD", "scenario", "assumption_added",
                       "The price without furniture plus $%s, the assumed resale value of the standard furniture package."
                       % format(int(round(pack["resale_value_usd"])), ","), None, NO_RANGE_SCENARIO),
            self._card("rent_unfurnished", rent, "USD/month", "trained", "area_level_model", rent_text, None, NO_RANGE_AREA),
            self._card("rent_furnished", rent_f, "USD/month", "scenario", "assumption_added",
                       "The rent without furniture plus $%s per month, a scenario amount: the package cost spread over %d months. "
                       "It is not a measured furnished-rent premium."
                       % (format(int(round(pack["monthly_amount_usd"])), ","), pack["rent_spread_months"]), None, NO_RANGE_SCENARIO),
        ]
        if image_info["status"] == "used":
            attributions.append("Photo: Mapillary contributor %s, %s." % (image_info.get("creator") or "unknown", image_info["license"]))
        if self.demo:
            limitations.insert(0, "DEMO MODE: these numbers come from models trained on synthetic data. They are not real valuations.")
            attributions = ["Demo mode: synthetic fixture data generated by this project."]

        any_ok = any(c["status"] == "ok" for c in cards)
        return {
            "prediction_mode": mode if any_ok else "none",
            "estimates": cards,
            "image": image_info,
            "furniture": pack,
            "data_period": {"sale": sale_period, "rent": "Zillow rent index for %s" % period["area_rent_month"]},
            "limitations": limitations,
            "attributions": attributions,
            "any_ok": any_ok,
            "estimate_date": date.today().isoformat(),
        }
