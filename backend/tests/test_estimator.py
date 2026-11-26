"""The estimator: photo handling, fallbacks, rejected outputs, demo and real never mixed."""
import math

import numpy as np
import pytest

from backend.app.estimator import Estimator
from backend.tests.conftest import Constant
from core.imagery import ExteriorImage, ImageryError, ImageryProvider, NoImageryProvider

SEATTLE = {"id": "seattle", "name": "Seattle", "state": "WA", "per_house_sales": True}
PORTLAND = {"id": "portland", "name": "Portland", "state": "OR", "per_house_sales": False}
INTERVAL = {"method": "split conformal on relative error", "target_coverage": 0.8, "relative_half_width": 0.25,
            "n_calibration": 500, "n_test": 1000, "measured_coverage_on_test": 0.81}
PHOTO_INTERVAL = dict(INTERVAL, relative_half_width=0.20, measured_coverage_on_test=0.79)


def make_bundle(sale=1000000.0, area_sale=800000.0, area_rent=2000.0, photo_residual=None, serve=True,
                synthetic=False, interval=INTERVAL):
    photo = None
    if photo_residual is not None:
        photo = {"residual_model": Constant(photo_residual), "interval": PHOTO_INTERVAL, "serve": serve}
    return {
        "area_sale": Constant(area_sale), "area_rent": Constant(area_rent),
        "seattle_sale": {"model": Constant(sale), "time_index": 80.0, "interval": interval},
        "seattle_photo": photo,
        "manifest": {
            "model_version": "homelens-test-1", "synthetic_training_data": synthetic,
            "data_period": {"area_sale_month": "2026-09", "area_rent_month": "2026-09",
                            "seattle_sales_from": "2023-11-05", "seattle_sales_to": "2026-11-04"},
            "metro_rent_ratio": {"seattle": {"metro": "Seattle, WA", "ratio": 1.5},
                                 "portland": {"metro": "Portland, OR", "ratio": 1.5}},
        },
    }


class FakeProvider(ImageryProvider):
    name = "fake"
    license = "CC BY-SA 4.0"

    def __init__(self, image="default", fail_search=False, fail_download=False, configured=True):
        self._configured = configured
        self.fail_search = fail_search
        self.fail_download = fail_download
        self.image = image
        self.searches = 0

    @property
    def configured(self):
        return self._configured

    def find_best(self, lat, lon):
        self.searches += 1
        if self.fail_search:
            raise ImageryError("provider down")
        if self.image == "default":
            return ExteriorImage(provider="fake", image_id="img-1", lat=lat, lon=lon, compass_angle=90.0,
                                 captured_at="2025-05-01", creator="mapper", license="CC BY-SA 4.0",
                                 thumb_url="https://x.fbcdn.net/a.jpg", page_url="https://example.org/img-1",
                                 distance_m=14.0)
        return self.image

    def download(self, image):
        if self.fail_download:
            raise ImageryError("download failed")
        return b"jpeg-bytes"


def build(bundle, provider, package, demo=False, photo_mode="on", house_table=None):
    return Estimator(bundle, provider, package, demo, embed_fn=lambda data, encoder="resnet18": np.zeros(512, dtype=np.float32),
                     house_table=house_table, photo_mode=photo_mode)


def card(result, key):
    return [c for c in result["estimates"] if c["key"] == key][0]


def test_photo_is_used_when_everything_is_available(package):
    est = build(make_bundle(photo_residual=math.log(1.10)), FakeProvider(), package)
    result = est.estimate(47.66, -122.34, SEATTLE, "seattle")
    assert result["prediction_mode"] == "location_plus_image"
    assert result["image"]["status"] == "used"
    assert result["image"]["image_id"] == "img-1" and result["image"]["license"] == "CC BY-SA 4.0"
    buy = card(result, "buy_unfurnished")
    assert buy["amount_usd"] == 1100000                       # location price 1,000,000 times photo factor 1.10
    assert buy["interval"]["low_usd"] == 880000 and buy["interval"]["high_usd"] == 1320000   # photo range, 20 percent
    assert buy["interval"]["measured_coverage"] == 0.79
    assert any("Mapillary" in a or "Photo" in a for a in result["attributions"])


def test_san_francisco_photo_adjusts_only_a_house_estimate(package):
    import pandas as pd

    facts = {"living_sqft": 1600, "year_built": 1920, "rooms": 6, "bedrooms": 3, "bathrooms": 2}
    sf = {
        "house": {"model": Constant(1_500_000), "feature_cols": list(facts), "time_index": 80, "interval": INTERVAL},
        "photo": {"residual_model": Constant(math.log(1.1)), "interval": PHOTO_INTERVAL,
                  "serve": True, "base": "location_plus_house_record"},
        "photo_report": {},
        "features": pd.DataFrame([facts], index=["sf-0001001"]),
        "sales_from": "2024-01-01", "sales_to": "2025-12-31", "price_source": "Test sales",
    }
    city = {"id": "san_francisco", "name": "San Francisco"}
    estimator = Estimator(make_bundle(), FakeProvider(), package, False,
                          embed_fn=lambda data, encoder: np.zeros(512), san_francisco=sf)
    with_photo = estimator.estimate(37.76, -122.48, city, "san_francisco", property_id="sf-0001001")
    assert with_photo["prediction_mode"] == "location_plus_house_record_plus_image"
    assert card(with_photo, "buy_unfurnished")["amount_usd"] == 1_650_000

    no_record = estimator.estimate(37.76, -122.48, city, "san_francisco")
    assert no_record["image"]["status"] == "no_photo_model"
    assert card(no_record, "buy_unfurnished")["basis"] == "area_level_model"


@pytest.mark.parametrize("provider,expected", [
    (FakeProvider(image=None), "unavailable"),
    (FakeProvider(fail_search=True), "error"),
    (FakeProvider(fail_download=True), "error"),
    (FakeProvider(configured=False), "not_configured"),
    (NoImageryProvider(), "not_configured"),
])
def test_missing_or_failing_imagery_falls_back_to_location_only(package, provider, expected):
    est = build(make_bundle(photo_residual=math.log(1.10)), provider, package)
    result = est.estimate(47.66, -122.34, SEATTLE, "seattle")
    assert result["image"]["status"] == expected
    assert result["prediction_mode"] == "location_only"
    buy = card(result, "buy_unfurnished")
    assert buy["amount_usd"] == 1000000                       # the photo factor was not applied
    assert buy["interval"]["low_usd"] == 750000               # the location-only range, 25 percent


@pytest.mark.parametrize("mode", ["auto", "off"])
def test_photo_model_that_lost_in_testing_is_not_used_in_auto_or_off_mode(package, mode):
    provider = FakeProvider()
    est = build(make_bundle(photo_residual=math.log(1.10), serve=False), provider, package, photo_mode=mode)
    result = est.estimate(47.66, -122.34, SEATTLE, "seattle")
    assert result["image"]["status"] == "no_photo_model"
    assert provider.searches == 0
    assert card(result, "buy_unfurnished")["amount_usd"] == 1000000


def test_photo_model_is_used_in_on_mode_and_says_what_it_did_in_testing(package):
    bundle = make_bundle(photo_residual=math.log(1.10), serve=False)
    bundle["manifest"]["metrics"] = {"seattle_photo": {"test_comparison": {
        "mae_lowered_by_usd": 200.0, "verdict": "no measurable difference"}}}
    result = build(bundle, FakeProvider(), package, photo_mode="on").estimate(47.66, -122.34, SEATTLE, "seattle")
    assert result["image"]["status"] == "used"
    assert "lowered the average error by $200 (no measurable difference)" in result["image"]["message"]
    assert card(result, "buy_unfurnished")["amount_usd"] == 1100000


RECORD = {"living_sqft": 2000.0, "grade": 8.0, "year_built": 1925.0, "year_renovated": 0.0, "condition": 3.0,
          "bedrooms": 3.0, "bathrooms": 2.0, "stories": 2.0, "basement_sqft": 0.0, "buildings_on_parcel": 1.0,
          "lot_sqft": 5000.0, "view_score": 0.0, "waterfront": 0.0}


class RowEcho:
    """A fake model that returns the living area times 600, so the test can see which inputs it got."""
    def predict(self, X):
        return np.array([600.0 * X[0][3]])


def with_house_model(bundle, base="location_plus_house_record"):
    import pandas as pd
    from core.house_features import FEATURE_COLUMNS
    bundle["seattle_house"] = {"model": RowEcho(), "time_index": 80.0, "interval": INTERVAL,
                               "feature_cols": FEATURE_COLUMNS}
    if bundle.get("seattle_photo"):
        bundle["seattle_photo"]["base"] = base
    table = pd.DataFrame([dict(RECORD, property_id="1234567890")]).set_index("property_id")
    return bundle, table


def test_county_record_is_used_when_the_parcel_is_known(package):
    bundle, table = with_house_model(make_bundle())
    est = build(bundle, FakeProvider(), package, house_table=table)
    result = est.estimate(47.66, -122.34, SEATTLE, "seattle", property_id="1234567890")
    buy = card(result, "buy_unfurnished")
    assert buy["amount_usd"] == 1200000                      # 2,000 sq ft times 600
    assert "2,000 sq ft living area" in buy["explanation"] and "built 1925" in buy["explanation"]
    assert result["prediction_mode"] == "location_plus_house_record"


def test_unknown_parcel_falls_back_to_location_only_and_skips_the_house_photo_model(package):
    bundle, table = with_house_model(make_bundle(photo_residual=math.log(1.10)))
    provider = FakeProvider()
    result = build(bundle, provider, package, house_table=table).estimate(
        47.66, -122.34, SEATTLE, "seattle", property_id="0000000000")
    assert card(result, "buy_unfurnished")["amount_usd"] == 1000000      # the location-only model
    assert result["image"]["status"] == "no_photo_model" and provider.searches == 0
    assert result["prediction_mode"] == "location_only"


def test_photo_on_top_of_the_county_record(package):
    bundle, table = with_house_model(make_bundle(photo_residual=math.log(1.10)))
    result = build(bundle, FakeProvider(), package, house_table=table).estimate(
        47.66, -122.34, SEATTLE, "seattle", property_id="1234567890")
    assert result["prediction_mode"] == "location_plus_house_record_plus_image"
    assert card(result, "buy_unfurnished")["amount_usd"] == 1320000      # 1,200,000 times 1.10


def test_user_can_ask_for_location_only(package):
    provider = FakeProvider()
    est = build(make_bundle(photo_residual=math.log(1.10)), provider, package)
    result = est.estimate(47.66, -122.34, SEATTLE, "seattle", use_image=False)
    assert result["image"]["status"] == "not_requested" and provider.searches == 0


def test_absurd_photo_factor_is_rejected(package):
    est = build(make_bundle(photo_residual=math.log(50.0)), FakeProvider(), package)
    result = est.estimate(47.66, -122.34, SEATTLE, "seattle")
    assert result["image"]["status"] == "error"
    assert card(result, "buy_unfurnished")["amount_usd"] == 1000000


@pytest.mark.parametrize("bad", [-5.0, 0.0, float("nan"), float("inf"), 9e12])
def test_impossible_model_outputs_are_never_displayed(package, bad):
    est = build(make_bundle(sale=bad, area_rent=bad), NoImageryProvider(), package)
    result = est.estimate(47.66, -122.34, SEATTLE, "seattle")
    for key in ("buy_unfurnished", "buy_furnished", "rent_unfurnished", "rent_furnished"):
        c = card(result, key)
        assert c["status"] == "unavailable" and c["amount_usd"] is None and c["interval"] is None
    assert result["any_ok"] is False and result["prediction_mode"] == "none"


def test_rent_uses_the_metro_single_family_ratio(package):
    est = build(make_bundle(), NoImageryProvider(), package)
    result = est.estimate(45.51, -122.60, PORTLAND, "portland")
    rent = card(result, "rent_unfurnished")
    assert rent["amount_usd"] == 3000                         # 2,000 times 1.5
    assert rent["basis"] == "area_level_model" and rent["interval"] is None
    assert "1.50" in rent["explanation"]
    assert card(result, "buy_unfurnished")["amount_usd"] == 800000
    assert card(result, "buy_unfurnished")["basis"] == "area_level_model"


def test_no_range_when_calibration_was_not_possible(package):
    est = build(make_bundle(interval=None), NoImageryProvider(), package)
    buy = card(est.estimate(47.66, -122.34, SEATTLE, "seattle"), "buy_unfurnished")
    assert buy["interval"] is None and "unavailable" in buy["interval_note"]


def test_synthetic_models_cannot_be_served_as_real_or_the_reverse(package):
    with pytest.raises(ValueError):
        build(make_bundle(synthetic=True), NoImageryProvider(), package, demo=False)
    with pytest.raises(ValueError):
        build(make_bundle(synthetic=False), NoImageryProvider(), package, demo=True)
    demo = build(make_bundle(synthetic=True), NoImageryProvider(), package, demo=True)
    result = demo.estimate(47.66, -122.34, SEATTLE, "seattle")
    assert all(c["kind"] == "demo" for c in result["estimates"])
    assert result["limitations"][0].startswith("DEMO MODE")
