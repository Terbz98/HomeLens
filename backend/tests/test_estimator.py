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


def build(bundle, provider, package, demo=False):
    return Estimator(bundle, provider, package, demo, embed_fn=lambda data: np.zeros(512, dtype=np.float32))


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


def test_photo_model_that_lost_in_testing_is_not_used_and_no_photo_is_fetched(package):
    provider = FakeProvider()
    est = build(make_bundle(photo_residual=math.log(1.10), serve=False), provider, package)
    result = est.estimate(47.66, -122.34, SEATTLE, "seattle")
    assert result["image"]["status"] == "no_photo_model"
    assert provider.searches == 0
    assert card(result, "buy_unfurnished")["amount_usd"] == 1000000


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
