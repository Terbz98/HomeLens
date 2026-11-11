"""Furniture math, price sanity limits, photo selection and the Mapillary provider (no network)."""
import httpx
import pytest

from core import furniture
from core.features import RENT_PRICE_LIMITS, SALE_PRICE_LIMITS, is_valid_price
from core.geo import angle_difference_deg, bearing_deg, tidy_street
from core.imagery import ExteriorImage, ImageryError, MapillaryProvider, NoImageryProvider, get_provider, pick_best_image
from core.imagery.mapillary import parse_image

PACKAGE = {
    "package_name": "Test", "package_version": "t", "condition": "used", "valuation_basis": "x",
    "fixed_inventory_note": "x", "rent_basis": "x", "resale_fraction": 0.5, "rent_spread_months": 10,
    "items": [{"name": "Sofa", "quantity": 1, "assumed_new_price_usd": 800},
              {"name": "Chair", "quantity": 4, "assumed_new_price_usd": 50}],
}


def test_furniture_amounts():
    assert furniture.new_cost_usd(PACKAGE) == 1000.0             # 800 + 4 * 50
    assert furniture.resale_value_usd(PACKAGE) == 500.0          # half of 1000
    assert furniture.monthly_amount_usd(PACKAGE) == 100.0        # 1000 spread over 10 months
    assert furniture.furnished_purchase(900000.0, PACKAGE) == 900500.0
    assert furniture.furnished_rent(3000.0, PACKAGE) == 3100.0
    assert furniture.summary(PACKAGE)["source"] == "assumption"


def test_real_furniture_config_is_valid_and_labeled_as_assumption():
    package = furniture.load_package()
    assert "assumption" in package["_comment"].lower()
    assert furniture.resale_value_usd(package) == round(furniture.new_cost_usd(package) * package["resale_fraction"], 2)


def test_bad_furniture_config_is_rejected():
    bad = dict(PACKAGE, resale_fraction=1.5)
    with pytest.raises(ValueError):
        furniture.validate_package(bad)
    with pytest.raises(ValueError):
        furniture.validate_package(dict(PACKAGE, items=[]))


@pytest.mark.parametrize("value,ok", [(900000, True), (-5, False), (0, False), (float("nan"), False),
                                      (float("inf"), False), (None, False), ("abc", False), (9e9, False)])
def test_sale_price_sanity(value, ok):
    assert is_valid_price(value, *SALE_PRICE_LIMITS) is ok


def test_rent_sanity():
    assert is_valid_price(2500, *RENT_PRICE_LIMITS)
    assert not is_valid_price(20, *RENT_PRICE_LIMITS)


def test_bearing_and_angles():
    assert bearing_deg(47.0, -122.0, 48.0, -122.0) == pytest.approx(0.0, abs=0.01)     # due north
    assert bearing_deg(47.0, -122.0, 47.0, -121.0) == pytest.approx(90.0, abs=1.0)     # due east
    assert angle_difference_deg(350.0, 10.0) == 20.0
    assert tidy_street("209 N 41ST ST") == "209 N 41st St"


def photo(image_id, lat, lon, angle, pano=False):
    return ExteriorImage(provider="test", image_id=image_id, lat=lat, lon=lon, compass_angle=angle,
                         captured_at=None, creator="someone", license="CC BY-SA 4.0", thumb_url=None,
                         page_url=None, is_pano=pano)


HOUSE = (47.65600, -122.35600)


def test_photo_pointing_at_the_house_wins_over_a_closer_one_pointing_away():
    north_of_house = (47.65615, -122.35600)       # about 17 m north of the house
    facing_house = photo("good", north_of_house[0], north_of_house[1], 180.0)     # looking south, at the house
    facing_away = photo("bad", 47.65610, -122.35600, 0.0)                         # closer, looking north
    best = pick_best_image([facing_away, facing_house], *HOUSE)
    assert best.image_id == "good"
    assert best.distance_m == pytest.approx(16.7, abs=1.0)
    assert best.view_angle_diff_deg == pytest.approx(0.0, abs=1.0)


def test_no_photo_when_everything_is_too_far_or_aimed_away():
    far = photo("far", 47.66000, -122.35600, 180.0)          # about 445 m away
    away = photo("away", 47.65615, -122.35600, 0.0)
    assert pick_best_image([far, away], *HOUSE) is None
    assert pick_best_image([], *HOUSE) is None


def test_normal_photo_is_preferred_over_panorama():
    pano = photo("pano", 47.65608, -122.35600, None, pano=True)
    normal = photo("normal", 47.65620, -122.35600, 180.0)
    assert pick_best_image([pano, normal], *HOUSE).image_id == "normal"
    assert pick_best_image([pano], *HOUSE).image_id == "pano"


def test_parse_mapillary_record():
    item = {"id": "123", "computed_geometry": {"type": "Point", "coordinates": [-122.356, 47.6561]},
            "compass_angle": 10.0, "computed_compass_angle": 95.5, "captured_at": 1717200000000,
            "creator": {"username": "mapper", "id": "9"}, "is_pano": False, "thumb_1024_url": "https://x.fbcdn.net/a.jpg"}
    image = parse_image(item)
    assert (image.lat, image.lon) == (47.6561, -122.356)
    assert image.compass_angle == 95.5                 # the corrected angle is preferred
    assert image.captured_at == "2024-06-01"
    assert image.creator == "mapper"
    assert image.license == "CC BY-SA 4.0"
    assert parse_image({"id": "1"}) is None            # no position, unusable


def fake_mapillary(handler):
    return MapillaryProvider("test-token", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_mapillary_search_sends_token_in_header_not_in_url():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [
            {"id": "1", "geometry": {"coordinates": [-122.356, 47.65615]}, "compass_angle": 180.0,
             "thumb_1024_url": "https://scontent.xx.fbcdn.net/a.jpg"}]})

    provider = fake_mapillary(handler)
    best = provider.find_best(*HOUSE)
    assert best.image_id == "1"
    assert "test-token" not in seen["url"]
    assert seen["auth"] == "OAuth test-token"
    assert "bbox=" in seen["url"]


def test_mapillary_errors_become_imagery_errors():
    with pytest.raises(ImageryError):
        fake_mapillary(lambda request: httpx.Response(500)).search(HOUSE[0], HOUSE[1], 40)

    def boom(request):
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(ImageryError):
        fake_mapillary(boom).search(HOUSE[0], HOUSE[1], 40)


def test_download_refuses_links_outside_the_allowed_hosts():
    provider = fake_mapillary(lambda request: httpx.Response(200, content=b"jpeg"))
    evil = photo("x", 47.0, -122.0, 0.0)
    evil.thumb_url = "https://evil.example.com/a.jpg"
    with pytest.raises(ImageryError):
        provider.download(evil)
    ok = photo("y", 47.0, -122.0, 0.0)
    ok.thumb_url = "https://scontent.xx.fbcdn.net/a.jpg"
    assert provider.download(ok) == b"jpeg"


def test_provider_choice_depends_on_token():
    assert isinstance(get_provider(""), NoImageryProvider)
    assert not get_provider("").configured
    assert get_provider("").find_best(*HOUSE) is None
    assert isinstance(get_provider("abc"), MapillaryProvider)
    assert get_provider("abc").configured


def test_fetch_images_records_provenance_and_coverage(tmp_path):
    import pandas as pd

    from pipeline import fetch_images

    class OneHouseProvider:
        configured = True

        def find_best(self, lat, lon):
            if lat > 47.60:
                image = photo("img-%0.3f" % lat, lat + 0.0001, lon, 180.0)
                image.distance_m = 11.0
                image.view_angle_diff_deg = 4.0
                image.thumb_url = "https://x.fbcdn.net/a.jpg"
                image.captured_at = "2025-05-01"
                return image
            return None

        def download(self, image):
            return b"jpeg-bytes"

    targets = [
        {"kind": "house", "property_id": "A", "zip": "98103", "city_id": "seattle", "lat": 47.66, "lon": -122.34},
        {"kind": "house", "property_id": "B", "zip": "98118", "city_id": "seattle", "lat": 47.54, "lon": -122.27},
    ]
    log_path = tmp_path / "log.csv"
    log = fetch_images.run(OneHouseProvider(), targets, log_path, tmp_path / "images" / "mapillary")
    found = log[log["status"] == "found"].iloc[0]
    assert found["property_id"] == "A"
    assert found["image_license"] == "CC BY-SA 4.0" and found["image_creator"] == "someone"
    assert found["image_captured_at"] == "2025-05-01" and float(found["image_distance_m"]) == 11.0
    assert (tmp_path / "images" / found["file"]).read_bytes() == b"jpeg-bytes"
    assert fetch_images.coverage(log)["house"]["seattle"] == {"searched": 2, "with_photo": 1, "share": 0.5}
    again = fetch_images.run(OneHouseProvider(), targets, log_path, tmp_path / "images" / "mapillary")
    assert len(again) == 2                       # nothing is searched twice
    spots = fetch_images.area_targets(pd.DataFrame({"zip": ["98103"], "supported_city_id": ["seattle"],
                                                    "lat": [47.67], "lon": [-122.34]}), 5)
    assert len(spots) == 5 and all(s["kind"] == "area_point" for s in spots)
