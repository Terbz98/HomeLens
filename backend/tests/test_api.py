"""The API: happy paths, bad requests, unsupported cases, limits, timeouts, demo labeling, CORS."""
import time

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.estimator import Estimator
from backend.app.main import create_app
from backend.tests.conftest import (BELLEVUE, EXTENSION_ORIGIN, NEW_YORK, PORTLAND_HOUSE, SEATTLE_HOUSE,
                                    estimate_body)
from core.imagery import NoImageryProvider
from core.paths import FIXTURES_DIR


def cards(response):
    return {c["key"]: c for c in response.json()["estimates"]}


# ------------------------------------------------------------------ health and cities

def test_health_reports_demo_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "demo"
    assert body["models_loaded"] is True
    assert body["model_version"].startswith("homelens-demo-")
    assert body["imagery_configured"] is False


def test_cities_lists_the_three_supported_cities(client):
    body = client.get("/cities").json()
    assert [c["id"] for c in body["cities"]] == ["seattle", "san_francisco", "portland"]
    assert body["supported_property_type"] == "detached_single_family"
    by_id = {c["id"]: c for c in body["cities"]}
    assert by_id["seattle"]["sale_basis"] == "per_house_model"
    assert by_id["portland"]["sale_basis"] == "area_level_model"


# ------------------------------------------------------------------ resolve

def test_resolve_place_with_address_and_position_needs_no_geocoder(client, geocoder):
    r = client.post("/resolve", json={"address": "12 Test Ave, Seattle, WA 98103", "source": "place", **SEATTLE_HOUSE})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert body["candidates"][0]["supported"] is True
    assert body["candidates"][0]["city_id"] == "seattle"
    assert body["candidates"][0]["match"] == "place"
    assert geocoder.calls == []


def test_resolve_bare_position_always_asks_for_confirmation(client):
    body = client.post("/resolve", json={"source": "street_view", **SEATTLE_HOUSE}).json()
    assert body["status"] == "needs_confirmation"
    assert body["candidates"][0]["match"] == "pin"
    assert body["candidates"][0]["address"] is None
    assert "not tied to one house" in body["message"]


def test_resolve_typed_address_uses_the_geocoder(client, geocoder):
    body = client.post("/resolve", json={"address": "12 test ave seattle wa", "source": "manual"}).json()
    assert geocoder.calls == ["12 test ave seattle wa"]
    assert body["status"] == "ok"
    assert body["candidates"][0]["match"] == "geocoder"
    assert body["candidates"][0]["zip"] == "98103"


def test_resolve_unknown_address_and_geocoder_outage(client, geocoder):
    assert client.post("/resolve", json={"address": "1 Nowhere Rd"}).json()["status"] == "not_found"
    geocoder.fail = True
    body = client.post("/resolve", json={"address": "12 Test Ave, Seattle"}).json()
    assert body["status"] == "not_found"
    assert "did not answer" in body["message"]


def test_resolve_outside_supported_cities(client):
    for point in (BELLEVUE, NEW_YORK):
        body = client.post("/resolve", json={"source": "map_center", **point}).json()
        assert body["status"] == "unsupported_location"
        assert body["candidates"][0]["supported"] is False


# ------------------------------------------------------------------ estimate

def test_estimate_returns_four_labeled_demo_estimates(client):
    r = client.post("/estimate", json=estimate_body(SEATTLE_HOUSE, address="12 Test Ave, Seattle, WA 98103"))
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "location_only"
    assert body["prediction_mode"] == "location_only"
    assert body["is_demo"] is True and body["result_kind"] == "demo"
    assert body["client_request_id"] == "req-1"
    assert body["property"]["city_id"] == "seattle"
    c = cards(r)
    assert list(c) == ["buy_unfurnished", "buy_furnished", "rent_unfurnished", "rent_furnished"]
    assert all(card["status"] == "ok" and card["kind"] == "demo" for card in c.values())
    assert c["buy_unfurnished"]["unit"] == "USD" and c["rent_unfurnished"]["unit"] == "USD/month"
    assert body["limitations"][0].startswith("DEMO MODE")
    assert body["model_version"].startswith("homelens-demo-")
    assert body["image"]["status"] in ("no_photo_model", "not_configured")


def test_furniture_is_added_exactly_and_shown_separately(client):
    r = client.post("/estimate", json=estimate_body(SEATTLE_HOUSE))
    c = cards(r)
    pack = r.json()["furniture"]
    assert pack["source"] == "assumption"
    assert c["buy_furnished"]["amount_usd"] - c["buy_unfurnished"]["amount_usd"] == round(pack["resale_value_usd"])
    assert c["rent_furnished"]["amount_usd"] - c["rent_unfurnished"]["amount_usd"] == round(pack["monthly_amount_usd"])
    assert c["buy_furnished"]["basis"] == "assumption_added"
    assert c["buy_furnished"]["interval"] is None
    assert "assum" in c["buy_furnished"]["explanation"]
    assert "not a measured" in c["rent_furnished"]["explanation"]


def test_area_level_city_has_no_range_and_says_so(client):
    r = client.post("/estimate", json=estimate_body(PORTLAND_HOUSE))
    c = cards(r)
    assert r.json()["property"]["city_id"] == "portland"
    assert c["buy_unfurnished"]["basis"] == "area_level_model"
    assert c["buy_unfurnished"]["interval"] is None
    assert "unavailable" in c["buy_unfurnished"]["interval_note"]
    assert any("area-level" in text for text in r.json()["limitations"])


def test_location_only_request_is_respected(client):
    body = client.post("/estimate", json=estimate_body(SEATTLE_HOUSE, use_image=False)).json()
    assert body["image"]["status"] == "not_requested"
    assert body["prediction_mode"] == "location_only"


def test_unsupported_location_and_property_give_no_numbers(client):
    for point in (BELLEVUE, NEW_YORK):
        body = client.post("/estimate", json=estimate_body(point)).json()
        assert body["status"] == "unsupported_location"
        assert body["estimates"] == [] and body["prediction_mode"] == "none"
    for kind in ("other", "unknown"):
        body = client.post("/estimate", json=estimate_body(SEATTLE_HOUSE, property_type=kind)).json()
        assert body["status"] == "unsupported_property"
        assert body["estimates"] == []


def test_invalid_and_incomplete_requests_get_structured_errors(client):
    bad_bodies = [
        {},                                                            # nothing at all
        {"lat": 47.6, "lng": -122.3},                                  # missing type and request id
        estimate_body(SEATTLE_HOUSE, lat=999),                         # impossible latitude
        estimate_body(SEATTLE_HOUSE, property_type="castle"),          # unknown property type
        estimate_body(SEATTLE_HOUSE, bedrooms=3),                      # extra field is refused
        estimate_body(SEATTLE_HOUSE, zip="9810"),                      # malformed ZIP
        estimate_body(SEATTLE_HOUSE, client_request_id="has spaces"),
    ]
    for body in bad_bodies:
        r = client.post("/estimate", json=body)
        assert r.status_code == 422, body
        error = r.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["fields"] and all("field" in f and "problem" in f for f in error["fields"])
    assert client.post("/resolve", json={"lat": 47.6}).status_code == 422          # lat without lng
    assert client.post("/resolve", json={"source": "manual"}).status_code == 422   # nothing to resolve
    assert client.post("/estimate", content="not json", headers={"content-type": "application/json"}).status_code == 422


def test_unknown_path_and_oversized_body(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    big = estimate_body(SEATTLE_HOUSE, address="x" * 20000)
    r = client.post("/estimate", json=big)
    assert r.status_code == 413 and r.json()["error"]["code"] == "body_too_large"


def test_rate_limit(demo_settings, demo_estimator, geocoder):
    demo_settings.requests_per_minute = 3
    limited = TestClient(create_app(settings=demo_settings, estimator=demo_estimator, geocoder=geocoder))
    codes = [limited.post("/estimate", json=estimate_body(SEATTLE_HOUSE)).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert limited.get("/health").status_code == 200          # reads are not limited


def test_no_models_means_insufficient_data_not_made_up_numbers(tmp_path, package, geocoder):
    settings = Settings(demo=True, model_dir=tmp_path, zip_table_path=FIXTURES_DIR / "synthetic_zip_values.csv",
                        parcels_path=None, requests_per_minute=1000)
    empty = Estimator.from_dir(tmp_path, NoImageryProvider(), package, demo=True)
    c = TestClient(create_app(settings=settings, estimator=empty, geocoder=geocoder))
    assert c.get("/health").json()["models_loaded"] is False
    body = c.post("/estimate", json=estimate_body(SEATTLE_HOUSE)).json()
    assert body["status"] == "insufficient_data"
    assert body["estimates"] == []


def test_slow_estimate_times_out_cleanly(demo_settings, demo_estimator, geocoder, monkeypatch):
    demo_settings.estimate_timeout_s = 0.2
    original = demo_estimator.estimate

    def slow(*args, **kwargs):
        time.sleep(1.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(demo_estimator, "estimate", slow)
    c = TestClient(create_app(settings=demo_settings, estimator=demo_estimator, geocoder=geocoder))
    r = c.post("/estimate", json=estimate_body(SEATTLE_HOUSE))
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"


def test_unexpected_error_hides_details(demo_settings, demo_estimator, geocoder, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret internal detail at 12 Test Ave")

    monkeypatch.setattr(demo_estimator, "estimate", broken)
    c = TestClient(create_app(settings=demo_settings, estimator=demo_estimator, geocoder=geocoder),
                   raise_server_exceptions=False)
    r = c.post("/estimate", json=estimate_body(SEATTLE_HOUSE))
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret" not in r.text and "Test Ave" not in r.text


# ------------------------------------------------------------------ cross-origin access

def test_cors_allows_the_extension_and_refuses_websites(client):
    ok = client.get("/health", headers={"Origin": EXTENSION_ORIGIN})
    assert ok.headers.get("access-control-allow-origin") == EXTENSION_ORIGIN
    for origin in ("https://evil.example.com", "https://www.google.com", "null"):
        r = client.get("/health", headers={"Origin": origin})
        assert "access-control-allow-origin" not in r.headers


def test_cors_can_be_pinned_to_one_extension_id(demo_settings, demo_estimator, geocoder):
    demo_settings.extension_id = "b" * 32
    c = TestClient(create_app(settings=demo_settings, estimator=demo_estimator, geocoder=geocoder))
    assert "access-control-allow-origin" not in c.get("/health", headers={"Origin": EXTENSION_ORIGIN}).headers
    pinned = "chrome-extension://" + "b" * 32
    assert c.get("/health", headers={"Origin": pinned}).headers["access-control-allow-origin"] == pinned
