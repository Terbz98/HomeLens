"""City lookup, parcel lookup, the Census geocoder client, settings and the request limiter."""
import httpx
import pandas as pd
import pytest

from backend.app.cities import Locator, house_number, zip_from_address
from backend.app.config import load_settings
from backend.app.geocode import (CensusGeocoder, GeocodeError, GeocodeResult, GeocoderChain, NominatimGeocoder,
                                 format_matched_address)
from backend.app.ratelimit import RateLimiter
from backend.tests.conftest import BELLEVUE, NEW_YORK, SEATTLE_HOUSE

PARCELS = pd.DataFrame({
    "property_id": ["P1", "P2", "P3"],
    "address": ["209 N 41st St, Seattle, WA 98103", "211 N 41st St, Seattle, WA 98103", "9 Far Away Rd, Seattle, WA 98103"],
    "zip": ["98103", "98103", "98103"],
    "lat": [47.67300, 47.67300, 47.69000],
    "lon": [-122.34300, -122.34320, -122.34300],
})


def test_zip_and_house_number_parsing():
    assert zip_from_address("209 N 41st St, Seattle, WA 98103") == "98103"
    assert zip_from_address("12345 Main St, Portland, OR 97214-1234") == "97214"
    assert zip_from_address("Space Needle") is None and zip_from_address(None) is None
    assert house_number("209 N 41st St") == "209" and house_number("Space Needle") is None


def test_city_lookup(zip_table, cities):
    locator = Locator(zip_table, cities)
    assert locator.locate(SEATTLE_HOUSE["lat"], SEATTLE_HOUSE["lng"])["city"]["id"] == "seattle"
    assert locator.locate(45.515, -122.600)["city"]["id"] == "portland"
    assert locator.locate(37.760, -122.484)["city"]["id"] == "san_francisco"
    bellevue = locator.locate(BELLEVUE["lat"], BELLEVUE["lng"])
    assert bellevue["city"] is None and bellevue["place_name"] == "Bellevue"
    far = locator.locate(NEW_YORK["lat"], NEW_YORK["lng"], zip_hint="98103")
    assert far["city"] is None          # the position wins over a ZIP typed in the address
    hinted = locator.locate(47.6185, -122.2055, zip_hint="98103")
    assert hinted["zip"] == "98103" and hinted["zip_source"] == "address"


def test_nearest_parcels_are_sorted_and_limited_by_distance(zip_table, cities):
    locator = Locator(zip_table, cities, PARCELS)
    near = locator.nearest_parcels(47.67300, -122.34305)
    assert [p["property_id"] for p in near] == ["P1", "P2"]        # P3 is about 1.9 km away
    assert near[0]["distance_m"] < near[1]["distance_m"] < 60
    assert locator.parcel_for_address(47.67300, -122.34305, "211 N 41st St, Seattle")["property_id"] == "P2"
    assert locator.parcel_for_address(47.67300, -122.34305, "999 N 41st St") is None
    assert Locator(zip_table, cities).nearest_parcels(47.673, -122.343) == []


def census(handler):
    return CensusGeocoder(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_geocoder_parses_a_match_and_only_calls_the_census_host():
    seen = {}

    def handler(request):
        seen["host"] = request.url.host
        return httpx.Response(200, json={"result": {"addressMatches": [{
            "matchedAddress": "209 N 41ST ST, SEATTLE, WA, 98103",
            "coordinates": {"x": -122.356, "y": 47.6561},
            "addressComponents": {"zip": "98103", "city": "SEATTLE", "state": "WA"}}]}})

    found = census(handler).lookup("209 n 41st st seattle")
    assert seen["host"] == "geocoding.geo.census.gov"
    assert found.address == "209 N 41st St, Seattle, WA 98103"
    assert (found.lat, found.lon, found.zip) == (47.6561, -122.356, "98103")
    assert format_matched_address("odd text") == "odd text"


def test_geocoder_no_match_timeout_and_bad_answers():
    assert census(lambda r: httpx.Response(200, json={"result": {"addressMatches": []}})).lookup("x y z") is None

    def timeout(request):
        raise httpx.ReadTimeout("slow")

    for handler in (timeout, lambda r: httpx.Response(503), lambda r: httpx.Response(200, json={"oops": 1}),
                    lambda r: httpx.Response(200, content=b"<html>")):
        with pytest.raises(GeocodeError):
            census(handler).lookup("209 N 41st St")


def nominatim(handler, **kwargs):
    return NominatimGeocoder(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


OSM_HOUSE = [{"lat": "37.7563068", "lon": "-122.4924614",
              "address": {"house_number": "1650", "road": "34th Avenue", "city": "San Francisco",
                          "ISO3166-2-lvl4": "US-CA", "postcode": "94166"}}]


def test_nominatim_parses_a_house_and_identifies_itself():
    seen = {}

    def handler(request):
        seen["host"] = request.url.host
        seen["agent"] = request.headers.get("user-agent")
        seen["country"] = request.url.params.get("countrycodes")
        return httpx.Response(200, json=OSM_HOUSE)

    found = nominatim(handler).lookup("1650 34th Ave, San Francisco, CA 94122")
    assert seen == {"host": "nominatim.openstreetmap.org", "agent": seen["agent"], "country": "us"}
    assert "HomeLens" in seen["agent"]
    assert found.address == "1650 34th Avenue, San Francisco, CA"
    assert (found.lat, found.lon) == (37.7563068, -122.4924614)
    assert "OpenStreetMap" in found.source


def test_nominatim_refuses_street_level_matches_and_handles_failures():
    street_only = [{"lat": "37.75", "lon": "-122.49", "address": {"road": "34th Avenue", "city": "San Francisco"}}]
    assert nominatim(lambda r: httpx.Response(200, json=street_only)).lookup("34th Ave") is None
    assert nominatim(lambda r: httpx.Response(200, json=[])).lookup("zzz") is None
    for handler in (lambda r: httpx.Response(429), lambda r: httpx.Response(200, json={"error": "x"}),
                    lambda r: httpx.Response(200, content=b"<html>")):
        with pytest.raises(GeocodeError):
            nominatim(handler).lookup("1650 34th Ave")


def test_nominatim_waits_one_second_between_requests():
    now = [100.0]
    naps = []

    def nap(seconds):
        naps.append(round(seconds, 2))
        now[0] += seconds

    geocoder = nominatim(lambda r: httpx.Response(200, json=[]), clock=lambda: now[0], sleep=nap)
    geocoder.lookup("a b c")
    now[0] += 0.25
    geocoder.lookup("a b c")
    assert naps == [0.75]


class Fixed:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def lookup(self, address):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_geocoder_chain_falls_back_when_a_service_is_down():
    hit = GeocodeResult("1 A St, Seattle, WA", 47.6, -122.3, "98103", "Seattle", "WA")
    down, second = Fixed(GeocodeError("address lookup timed out")), Fixed(hit)
    assert GeocoderChain([down, second]).lookup("1 A St") is hit
    assert (down.calls, second.calls) == (1, 1)
    first, unused = Fixed(hit), Fixed(hit)
    GeocoderChain([first, unused]).lookup("1 A St")
    assert unused.calls == 0                                     # stops at the first match
    assert GeocoderChain([Fixed(None), Fixed(GeocodeError("down"))]).lookup("x") is None    # one service answered: not found
    with pytest.raises(GeocodeError):
        GeocoderChain([Fixed(GeocodeError("down")), Fixed(GeocodeError("down too"))]).lookup("x")


def test_settings():
    s = load_settings({})
    assert s.demo is False and s.cors_origin_regex == r"^chrome-extension://[a-p]{32}$"
    demo = load_settings({"HOMELENS_DEMO": "1", "HOMELENS_EXTENSION_ID": "c" * 32})
    assert demo.demo and demo.parcels_path is None and "synthetic" in demo.zip_table_path.name
    assert demo.cors_origin_regex == "^chrome-extension://%s$" % ("c" * 32)
    with pytest.raises(ValueError):
        load_settings({"HOMELENS_EXTENSION_ID": "not-an-id"})
    for bad in ("*", "https://example.com"):
        with pytest.raises(ValueError):
            load_settings({"HOMELENS_EXTRA_ORIGINS": bad})
    assert load_settings({"HOMELENS_EXTRA_ORIGINS": "http://127.0.0.1:5500"}).extra_origins == ["http://127.0.0.1:5500"]
    assert s.geocoders == ["nominatim", "census"]
    assert load_settings({"HOMELENS_GEOCODERS": "census"}).geocoders == ["census"]
    with pytest.raises(ValueError):
        load_settings({"HOMELENS_GEOCODERS": "https://evil.example.com"})


def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(2, window_s=60, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")                     # other clients are counted separately
    now[0] = 61.0
    assert limiter.allow("a")                     # the window has moved on
