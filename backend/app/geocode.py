"""Address lookup (geocoding): turning a typed address into a position.

Two free services are supported, each on one fixed host:
  - OpenStreetMap Nominatim   nominatim.openstreetmap.org
  - US Census Bureau geocoder geocoding.geo.census.gov
The typed address is sent as a query parameter to those services only. The
backend never fetches a link supplied by a user.

Nominatim's usage policy asks for at most one request per second and a
clear User-Agent, and its data is (c) OpenStreetMap contributors (ODbL).
"""
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

from core.geo import tidy_street

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
BENCHMARK = "Public_AR_Current"


class GeocodeError(Exception):
    """The geocoder could not be reached or answered with something unusable."""


@dataclass
class GeocodeResult:
    address: str
    lat: float
    lon: float
    zip: Optional[str]
    city: Optional[str]
    state: Optional[str]
    source: str = "US Census Bureau geocoder"


def format_matched_address(text: str) -> str:
    """'209 N 41ST ST, SEATTLE, WA, 98103' -> '209 N 41st St, Seattle, WA 98103'."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        return text
    street, city, state, zip_code = parts
    return "%s, %s, %s %s" % (tidy_street(street), city.title(), state.upper(), zip_code)


class CensusGeocoder:
    def __init__(self, timeout_s: float = 8.0, client: Optional[httpx.Client] = None):
        self._timeout = timeout_s
        self._client = client

    def lookup(self, address: str) -> Optional[GeocodeResult]:
        """Return the best match for a one-line address, or None when nothing matches."""
        params = {"address": address, "benchmark": BENCHMARK, "format": "json"}
        try:
            if self._client is not None:
                response = self._client.get(CENSUS_URL, params=params, timeout=self._timeout)
            else:
                response = httpx.get(CENSUS_URL, params=params, timeout=self._timeout)
        except httpx.TimeoutException:
            raise GeocodeError("address lookup timed out")
        except httpx.HTTPError:
            raise GeocodeError("address lookup failed")
        if response.status_code != 200:
            raise GeocodeError("address lookup returned status %d" % response.status_code)
        try:
            matches = response.json()["result"]["addressMatches"]
        except (ValueError, KeyError, TypeError):
            raise GeocodeError("address lookup returned an unexpected answer")
        if not matches:
            return None
        best = matches[0]
        try:
            parts = best.get("addressComponents", {})
            return GeocodeResult(
                address=format_matched_address(str(best["matchedAddress"])),
                lat=float(best["coordinates"]["y"]),
                lon=float(best["coordinates"]["x"]),
                zip=parts.get("zip"),
                city=parts.get("city"),
                state=parts.get("state"),
            )
        except (KeyError, TypeError, ValueError):
            raise GeocodeError("address lookup returned an unexpected answer")


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "HomeLens-student-prototype/1.0 (local, low-volume use)"
NOMINATIM_MIN_INTERVAL_S = 1.0


class NominatimGeocoder:
    """OpenStreetMap address lookup. Only house-level matches are accepted."""

    def __init__(self, timeout_s: float = 8.0, client: Optional[httpx.Client] = None,
                 clock=time.monotonic, sleep=time.sleep):
        self._timeout = timeout_s
        self._client = client
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_call = None

    def _wait_for_turn(self) -> None:
        """Keep to the service's limit of one request per second."""
        with self._lock:
            now = self._clock()
            if self._last_call is not None and now - self._last_call < NOMINATIM_MIN_INTERVAL_S:
                self._sleep(NOMINATIM_MIN_INTERVAL_S - (now - self._last_call))
            self._last_call = self._clock()

    def lookup(self, address: str) -> Optional[GeocodeResult]:
        self._wait_for_turn()
        params = {"q": address, "format": "jsonv2", "addressdetails": "1", "limit": "1", "countrycodes": "us"}
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        try:
            if self._client is not None:
                response = self._client.get(NOMINATIM_URL, params=params, headers=headers, timeout=self._timeout)
            else:
                response = httpx.get(NOMINATIM_URL, params=params, headers=headers, timeout=self._timeout)
        except httpx.TimeoutException:
            raise GeocodeError("address lookup timed out")
        except httpx.HTTPError:
            raise GeocodeError("address lookup failed")
        if response.status_code != 200:
            raise GeocodeError("address lookup returned status %d" % response.status_code)
        try:
            matches = response.json()
            if not isinstance(matches, list):
                raise ValueError("not a list")
            if not matches:
                return None
            best = matches[0]
            parts = best.get("address") or {}
            number, road = parts.get("house_number"), parts.get("road")
            if not number or not road:
                return None  # matched a street or an area, not one house
            city = parts.get("city") or parts.get("town") or parts.get("village") or ""
            state = str(parts.get("ISO3166-2-lvl4", "")).replace("US-", "")
            zip_code = str(parts.get("postcode") or "")[:5] or None
            text = "%s %s, %s, %s" % (number, road, city, state)
            return GeocodeResult(address=text.strip(", "), lat=float(best["lat"]), lon=float(best["lon"]),
                                 zip=zip_code, city=city or None, state=state or None,
                                 source="OpenStreetMap Nominatim, data (c) OpenStreetMap contributors")
        except (KeyError, TypeError, ValueError):
            raise GeocodeError("address lookup returned an unexpected answer")


class GeocoderChain:
    """Try several geocoders in order. The first one that finds the address wins."""

    def __init__(self, geocoders: List):
        self._geocoders = geocoders

    def lookup(self, address: str) -> Optional[GeocodeResult]:
        last_error = None
        answered = False
        for geocoder in self._geocoders:
            try:
                found = geocoder.lookup(address)
            except GeocodeError as exc:
                last_error = exc
                continue
            answered = True
            if found is not None:
                return found
        if answered or last_error is None:
            return None            # at least one service answered, and none knows the address
        raise last_error           # every service failed


def build_geocoder(names: List[str], timeout_s: float) -> GeocoderChain:
    known = {"nominatim": NominatimGeocoder, "census": CensusGeocoder}
    return GeocoderChain([known[name](timeout_s=timeout_s) for name in names])
