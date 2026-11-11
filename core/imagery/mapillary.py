"""Mapillary street-level photos (https://www.mapillary.com).

Mapillary images are shared by their contributors under CC BY-SA 4.0.
The API needs a free client token, read from the MAPILLARY_TOKEN
environment variable. The token is sent in a header, never in a URL.
"""
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from core.geo import bbox_around
from core.imagery.base import ExteriorImage, ImageryError, ImageryProvider

API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,computed_geometry,geometry,computed_compass_angle,compass_angle,captured_at,creator,is_pano,thumb_1024_url"
LICENSE = "CC BY-SA 4.0"
# Thumbnail links returned by the API point at these hosts. Nothing else is fetched.
ALLOWED_IMAGE_HOST_SUFFIXES = (".fbcdn.net", ".mapillary.com")
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _allowed_image_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and host.endswith(ALLOWED_IMAGE_HOST_SUFFIXES)


def _captured_iso(value) -> Optional[str]:
    """Mapillary gives capture time in milliseconds since 1970."""
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def parse_image(item: dict) -> Optional[ExteriorImage]:
    """Turn one API record into an ExteriorImage. Returns None if it has no position."""
    geometry = item.get("computed_geometry") or item.get("geometry") or {}
    coords = geometry.get("coordinates")
    if not coords or len(coords) < 2 or not item.get("id"):
        return None
    angle = item.get("computed_compass_angle")
    if angle is None:
        angle = item.get("compass_angle")
    creator = item.get("creator") or {}
    image_id = str(item["id"])
    return ExteriorImage(
        provider="mapillary",
        image_id=image_id,
        lat=float(coords[1]),
        lon=float(coords[0]),
        compass_angle=float(angle) if angle is not None else None,
        captured_at=_captured_iso(item.get("captured_at")),
        creator=creator.get("username"),
        license=LICENSE,
        thumb_url=item.get("thumb_1024_url"),
        page_url="https://www.mapillary.com/app/?pKey=%s" % image_id,
        is_pano=bool(item.get("is_pano", False)),
    )


class MapillaryProvider(ImageryProvider):
    name = "mapillary"
    license = LICENSE

    def __init__(self, token: Optional[str], timeout_s: float = 8.0, client: Optional[httpx.Client] = None):
        self._token = (token or "").strip()
        self._timeout = timeout_s
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self._token)

    def _get(self, url: str, **kwargs) -> httpx.Response:
        try:
            if self._client is not None:
                return self._client.get(url, timeout=self._timeout, **kwargs)
            return httpx.get(url, timeout=self._timeout, **kwargs)
        except httpx.HTTPError as exc:
            # The message is kept generic so a token can never leak into logs.
            raise ImageryError("mapillary request failed: %s" % type(exc).__name__)

    def search(self, lat: float, lon: float, radius_m: float) -> List[ExteriorImage]:
        if not self.configured:
            raise ImageryError("mapillary token is not configured")
        west, south, east, north = bbox_around(lat, lon, radius_m)
        params = {
            "fields": FIELDS,
            "bbox": "%.7f,%.7f,%.7f,%.7f" % (west, south, east, north),
            "limit": "200",
        }
        response = self._get(API_URL, params=params, headers={"Authorization": "OAuth %s" % self._token})
        if response.status_code != 200:
            raise ImageryError("mapillary returned status %d" % response.status_code)
        try:
            items = response.json().get("data", [])
        except ValueError:
            raise ImageryError("mapillary returned invalid JSON")
        images = []
        for item in items:
            image = parse_image(item)
            if image is not None:
                images.append(image)
        return images

    def download(self, image: ExteriorImage) -> bytes:
        if not image.thumb_url or not _allowed_image_url(image.thumb_url):
            raise ImageryError("image link is missing or not on an allowed host")
        response = self._get(image.thumb_url)
        if response.status_code != 200:
            raise ImageryError("image download returned status %d" % response.status_code)
        if len(response.content) > MAX_IMAGE_BYTES:
            raise ImageryError("image is larger than the allowed size")
        return response.content
