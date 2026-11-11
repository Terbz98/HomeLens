"""The imagery provider interface.

HomeLens never reads pixels from Google Maps or Street View. Exterior photos
come from a provider whose license allows analysis. Any provider that
implements ImageryProvider can be plugged in.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from core.geo import angle_difference_deg, bearing_deg, haversine_m

# A photo taken farther than this from the house is treated as not showing it.
MAX_MATCH_DISTANCE_M = 40.0
# The camera must point within this many degrees of the house.
MAX_VIEW_ANGLE_DEG = 70.0
# Photos taken closer than this are usually on the lot itself or badly located.
MIN_MATCH_DISTANCE_M = 3.0


@dataclass
class ExteriorImage:
    """One candidate photo plus where it came from (its provenance)."""
    provider: str
    image_id: str
    lat: float
    lon: float
    compass_angle: Optional[float]
    captured_at: Optional[str]
    creator: Optional[str]
    license: str
    thumb_url: Optional[str]
    page_url: Optional[str]
    is_pano: bool = False
    distance_m: Optional[float] = None
    view_angle_diff_deg: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ImageryError(Exception):
    """The provider could not be reached or returned something unusable."""


class ImageryProvider:
    """Base class. Subclasses fill in name, configured, search and download."""

    name = "base"
    license = ""

    @property
    def configured(self) -> bool:
        """True when the provider has what it needs (for example a token)."""
        return False

    def search(self, lat: float, lon: float, radius_m: float) -> List[ExteriorImage]:
        """Return photos taken near a point. No ranking."""
        raise NotImplementedError

    def download(self, image: ExteriorImage) -> bytes:
        """Return the JPEG bytes of a photo."""
        raise NotImplementedError

    def find_best(self, lat: float, lon: float) -> Optional[ExteriorImage]:
        """Return the photo most likely to show the house at (lat, lon), or None."""
        candidates = self.search(lat, lon, MAX_MATCH_DISTANCE_M)
        return pick_best_image(candidates, lat, lon)


def pick_best_image(images: List[ExteriorImage], lat: float, lon: float) -> Optional[ExteriorImage]:
    """Choose the photo that is close to the house and pointed at it.

    Normal photos are preferred over 360 degree panoramas. Among normal
    photos, the score is distance in meters plus half the pointing error in
    degrees, so a close photo aimed away loses to a slightly farther one
    aimed at the house.
    """
    scored = []
    for img in images:
        distance = float(haversine_m(img.lat, img.lon, lat, lon))
        if distance > MAX_MATCH_DISTANCE_M or distance < MIN_MATCH_DISTANCE_M:
            continue
        img.distance_m = round(distance, 1)
        if img.is_pano:
            # A panorama sees every direction, but the house is a small part of it.
            img.view_angle_diff_deg = None
            scored.append((1000.0 + distance, img))
            continue
        if img.compass_angle is None:
            continue
        to_house = bearing_deg(img.lat, img.lon, lat, lon)
        diff = angle_difference_deg(img.compass_angle, to_house)
        if diff > MAX_VIEW_ANGLE_DEG:
            continue
        img.view_angle_diff_deg = round(diff, 1)
        scored.append((distance + 0.5 * diff, img))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0])
    return scored[0][1]
