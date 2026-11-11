"""Exterior imagery providers. Google Maps imagery is deliberately not one of them."""
import os
from typing import Optional

from core.imagery.base import ExteriorImage, ImageryError, ImageryProvider, pick_best_image
from core.imagery.mapillary import MapillaryProvider
from core.imagery.none import NoImageryProvider

__all__ = [
    "ExteriorImage",
    "ImageryError",
    "ImageryProvider",
    "MapillaryProvider",
    "NoImageryProvider",
    "pick_best_image",
    "get_provider",
]


def get_provider(token: Optional[str] = None) -> ImageryProvider:
    """Return the Mapillary provider when a token exists, else the empty provider."""
    token = token if token is not None else os.environ.get("MAPILLARY_TOKEN", "")
    if token and token.strip():
        return MapillaryProvider(token)
    return NoImageryProvider()
