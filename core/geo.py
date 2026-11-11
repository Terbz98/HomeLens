"""Small geography helpers: distances, bearings, and config loading."""
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.paths import CONFIG_DIR

EARTH_RADIUS_M = 6371000.0


def load_cities(config_dir: Optional[Path] = None) -> List[Dict]:
    """Return the list of supported cities from config/cities.json."""
    path = (config_dir or CONFIG_DIR) / "cities.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["cities"]


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two points. Works on numbers or numpy arrays."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass direction (0 = north, 90 = east) from point 1 toward point 2."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_difference_deg(a: float, b: float) -> float:
    """Smallest difference between two compass directions, from 0 to 180."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def bbox_around(lat: float, lon: float, radius_m: float):
    """Return (min_lon, min_lat, max_lon, max_lat) of a square around a point."""
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def valid_lat_lon(lat, lon) -> bool:
    """True when lat and lon are real numbers inside the valid ranges."""
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat) or math.isnan(lon):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


_ORDINAL = re.compile(r"(\d)(St|Nd|Rd|Th)\b")


def tidy_street(text: str) -> str:
    """'209 N 41ST ST' -> '209 N 41st St'."""
    return _ORDINAL.sub(lambda m: m.group(1) + m.group(2).lower(), str(text).strip().title())
