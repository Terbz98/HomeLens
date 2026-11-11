"""Work out which supported city (if any) a point or ZIP belongs to, and find nearby parcels."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.features import to_xyz_km
from core.geo import haversine_m

# A point farther than this from every known ZIP center is outside the covered metro areas.
MAX_KM_TO_ZIP_CENTER = 15.0
MAX_PARCEL_DISTANCE_M = 60.0
ZIP_IN_TEXT = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def zip_from_address(address: Optional[str]) -> Optional[str]:
    """Pick the ZIP code out of an address string. The last 5 digit group wins."""
    if not address:
        return None
    found = ZIP_IN_TEXT.findall(address)
    return found[-1] if found else None


def house_number(address: Optional[str]) -> Optional[str]:
    """Leading street number of an address, for example '209' in '209 N 41st St'."""
    if not address:
        return None
    m = re.match(r"\s*(\d+)[A-Za-z]?\b", address)
    return m.group(1) if m else None


class Locator:
    """Answers 'which city is this?' and 'which parcels are near this point?'."""

    def __init__(self, zip_table: pd.DataFrame, cities: List[Dict], parcels: Optional[pd.DataFrame] = None):
        self.cities = {c["id"]: c for c in cities}
        table = zip_table.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        self._zips = table
        self._zip_city = dict(zip(table["zip"], table["supported_city_id"].fillna("")))
        self._zip_metro = dict(zip(table["zip"], table["metro_id"]))
        self._zip_place = dict(zip(table["zip"], table["city"]))
        self._zip_tree = cKDTree(to_xyz_km(table[["lat", "lon"]].to_numpy(dtype=float)))
        self._parcels = None
        self._parcel_tree = None
        if parcels is not None and len(parcels):
            self._parcels = parcels.reset_index(drop=True)
            self._parcel_tree = cKDTree(to_xyz_km(self._parcels[["lat", "lon"]].to_numpy(dtype=float)))

    @classmethod
    def from_files(cls, zip_table_path: Path, cities: List[Dict], parcels_path: Optional[Path]) -> "Locator":
        zips = pd.read_csv(zip_table_path, dtype={"zip": str}, keep_default_na=False,
                           na_values={"lat": [""], "lon": [""], "zhvi_usd": [""], "zori_usd_month": [""]})
        parcels = None
        if parcels_path is not None and Path(parcels_path).exists():
            parcels = pd.read_csv(parcels_path, dtype={"zip": str, "property_id": str})
        return cls(zips, cities, parcels)

    @property
    def has_parcels(self) -> bool:
        return self._parcel_tree is not None

    def nearest_zip(self, lat: float, lon: float) -> Tuple[Optional[str], float]:
        """ZIP whose center point is nearest, and the distance to it in km.

        This is an approximation: ZIP areas are not circles. It is only used
        when no ZIP is known from an address.
        """
        dist, idx = self._zip_tree.query(to_xyz_km(np.array([[lat, lon]]))[0])
        if not np.isfinite(dist) or dist > MAX_KM_TO_ZIP_CENTER:
            return None, float(dist)
        return str(self._zips.loc[int(idx), "zip"]), float(dist)

    def locate(self, lat: float, lon: float, zip_hint: Optional[str] = None) -> Dict:
        """Return the ZIP, city and metro for a point.

        Keys: zip, zip_source ('address' or 'nearest_center'), city (config dict or None),
        metro_id, place_name.
        """
        nearest, _ = self.nearest_zip(lat, lon)
        if nearest is None:
            # The point is far from every covered ZIP, whatever the address text says.
            return {"zip": zip_hint, "zip_source": "none", "city": None, "metro_id": None, "place_name": None}
        if zip_hint and zip_hint in self._zip_city:
            zip_code, source = zip_hint, "address"
        else:
            zip_code, source = nearest, "nearest_center"
        city_id = self._zip_city.get(zip_code, "")
        return {
            "zip": zip_code,
            "zip_source": source,
            "city": self.cities.get(city_id),
            "metro_id": self._zip_metro.get(zip_code),
            "place_name": self._zip_place.get(zip_code),
        }

    def nearest_parcels(self, lat: float, lon: float, count: int = 3) -> List[Dict]:
        """Up to `count` known single-family parcels within MAX_PARCEL_DISTANCE_M, nearest first."""
        if self._parcel_tree is None:
            return []
        k = min(count, len(self._parcels))
        _, idx = self._parcel_tree.query(to_xyz_km(np.array([[lat, lon]]))[0], k=k)
        out = []
        for i in np.atleast_1d(idx):
            row = self._parcels.loc[int(i)]
            distance = float(haversine_m(lat, lon, row["lat"], row["lon"]))
            if distance <= MAX_PARCEL_DISTANCE_M:
                out.append({
                    "property_id": str(row["property_id"]),
                    "address": str(row["address"]),
                    "zip": str(row["zip"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "distance_m": round(distance, 1),
                })
        return sorted(out, key=lambda parcel: parcel["distance_m"])

    def parcel_for_address(self, lat: float, lon: float, address: Optional[str]) -> Optional[Dict]:
        """The nearby parcel whose street number matches the address, if there is exactly that match."""
        number = house_number(address)
        if number is None:
            return None
        for parcel in self.nearest_parcels(lat, lon, count=6):
            if house_number(parcel["address"]) == number:
                return parcel
        return None
