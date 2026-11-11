"""Feature processing shared by training and by the backend.

Training and prediction must build features the same way, so both import
these functions. The only inputs are things the app knows at prediction
time: latitude, longitude, a date, and (optionally) a photo embedding.
"""
import numpy as np

FEATURE_VERSION = "2"

# Month zero for the time feature. Any fixed month works.
TIME_ORIGIN_YEAR = 2020


EARTH_RADIUS_KM = 6371.0


def to_xyz_km(latlon):
    """Turn [latitude, longitude] rows into [x, y, z] rows in kilometers.

    Each point is placed on a sphere the size of the Earth. The straight-line
    distance between two such points is, for nearby houses, the same as the
    distance over the ground. Raw degrees would not work for "nearest
    neighbor" searches because one degree of longitude is shorter than one
    degree of latitude.
    """
    latlon = np.asarray(latlon, dtype=float)
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    x = EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon)
    y = EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon)
    z = EARTH_RADIUS_KM * np.sin(lat)
    return np.column_stack([x, y, z])


def month_index(year: int, month: int) -> float:
    """Months since January of TIME_ORIGIN_YEAR."""
    return float((year - TIME_ORIGIN_YEAR) * 12 + (month - 1))


def month_index_from_dates(dates) -> np.ndarray:
    """Month index for a pandas datetime Series or DatetimeIndex."""
    return ((dates.dt.year - TIME_ORIGIN_YEAR) * 12 + (dates.dt.month - 1)).to_numpy(dtype=float)


def xyz_only(X):
    """[latitude, longitude, month index] rows to [x, y, z] km. The month is dropped."""
    X = np.asarray(X, dtype=float)
    return to_xyz_km(X[:, :2])


def is_valid_price(value, low: float, high: float) -> bool:
    """A model output is usable only if it is a finite number in a sane range."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v)) and low <= v <= high


# Sanity limits. A prediction outside these is rejected, never displayed.
SALE_PRICE_LIMITS = (50000.0, 30000000.0)
RENT_PRICE_LIMITS = (300.0, 50000.0)
