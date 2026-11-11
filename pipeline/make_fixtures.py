"""Create the small SYNTHETIC fixture dataset in data/fixtures.

Usage:
    python -m pipeline.make_fixtures

Every number written here comes from a made-up formula plus random noise
with a fixed seed. Nothing is market data. The fixtures exist so the tests
and the demo mode can run without downloading anything. ZIP codes and
their rough center points are real places so that city checks work, but
the values attached to them are invented.
"""
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

from core.paths import FIXTURES_DIR
from pipeline import schema

SEED = 20261111

# (zip, city, state, metro, metro_id, supported_city_id, lat, lon)
FIXTURE_ZIPS = [
    ("98103", "Seattle", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "seattle", 47.673, -122.343),
    ("98105", "Seattle", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "seattle", 47.661, -122.283),
    ("98107", "Seattle", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "seattle", 47.668, -122.378),
    ("98115", "Seattle", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "seattle", 47.685, -122.282),
    ("98116", "Seattle", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "seattle", 47.574, -122.395),
    ("98117", "Seattle", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "seattle", 47.688, -122.381),
    ("98118", "Seattle", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "seattle", 47.542, -122.269),
    ("98125", "Seattle", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "seattle", 47.717, -122.300),
    ("98004", "Bellevue", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "", 47.618, -122.205),
    ("98052", "Redmond", "WA", "Seattle-Tacoma-Bellevue, WA", "seattle", "", 47.681, -122.121),
    ("94110", "San Francisco", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "san_francisco", 37.750, -122.415),
    ("94112", "San Francisco", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "san_francisco", 37.720, -122.443),
    ("94116", "San Francisco", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "san_francisco", 37.744, -122.486),
    ("94121", "San Francisco", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "san_francisco", 37.778, -122.494),
    ("94122", "San Francisco", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "san_francisco", 37.760, -122.484),
    ("94127", "San Francisco", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "san_francisco", 37.736, -122.459),
    ("94131", "San Francisco", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "san_francisco", 37.745, -122.442),
    ("94134", "San Francisco", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "san_francisco", 37.721, -122.413),
    ("94608", "Emeryville", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "", 37.836, -122.286),
    ("94611", "Oakland", "CA", "San Francisco-Oakland-Berkeley, CA", "san_francisco", "", 37.830, -122.220),
    ("97202", "Portland", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "portland", 45.482, -122.643),
    ("97206", "Portland", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "portland", 45.482, -122.598),
    ("97211", "Portland", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "portland", 45.581, -122.637),
    ("97212", "Portland", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "portland", 45.544, -122.644),
    ("97213", "Portland", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "portland", 45.538, -122.600),
    ("97214", "Portland", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "portland", 45.514, -122.643),
    ("97215", "Portland", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "portland", 45.515, -122.600),
    ("97217", "Portland", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "portland", 45.601, -122.701),
    ("97005", "Beaverton", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "", 45.491, -122.804),
    ("97222", "Milwaukie", "OR", "Portland-Vancouver-Hillsboro, OR-WA", "portland", "", 45.441, -122.618),
]

SYNTHETIC_BASE_VALUE = {"seattle": 800000.0, "san_francisco": 1300000.0, "portland": 500000.0}
SYNTHETIC_BASE_RENT = {"seattle": 2300.0, "san_francisco": 3300.0, "portland": 1800.0}
N_SYNTHETIC_SALES = 400


def make_zip_table(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    centers = {}
    for metro_id in SYNTHETIC_BASE_VALUE:
        points = [(z[6], z[7]) for z in FIXTURE_ZIPS if z[4] == metro_id]
        centers[metro_id] = (float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points])))
    for zip_code, city, state, metro, metro_id, supported, lat, lon in FIXTURE_ZIPS:
        # Invented pattern: values rise a little toward the north and west of each metro.
        lat0, lon0 = centers[metro_id]
        tilt = float(np.clip(1.0 + 2.0 * (lat - lat0) - 1.5 * (lon - lon0), 0.6, 1.6))
        noise = rng.normal(1.0, 0.05)
        rows.append({
            "zip": zip_code, "city": city, "state": state, "metro": metro, "metro_id": metro_id,
            "supported_city_id": supported, "lat": lat, "lon": lon,
            "zhvi_usd": round(SYNTHETIC_BASE_VALUE[metro_id] * tilt * noise, 0),
            "zhvi_month": "2026-01",
            "zori_usd_month": round(SYNTHETIC_BASE_RENT[metro_id] * tilt * noise, 0),
            "zori_month": "2026-01",
            "synthetic": True,
        })
    return pd.DataFrame(rows)[schema.AREA_COLUMNS]


def make_sales(rng: np.random.Generator) -> pd.DataFrame:
    """Invented Seattle sales. The price depends on position, time and a 'brightness' factor
    that the synthetic photos also carry, so the photo model has something to find."""
    lat = rng.uniform(47.52, 47.72, N_SYNTHETIC_SALES)
    lon = rng.uniform(-122.40, -122.27, N_SYNTHETIC_SALES)
    day = rng.integers(0, 720, N_SYNTHETIC_SALES)
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(day, unit="D")
    brightness = rng.uniform(0.2, 0.9, N_SYNTHETIC_SALES)
    price = (500000 + 1500000 * (lat - 47.52) + 900000 * (-122.27 - lon) + 150 * day
             + 400000 * brightness) * rng.normal(1.0, 0.08, N_SYNTHETIC_SALES)
    ids = ["SYN%05d" % i for i in range(N_SYNTHETIC_SALES)]
    # A few repeat sales so the grouping logic is exercised.
    for i in range(0, 40, 2):
        ids[i + 1] = ids[i]
        lat[i + 1] = lat[i]
        lon[i + 1] = lon[i]
        brightness[i + 1] = brightness[i]
    df = pd.DataFrame({
        "property_id": ids,
        "address": ["%d Synthetic Example St, Seattle, WA 98103" % (100 + i) for i in range(N_SYNTHETIC_SALES)],
        "zip": "98103",
        "city_id": "seattle",
        "lat": np.round(lat, 6),
        "lon": np.round(lon, 6),
        "property_type": schema.SUPPORTED_PROPERTY_TYPE,
        "record_date": dates.strftime("%Y-%m-%d"),
        "target_type": "sale",
        "price_usd": np.round(price, 0),
        "currency": "USD",
        "price_unit": "total",
        "price_basis": "completed_transaction",
        "source": "SYNTHETIC fixture, generated by pipeline/make_fixtures.py",
        "synthetic": True,
        "furnished": "unknown",
        "brightness": np.round(brightness, 4),
    })
    return df.sort_values(["record_date", "property_id"]).reset_index(drop=True)


def make_images(sales: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Tiny flat-color JPEGs standing in for exterior photos. One per property."""
    image_dir = FIXTURES_DIR / "synthetic_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for property_id, group in sales.groupby("property_id"):
        level = int(255 * float(group["brightness"].iloc[0]))
        pixels = np.clip(rng.normal(level, 6, (32, 32, 3)), 0, 255).astype(np.uint8)
        name = "synthetic_%s.jpg" % property_id
        Image.fromarray(pixels).save(image_dir / name, quality=85)
        rows.append({
            "property_id": property_id,
            "image_ref": "synthetic-" + property_id,
            "image_provider": "synthetic",
            "image_license": "Synthetic fixture image, generated by this project, no restrictions",
            "image_captured_at": "2026-01-01",
            "image_distance_m": 12.0,
            "file": "synthetic_images/" + name,
        })
    return pd.DataFrame(rows)


def main() -> int:
    rng = np.random.default_rng(SEED)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    zips = make_zip_table(rng)
    sales = make_sales(rng)
    images = make_images(sales, rng)
    zips.to_csv(FIXTURES_DIR / "synthetic_zip_values.csv", index=False)
    sales.drop(columns=["brightness"]).to_csv(FIXTURES_DIR / "synthetic_seattle_sales.csv", index=False)
    images.to_csv(FIXTURES_DIR / "synthetic_image_matches.csv", index=False)
    # Embeddings of the synthetic images, so tests and demo mode do not need to run the encoder.
    from pipeline.embed_images import embed_files
    embedding_path = FIXTURES_DIR / "synthetic_image_embeddings.npz"
    if embedding_path.exists():
        embedding_path.unlink()
    embed_files({row.image_ref: FIXTURES_DIR / row.file for row in images.itertuples()}, embedding_path)
    ratio = {mid: {"metro": "SYNTHETIC", "month": "2026-01", "single_family_rent_index": 1.2,
                   "all_homes_rent_index": 1.0, "ratio": 1.2} for mid in SYNTHETIC_BASE_VALUE}
    with open(FIXTURES_DIR / "synthetic_metro_rent_ratio.json", "w", encoding="utf-8") as f:
        json.dump(ratio, f, indent=2)
    with open(FIXTURES_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write("# Synthetic fixtures\n\n"
                "Everything in this folder is SYNTHETIC. It was generated by "
                "`python -m pipeline.make_fixtures` from made-up formulas and random noise.\n\n"
                "None of these numbers is market data. ZIP codes and their rough center points are real "
                "places so that city checks work, but the home values, rents, sales, addresses and images "
                "attached to them are invented. They exist only for automated tests and for the clearly "
                "labeled demo mode.\n")
    print("wrote %d synthetic ZIP rows, %d synthetic sales, %d synthetic images to %s" % (
        len(zips), len(sales), len(images), FIXTURES_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
