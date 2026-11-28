"""Build sale and house records for King County cities outside Seattle.

Usage:
    python -m pipeline.king_county_cities --download   # fetch the parcel list first
    python -m pipeline.king_county_cities

These cities are never used for training. They are the test set for the
ZIP-relative model (pipeline/train_zip_relative.py): a model trained on
Seattle only is scored on houses in Bellevue, Kent and the other cities in
config/data_sources.json ("king_county_test_city_parcels").

The sales, building and parcel files are the same county-wide King County
files the Seattle pipeline downloads, so the same sale filter and the same
house record (core/house_features.py) are used. Only the parcel list,
which gives each parcel's city, address and position, is fetched separately.

Outputs in data/processed:
    test_city_sales.csv           one row per open-market single-family sale, import schema
    test_city_parcels.csv         address, city and coordinates per parcel
    test_city_house_features.csv  King County building and parcel facts per parcel
Report: reports/test_city_data.json
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from core import house_features
from core.geo import tidy_street
from core.paths import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from pipeline import schema
from pipeline.build_dataset import filter_market_sales, read_sales
from pipeline.download import download_parcels, load_sources
from pipeline.validate import print_results, validate_records

SPEC_KEY = "king_county_test_city_parcels"


def city_id(name: str) -> str:
    """'Federal Way' -> 'federal_way'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def test_cities() -> List[str]:
    return list(load_sources()[SPEC_KEY]["cities"])


def read_test_city_parcels(path: Path) -> pd.DataFrame:
    """One row per parcel: property_id, address, zip, city, city_id, lat, lon."""
    p = pd.read_csv(path, dtype={"PIN": str, "ZIP5": str})
    p["PIN"] = p["PIN"].str.zfill(10)
    p = p.dropna(subset=["LAT", "LON", "ADDR_FULL", "CTYNAME"])
    p = p.sort_values("PRIMARY_ADDR", ascending=False).drop_duplicates("PIN")
    zip5 = p["ZIP5"].fillna("").str[:5]
    out = pd.DataFrame({
        "property_id": p["PIN"],
        "address": p["ADDR_FULL"].map(tidy_street) + ", " + p["CTYNAME"] + ", WA " + zip5,
        "zip": zip5,
        "city": p["CTYNAME"],
        "city_id": p["CTYNAME"].map(city_id),
        "lat": p["LAT"].round(6),
        "lon": p["LON"].round(6),
    })
    return out[out["zip"].str.len() == 5].reset_index(drop=True)


def join_sales(sales: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """Sales (already filtered) on the test-city parcels, in the import schema plus a city column."""
    joined = sales.merge(parcels, on="property_id", how="inner")
    records = pd.DataFrame({
        "property_id": joined["property_id"],
        "address": joined["address"],
        "zip": joined["zip"],
        "city_id": joined["city_id"],
        "city": joined["city"],
        "lat": joined["lat"],
        "lon": joined["lon"],
        "property_type": schema.SUPPORTED_PROPERTY_TYPE,
        "record_date": joined["sale_date"].dt.strftime("%Y-%m-%d"),
        "target_type": "sale",
        "price_usd": joined["SalePrice"].astype(float),
        "currency": "USD",
        "price_unit": "total",
        "price_basis": "completed_transaction",
        "source": "King County Department of Assessments, Real Property Sales",
        "synthetic": False,
        "furnished": "unknown",
    })
    return records.sort_values(["record_date", "property_id"]).reset_index(drop=True)


def build(raw_dir: Path, processed_dir: Path) -> Dict:
    spec = load_sources()[SPEC_KEY]
    parcels = read_test_city_parcels(raw_dir / spec["dest"])
    sales, counts = filter_market_sales(read_sales(raw_dir))
    records = join_sales(sales, parcels)
    counts["on_a_test_city_single_family_parcel"] = int(len(records))

    checks, cleaned = validate_records(records, city_ids=sorted(parcels["city_id"].unique()))
    print_results(checks)
    # Like the Seattle build: flagged rows (for example exact duplicates) are already removed from `cleaned`.
    if cleaned.empty:
        raise ValueError("No test-city sales passed record validation")
    cleaned = cleaned[schema.REQUIRED_COLUMNS + ["furnished", "city"]]

    features = house_features.read_king_county(raw_dir)
    features = features[features.index.isin(parcels["property_id"])]

    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_dir / "test_city_sales.csv", index=False)
    parcels.to_csv(processed_dir / "test_city_parcels.csv", index=False)
    features.to_csv(processed_dir / "test_city_house_features.csv")
    return {
        "cities": spec["cities"],
        "filter_counts": counts,
        "rows_after_checks": int(len(cleaned)),
        "sales_per_city": {k: int(v) for k, v in cleaned["city"].value_counts().sort_index().items()},
        "parcels": int(len(parcels)),
        "parcels_with_a_building_record": int(len(features)),
        "sales_from": str(cleaned["record_date"].min()),
        "sales_to": str(cleaned["record_date"].max()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="fetch the test-city parcel list first")
    args = parser.parse_args()
    if args.download and not download_parcels(RAW_DIR, spec_key=SPEC_KEY):
        return 1
    report = build(RAW_DIR, PROCESSED_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "test_city_data.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Test-city sales kept: %d (%s to %s)" % (report["rows_after_checks"], report["sales_from"], report["sales_to"]))
    print("Per city: %s" % json.dumps(report["sales_per_city"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
