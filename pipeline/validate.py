"""Check a housing records file against the import schema.

Usage:
    python -m pipeline.validate path/to/records.csv
    python -m pipeline.validate path/to/records.csv --clean-out cleaned.csv

The command prints one line per check and exits with code 1 when any
blocking problem is found. Rows that fail a blocking check are left out of
the cleaned output.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.geo import load_cities
from pipeline import schema


def _result(name: str, bad_mask: pd.Series, blocking: bool, note: str = "") -> Dict:
    return {"check": name, "rows_flagged": int(bad_mask.sum()), "blocking": blocking, "note": note}


def read_records(path: Path) -> pd.DataFrame:
    """Read a records CSV keeping IDs and ZIP codes as text."""
    return pd.read_csv(path, dtype={"property_id": str, "zip": str, "image_ref": str}, keep_default_na=True)


def validate_records(df: pd.DataFrame, city_ids: Optional[List[str]] = None) -> Tuple[List[Dict], pd.DataFrame]:
    """Run every check. Returns (list of check results, cleaned rows)."""
    if city_ids is None:
        city_ids = [c["id"] for c in load_cities()]
    results: List[Dict] = []

    missing_cols = [c for c in schema.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        results.append({"check": "required_columns", "rows_flagged": len(df), "blocking": True,
                        "note": "missing columns: %s" % ", ".join(missing_cols)})
        return results, df.iloc[0:0].copy()
    results.append({"check": "required_columns", "rows_flagged": 0, "blocking": True, "note": ""})

    df = df.copy()
    for col in schema.OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    bad = pd.Series(False, index=df.index)

    # 1. Missing values in required fields.
    text_cols = [c for c in schema.REQUIRED_COLUMNS if c not in ("lat", "lon", "price_usd", "synthetic")]
    missing = df[schema.REQUIRED_COLUMNS].isna().any(axis=1)
    for col in text_cols:
        missing |= df[col].astype(str).str.strip().eq("")
    results.append(_result("missing_values", missing, True))
    bad |= missing

    # 2. Currency and units.
    wrong_currency = df["currency"].astype(str).str.upper().ne("USD") & ~missing
    results.append(_result("currency_not_usd", wrong_currency, True))
    bad |= wrong_currency

    bad_target = ~df["target_type"].isin(schema.TARGET_TYPES) & ~missing
    results.append(_result("unknown_target_type", bad_target, True))
    bad |= bad_target

    expected_unit = df["target_type"].map(schema.UNIT_FOR_TARGET)
    wrong_unit = df["price_unit"].ne(expected_unit) & ~missing & ~bad_target
    results.append(_result("unit_mismatch", wrong_unit, True, "sales must be 'total', rents must be 'per_month'"))
    bad |= wrong_unit

    bad_basis = ~df["price_basis"].isin(schema.PRICE_BASES) & ~missing
    results.append(_result("unknown_price_basis", bad_basis, True))
    bad |= bad_basis

    # 3. Coordinates.
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    bad_coords = lat.isna() | lon.isna() | (lat.abs() > 90) | (lon.abs() > 180) | ((lat == 0) & (lon == 0))
    results.append(_result("invalid_coordinates", bad_coords, True))
    bad |= bad_coords

    # 4. Supported city and property type.
    bad_city = ~df["city_id"].isin(city_ids) & ~missing
    results.append(_result("unsupported_city", bad_city, True))
    bad |= bad_city

    bad_type = df["property_type"].ne(schema.SUPPORTED_PROPERTY_TYPE) & ~missing
    results.append(_result("unsupported_property_type", bad_type, True))
    bad |= bad_type

    # 5. Prices.
    price = pd.to_numeric(df["price_usd"], errors="coerce")
    low = df["target_type"].map(lambda t: schema.PLAUSIBLE_PRICE.get(t, (0, 0))[0])
    high = df["target_type"].map(lambda t: schema.PLAUSIBLE_PRICE.get(t, (0, 0))[1])
    bad_price = price.isna() | (price < low) | (price > high)
    results.append(_result("implausible_price", bad_price & ~missing, True))
    bad |= bad_price

    # 6. Dates.
    dates = pd.to_datetime(df["record_date"], errors="coerce")
    bad_date = dates.isna() | (dates > pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
    results.append(_result("invalid_or_future_date", bad_date & ~missing, True))
    bad |= bad_date

    # 7. Exact duplicate listings: same house, date, target and price.
    key = ["property_id", "record_date", "target_type", "price_usd"]
    duplicate = df.duplicated(subset=key, keep="first")
    results.append(_result("duplicate_rows", duplicate, True, "only the first copy is kept"))
    bad |= duplicate

    # 8. Repeated properties. Allowed, but the split must keep them together.
    repeated = df.duplicated(subset=["property_id", "target_type"], keep=False) & ~duplicate
    results.append(_result("repeated_properties", repeated, False,
                           "kept; rows for one property are never split across training and testing"))

    # 9. Furnishing status values.
    furnished = df["furnished"].fillna("").astype(str).str.lower()
    bad_furnished = ~furnished.isin(schema.FURNISHED_VALUES)
    results.append(_result("unknown_furnished_value", bad_furnished, True))
    bad |= bad_furnished

    # 10. Photo and property mismatch.
    has_image = df["image_ref"].notna() & df["image_ref"].astype(str).str.strip().ne("")
    distance = pd.to_numeric(df["image_distance_m"], errors="coerce")
    mismatch = has_image & (distance.isna() | (distance > schema.MAX_IMAGE_DISTANCE_M))
    results.append(_result("image_property_mismatch", mismatch, False,
                           "photo is dropped for these rows, the price row is kept"))
    no_license = has_image & (df["image_license"].isna() | df["image_license"].astype(str).str.strip().eq(""))
    results.append(_result("image_without_license", no_license, False, "photo is dropped for these rows"))

    cleaned = df.loc[~bad].copy()
    drop_image = (mismatch | no_license).loc[cleaned.index]
    for col in ("image_ref", "image_provider", "image_license", "image_captured_at", "image_distance_m"):
        cleaned[col] = cleaned[col].astype(object)
        cleaned.loc[drop_image, col] = np.nan
    return results, cleaned


def validate_area_table(df: pd.DataFrame) -> List[Dict]:
    """Checks for the ZIP level table of typical values."""
    results: List[Dict] = []
    missing_cols = [c for c in schema.AREA_COLUMNS if c not in df.columns]
    if missing_cols:
        return [{"check": "required_columns", "rows_flagged": len(df), "blocking": True,
                 "note": "missing columns: %s" % ", ".join(missing_cols)}]
    results.append(_result("duplicate_zip", df.duplicated(subset=["zip"]), True))
    results.append(_result("missing_coordinates", df["lat"].isna() | df["lon"].isna(), True))
    zhvi = df["zhvi_usd"]
    low, high = schema.PLAUSIBLE_PRICE["sale"]
    results.append(_result("implausible_home_value", zhvi.notna() & ((zhvi < low) | (zhvi > high)), True))
    zori = df["zori_usd_month"]
    low, high = schema.PLAUSIBLE_PRICE["rent"]
    results.append(_result("implausible_rent", zori.notna() & ((zori < low) | (zori > high)), True))
    results.append(_result("no_home_value", zhvi.isna(), False, "ZIP is not used for the sale model"))
    results.append(_result("no_rent_value", zori.isna(), False, "ZIP is not used for the rent model"))
    return results


def has_blocking_problem(results: List[Dict]) -> bool:
    return any(r["blocking"] and r["rows_flagged"] > 0 for r in results)


def print_results(results: List[Dict]) -> None:
    for r in results:
        if r["rows_flagged"] == 0:
            status = "ok"
        elif r["blocking"]:
            status = "FAIL"
        else:
            status = "note"
        line = "%-5s %-28s %7d rows" % (status, r["check"], r["rows_flagged"])
        if r["note"] and r["rows_flagged"] > 0:
            line += "  (%s)" % r["note"]
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a HomeLens housing records CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--clean-out", help="write rows that pass every blocking check to this file")
    parser.add_argument("--report-out", help="write the check results as JSON")
    args = parser.parse_args()

    df = read_records(Path(args.csv_path))
    results, cleaned = validate_records(df)
    print("%d rows read from %s" % (len(df), args.csv_path))
    print_results(results)
    print("%d rows pass every blocking check" % len(cleaned))
    if args.clean_out:
        cleaned.to_csv(args.clean_out, index=False)
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump({"rows_read": len(df), "rows_clean": len(cleaned), "checks": results}, f, indent=2)
    return 1 if has_blocking_problem(results) else 0


if __name__ == "__main__":
    sys.exit(main())
