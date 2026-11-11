"""Turn the raw downloads into the tables the models train on.

Usage:
    python -m pipeline.build_dataset

Outputs in data/processed:
    zip_values.csv        one row per ZIP: typical home value and rent (Zillow), center point (Census)
    zhvi_history.csv      Zillow home value per Seattle-area ZIP per month (for a baseline)
    metro_rent_ratio.json single-family rent divided by all-homes rent, per metro (Zillow)
    seattle_sales.csv     one row per real single-family sale in Seattle (King County), import schema
    seattle_parcels.csv   address and coordinates of Seattle single-family parcels
    validation_report.json results of every data check
"""
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.geo import load_cities, tidy_street
from core.paths import PROCESSED_DIR, RAW_DIR
from pipeline import schema
from pipeline.validate import has_blocking_problem, print_results, validate_area_table, validate_records

ZILLOW_META = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName", "State", "City", "Metro", "CountyName"]
HISTORY_MONTHS = 40
SALES_WINDOW_YEARS = 3

# King County code values, from the county's own lookup file (Lookup.zip).
KC_PRINCIPAL_USE_RESIDENTIAL = 6
KC_PROPERTY_CLASS_RES_IMPROVED = 8
KC_PROPERTY_TYPE_SINGLE_FAMILY = 11
KC_DEED_TYPES = (2, 3)            # Warranty Deed, Statutory Warranty Deed
KC_SALE_REASON_NONE = 1


def read_zillow_zip_file(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Read a Zillow ZIP-level CSV. Returns (table, list of month columns)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    months = [c for c in header if c not in ZILLOW_META]
    keep_months = months[-HISTORY_MONTHS:]
    df = pd.read_csv(path, usecols=ZILLOW_META + keep_months, dtype={"RegionName": str})
    df["zip"] = df["RegionName"].str.zfill(5)
    return df, keep_months


def latest_value(df: pd.DataFrame, months: List[str]) -> Tuple[pd.Series, str]:
    """Value in the newest month of the file. ZIPs with no value that month stay empty."""
    last = months[-1]
    return df[last], last[:7]


def read_gazetteer(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as f:
            g = pd.read_csv(f, sep="\t", dtype={"GEOID": str})
    g.columns = [c.strip() for c in g.columns]
    g["zip"] = g["GEOID"].str.zfill(5)
    return g[["zip", "INTPTLAT", "INTPTLONG"]].rename(columns={"INTPTLAT": "lat", "INTPTLONG": "lon"})


def build_zip_values(raw_dir: Path, cities: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the ZIP table and the Seattle-area value history."""
    zhvi, zhvi_months = read_zillow_zip_file(raw_dir / "zillow" / "Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv")
    zori, zori_months = read_zillow_zip_file(raw_dir / "zillow" / "Zip_zori_uc_sfrcondomfr_sm_month.csv")
    centers = read_gazetteer(raw_dir / "census" / "2023_Gaz_zcta_national.zip")

    metro_to_id = {c["zillow_metro"]: c["id"] for c in cities}
    zhvi = zhvi[zhvi["Metro"].isin(metro_to_id)].copy()
    zhvi["zhvi_usd"], zhvi_month = latest_value(zhvi, zhvi_months)
    zori = zori[zori["Metro"].isin(metro_to_id)].copy()
    zori["zori_usd_month"], zori_month = latest_value(zori, zori_months)

    table = zhvi[["zip", "City", "State", "Metro", "zhvi_usd"]].merge(
        zori[["zip", "zori_usd_month"]], on="zip", how="left")
    table = table.merge(centers, on="zip", how="left")
    table = table.rename(columns={"City": "city", "State": "state", "Metro": "metro"})
    table["metro_id"] = table["metro"].map(metro_to_id)
    table["supported_city_id"] = ""
    for c in cities:
        in_city = (table["city"] == c["zillow_city"]) & (table["state"] == c["state"])
        in_city |= table["zip"].isin(c.get("extra_zips", []))
        table.loc[in_city, "supported_city_id"] = c["id"]
    table["zhvi_month"] = zhvi_month
    table["zori_month"] = zori_month
    table["synthetic"] = False
    table = table[schema.AREA_COLUMNS].sort_values("zip").reset_index(drop=True)

    seattle = zhvi[zhvi["Metro"] == [c for c in cities if c["id"] == "seattle"][0]["zillow_metro"]]
    history = seattle[["zip"] + zhvi_months].melt(id_vars=["zip"], var_name="month", value_name="zhvi_usd")
    history["month"] = history["month"].str[:7]
    history = history.dropna(subset=["zhvi_usd"]).sort_values(["zip", "month"]).reset_index(drop=True)
    return table, history


def build_metro_rent_ratio(raw_dir: Path, cities: List[Dict]) -> Dict:
    """Single-family rent index divided by all-homes rent index, newest shared month, per metro."""
    sfr = pd.read_csv(raw_dir / "zillow" / "Metro_zori_uc_sfr_sm_month.csv")
    allh = pd.read_csv(raw_dir / "zillow" / "Metro_zori_uc_sfrcondomfr_sm_month.csv")
    meta = [c for c in sfr.columns if not c[:4].isdigit()]
    months = [c for c in sfr.columns if c not in meta and c in allh.columns]
    out = {}
    for c in cities:
        name = c["zillow_metro_short"]
        row_sfr = sfr[sfr["RegionName"] == name]
        row_all = allh[allh["RegionName"] == name]
        if row_sfr.empty or row_all.empty:
            continue
        for month in reversed(months):
            a = row_sfr[month].iloc[0]
            b = row_all[month].iloc[0]
            if pd.notna(a) and pd.notna(b) and b > 0:
                out[c["id"]] = {
                    "metro": name,
                    "month": month[:7],
                    "single_family_rent_index": round(float(a), 2),
                    "all_homes_rent_index": round(float(b), 2),
                    "ratio": round(float(a) / float(b), 4),
                }
                break
    return out


def read_parcels(raw_dir: Path) -> pd.DataFrame:
    p = pd.read_csv(raw_dir / "king_county" / "seattle_single_family_parcels.csv",
                    dtype={"PIN": str, "ZIP5": str})
    p["PIN"] = p["PIN"].str.zfill(10)
    p = p.dropna(subset=["LAT", "LON", "ADDR_FULL"])
    p = p.sort_values("PRIMARY_ADDR", ascending=False).drop_duplicates("PIN")
    out = pd.DataFrame({
        "property_id": p["PIN"],
        "address": p["ADDR_FULL"].map(tidy_street) + ", Seattle, WA " + p["ZIP5"].fillna("").str[:5],
        "zip": p["ZIP5"].fillna("").str[:5],
        "lat": p["LAT"].round(6),
        "lon": p["LON"].round(6),
    })
    return out[out["zip"].str.len() == 5].reset_index(drop=True)


def read_sales(raw_dir: Path) -> pd.DataFrame:
    """Read King County sales. Buyer and seller names are never loaded."""
    wanted = ["ExciseTaxNbr", "Major", "Minor", "DocumentDate", "SalePrice", "PropertyType",
              "PrincipalUse", "SaleInstrument", "SaleReason", "PropertyClass", "SaleWarning"]
    with zipfile.ZipFile(raw_dir / "king_county" / "Real Property Sales.zip") as zf:
        with zf.open(zf.namelist()[0]) as f:
            s = pd.read_csv(f, encoding="latin-1", usecols=wanted, dtype={"Major": str, "Minor": str},
                            low_memory=False)
    s["sale_date"] = pd.to_datetime(s["DocumentDate"], errors="coerce")
    s["property_id"] = s["Major"].str.strip().str.zfill(6) + s["Minor"].str.strip().str.zfill(4)
    return s


def filter_market_sales(s: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Keep ordinary open-market sales of one improved single-family parcel."""
    counts = {"all_rows": int(len(s))}
    s = s.dropna(subset=["sale_date"])
    newest = s["sale_date"].max()
    start = newest - pd.DateOffset(years=SALES_WINDOW_YEARS)
    s = s[s["sale_date"] > start]
    counts["in_last_%d_years" % SALES_WINDOW_YEARS] = int(len(s))

    # A sale that covers several parcels has one price for all of them: drop it.
    parcels_per_sale = s.groupby("ExciseTaxNbr")["property_id"].transform("nunique")
    s = s[parcels_per_sale == 1]
    counts["single_parcel_sales"] = int(len(s))

    s = s[s["SalePrice"] > 0]
    counts["price_above_zero"] = int(len(s))

    s = s[(s["PrincipalUse"] == KC_PRINCIPAL_USE_RESIDENTIAL)
          & (s["PropertyClass"] == KC_PROPERTY_CLASS_RES_IMPROVED)
          & (s["PropertyType"] == KC_PROPERTY_TYPE_SINGLE_FAMILY)]
    counts["residential_improved_single_family"] = int(len(s))

    warning = s["SaleWarning"].fillna("").astype(str).str.strip()
    s = s[s["SaleInstrument"].isin(KC_DEED_TYPES) & (s["SaleReason"] == KC_SALE_REASON_NONE) & warning.eq("")]
    counts["warranty_deed_no_warnings"] = int(len(s))
    return s, counts


def build_seattle_sales(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    parcels = read_parcels(raw_dir)
    sales, counts = filter_market_sales(read_sales(raw_dir))
    joined = sales.merge(parcels, on="property_id", how="inner")
    counts["on_seattle_single_family_parcel"] = int(len(joined))
    records = pd.DataFrame({
        "property_id": joined["property_id"],
        "address": joined["address"],
        "zip": joined["zip"],
        "city_id": "seattle",
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
    records = records.sort_values(["record_date", "property_id"]).reset_index(drop=True)
    return records, parcels, counts


def main() -> int:
    cities = load_cities()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict = {}

    print("Building ZIP table from Zillow and Census files")
    zip_table, history = build_zip_values(RAW_DIR, cities)
    # A few ZIPs (for example post office box ZIPs) have no Census center point. They are left out.
    no_center = zip_table["lat"].isna() | zip_table["lon"].isna()
    dropped_zips = zip_table.loc[no_center, "zip"].tolist()
    if dropped_zips:
        print("Leaving out %d ZIP(s) with no Census center point: %s" % (len(dropped_zips), ", ".join(dropped_zips)))
    zip_table = zip_table[~no_center].reset_index(drop=True)
    area_checks = validate_area_table(zip_table)
    print_results(area_checks)
    report["zip_values"] = {"rows": int(len(zip_table)), "zips_without_center_point": dropped_zips,
                            "checks": area_checks}
    if has_blocking_problem(area_checks):
        print("ZIP table failed validation")
        return 1
    zip_table.to_csv(PROCESSED_DIR / "zip_values.csv", index=False)
    history.to_csv(PROCESSED_DIR / "zhvi_history.csv", index=False)

    ratios = build_metro_rent_ratio(RAW_DIR, cities)
    with open(PROCESSED_DIR / "metro_rent_ratio.json", "w", encoding="utf-8") as f:
        json.dump(ratios, f, indent=2)
    report["metro_rent_ratio"] = ratios

    print("Building Seattle sales from King County files")
    records, parcels, counts = build_seattle_sales(RAW_DIR)
    checks, cleaned = validate_records(records)
    print_results(checks)
    report["seattle_sales"] = {
        "filter_counts": counts,
        "rows_before_checks": int(len(records)),
        "rows_after_checks": int(len(cleaned)),
        "checks": checks,
    }
    cleaned = cleaned[schema.REQUIRED_COLUMNS + ["furnished"]]
    cleaned.to_csv(PROCESSED_DIR / "seattle_sales.csv", index=False)
    parcels.to_csv(PROCESSED_DIR / "seattle_parcels.csv", index=False)

    with open(PROCESSED_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for c in cities:
        in_city = zip_table[zip_table["supported_city_id"] == c["id"]]
        print("%-14s %3d ZIPs in city, %3d with home value, %3d with rent" % (
            c["name"], len(in_city), in_city["zhvi_usd"].notna().sum(), in_city["zori_usd_month"].notna().sum()))
    print("Metro area ZIPs with a home value: %d" % zip_table["zhvi_usd"].notna().sum())
    print("Seattle sales kept: %d (from %s to %s)" % (
        len(cleaned), cleaned["record_date"].min(), cleaned["record_date"].max()))
    print("Sale filter steps: %s" % json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
