"""Build San Francisco's detached house records from public parcel data."""
import argparse
import json
import re
import ssl
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from core.features import SALE_PRICE_LIMITS
from core.geo import san_francisco_street_key, tidy_street
from core.paths import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from pipeline.download import download_file, provenance_entry, record_provenance, sha256_of
from pipeline.validate import has_blocking_problem, validate_records

CITY = "san_francisco"
FIRST_ROLL, LAST_ROLL = 2017, 2025
SALES_FILE = "realestatedataset_sf_2026_10.csv"
SALES_SOURCE = "RealEstateDataset San Francisco County (October 2026)"
SALES_URL = ("https://realestatedataset.com/data/latest/county=06075.csv"
             "?dl=realestatedataset-san-francisco-county-ca-2026-10.csv")
SALES_SHA256 = "e24e195a7712fcf6483cfd4bbd39415fb8a4e60e4b08910ba7efbca8fc6582b1"
ROLL_FIELDS = {
    "property_area": "living_sqft",
    "year_property_built": "year_built",
    "number_of_bedrooms": "bedrooms",
    "number_of_bathrooms": "bathrooms",
    "number_of_rooms": "rooms",
    "number_of_stories": "stories",
    "lot_area": "lot_sqft",
}
FEATURE_COLUMNS = tuple(ROLL_FIELDS.values())
SALE_COLUMNS = ("city", "county_fips", "parcel_apn", "street", "unit", "zip", "property_type",
                "last_sold_date", "last_sold_price", "listing_type")
NONSTANDARD_LISTINGS = {"auction", "bank_owned", "foreclosure", "pre_foreclosure"}
ADDRESS_COLUMNS = ("property_id", "address", "zip", "lat", "lon")


def _datasf(asset: str, ext: str, fields: str, order: str, limit: int, where: str = "") -> str:
    params = {"$select": fields, "$order": order, "$limit": limit}
    if where:
        params["$where"] = where
    return "https://data.sf.gov/resource/%s.%s?%s" % (asset, ext, urlencode(params))


def download_inputs(raw: Path) -> None:
    roll_fields = ",".join(("closed_roll_year", "parcel_number", "current_sales_date", *ROLL_FIELDS,
                           "number_of_units", "construction_type", "property_class_code", "use_code"))
    inputs = [
        ("addresses.csv", _datasf("ramy-di5m", "csv", "eas_fullid,parcel_number,zip_code,address,longitude,latitude",
                                  "eas_fullid", 400000, "unit_number is null and parcel_number is not null "
                                  "and latitude is not null and longitude is not null"), "DataSF PDDL"),
        ("footprints.geojson", _datasf("ynuv-fyni", "geojson", "mblr,shape", "sf16_bldgid", 350000), "DataSF PDDL"),
        (SALES_FILE, SALES_URL, "Publisher's CC0 1.0 dedication"),
    ]
    for year in range(FIRST_ROLL, LAST_ROLL + 1):
        where = "closed_roll_year=%d and property_class_code='D' and use_code='SRES'" % year
        inputs.append(("house_roll_%d.csv" % year, _datasf("wv5m-vpq2", "csv", roll_fields, "row_id",
                                                            150000, where), "DataSF PDDL"))
    for name, url, license_note in inputs:
        path = raw / name
        downloaded = not path.exists()
        if downloaded:
            print("Downloading " + name, flush=True)
            try:
                download_file(url, path)
            except URLError as exc:
                if sys.platform != "win32" or not isinstance(exc.reason, ssl.SSLCertVerificationError):
                    raise
                # Windows curl uses the Windows certificate store for intercepted TLS.
                part = path.with_suffix(path.suffix + ".part")
                subprocess.run(["curl.exe", "--fail", "--silent", "--show-error", "--location",
                                "--compressed", "--max-time", "900", "--output", str(part), url], check=True)
                part.replace(path)
        entry = provenance_entry(path, url, "San Francisco house and sale source", license_note, downloaded)
        record_provenance(raw, Path(name).stem, entry)


def parcel_key(value) -> str:
    """The city uses a four-digit block and three-digit lot, sometimes with a suffix."""
    raw = re.sub(r"[\s-]", "", str(value).upper())
    if not re.fullmatch(r"[0-9]{6,7}[A-Z]?", raw):
        return ""
    digits = 7 + int(raw[-1].isalpha())
    return "sf-" + raw.zfill(digits)


def read_roll(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, dtype={"parcel_number": str})
    rows["property_id"] = rows["parcel_number"].map(parcel_key)
    rows = rows[(rows["property_id"] != "") & rows["property_class_code"].eq("D") & rows["use_code"].eq("SRES")]
    rows = rows.drop_duplicates().drop_duplicates("property_id", keep=False)
    rows = rows.rename(columns=ROLL_FIELDS)
    rows["official_sale_date"] = pd.to_datetime(rows["current_sales_date"], errors="coerce").dt.normalize()
    return rows.set_index("property_id")


def complete_records(roll: pd.DataFrame) -> pd.DataFrame:
    features = roll[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    valid = features.gt(0).all(axis=1) & np.isfinite(features).all(axis=1)
    valid &= features["year_built"].between(1800, roll["closed_roll_year"] + 1)
    valid &= roll["number_of_units"].eq(1)
    return features.loc[valid]


def detached_parcels(path: Path, candidates: pd.Index) -> set:
    """Require a mapped main footprint separated from buildings on other parcels."""
    from shapely import STRtree
    from shapely.affinity import scale
    from shapely.geometry import shape

    # At San Francisco's latitude, these degree-to-metre scales are adequate for a 30 cm guard.
    geometries, parcels = [], []
    with path.open(encoding="utf-8") as f:
        features = json.load(f)["features"]
    for feature in features:
        if not feature["geometry"]:
            continue
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            continue
        geometries.append(scale(geometry, xfact=87900, yfact=111000, origin=(0, 0)))
        parcel = feature["properties"].get("mblr") or ""
        parcels.append(parcel_key(parcel[2:]) if parcel.startswith("SF") else "")
    parcel_ids = np.asarray(parcels)
    main = (pd.DataFrame({"parcel": parcel_ids, "area": [g.area for g in geometries]})
            .loc[lambda table: table["parcel"].isin(candidates)]
            .sort_values("area", ascending=False).drop_duplicates("parcel"))
    indices = main.index.to_numpy()
    selected = np.asarray(geometries, dtype=object)[indices]
    # The source splits footprints at common walls. Reject touching or nearly touching structures.
    left, right = STRtree(geometries).query(selected, predicate="dwithin", distance=0.30)
    attached = left[(indices[left] != right) & (parcel_ids[indices[left]] != parcel_ids[right])]
    detached = np.ones(len(main), dtype=bool)
    detached[attached] = False
    return set(main.loc[detached, "parcel"])


def read_addresses(path: Path, detached: set) -> pd.DataFrame:
    addresses = pd.read_csv(path, dtype={"parcel_number": str, "zip_code": str})
    addresses["property_id"] = addresses["parcel_number"].map(parcel_key)
    addresses = addresses.rename(columns={"latitude": "lat", "longitude": "lon", "zip_code": "zip"})
    addresses = addresses[addresses["property_id"].isin(detached)].dropna(subset=list(ADDRESS_COLUMNS))
    addresses = addresses[addresses["zip"].str.fullmatch(r"941\d{2}") &
                          addresses["lat"].between(37.69, 37.84) & addresses["lon"].between(-122.53, -122.34)]
    addresses["street_key"] = addresses["address"].map(san_francisco_street_key)
    addresses = addresses[addresses["street_key"].ne("")]
    addresses = addresses.drop_duplicates(["property_id", "street_key"])
    shared_addresses = addresses.groupby("street_key")["property_id"].transform("nunique")
    addresses = addresses[shared_addresses.eq(1)].copy()
    addresses["address"] = addresses["address"].map(tidy_street) + ", San Francisco, CA " + addresses["zip"]
    return addresses


def read_sales(path: Path, addresses: pd.DataFrame, rolls: dict, required: pd.DataFrame) -> tuple:
    sales = pd.read_csv(path, usecols=SALE_COLUMNS, dtype={column: str for column in SALE_COLUMNS})
    sales = sales[sales["city"].eq("San Francisco") & sales["county_fips"].eq("06075")
                  & sales["property_type"].eq("single_family") & sales["unit"].isna()
                  & ~sales["listing_type"].isin(NONSTANDARD_LISTINGS)].copy()
    sales["record_date"] = pd.to_datetime(sales["last_sold_date"], errors="coerce", format="%Y-%m-%d")
    sales["price_usd"] = pd.to_numeric(sales["last_sold_price"], errors="coerce")
    sales["property_id"] = sales["parcel_apn"].map(parcel_key)
    sales["street_key"] = sales["street"].fillna("").map(san_francisco_street_key)
    sales = sales[sales["record_date"].between("%d-01-01" % (FIRST_ROLL + 1), "2026-09-30")
                  & sales["price_usd"].between(*SALE_PRICE_LIMITS) & sales["property_id"].ne("")]
    counts = {"candidate_price_rows": len(sales)}
    matches = addresses[[*ADDRESS_COLUMNS, "street_key"]]
    sales = sales.merge(matches, on=["property_id", "street_key", "zip"], how="inner", validate="many_to_one")
    counts["detached_exact_address_and_parcel"] = len(sales)

    # Where more than one home record supplies the same parcel/date, conflicting prices are unusable.
    sales = sales.drop_duplicates(["property_id", "record_date", "price_usd"])
    sales = sales.drop_duplicates(["property_id", "record_date"], keep=False)
    official = (pd.concat([r["official_sale_date"] for r in rolls.values()]).dropna()
                .reset_index().drop_duplicates())
    observed = sales[["property_id", "record_date"]].merge(official, on="property_id", how="left")
    agree = (observed["record_date"] - observed["official_sale_date"]).abs().dt.days.le(14)
    confirmed = observed.loc[agree, ["property_id", "record_date"]].drop_duplicates()
    sales = sales.merge(confirmed.assign(assessor_date_confirmed=True), on=["property_id", "record_date"], how="left")
    sales["assessor_date_confirmed"] = sales["assessor_date_confirmed"].eq(True)

    # Use the prior roll's features, and require a usable record for the same parcel today.
    sales = sales[sales["property_id"].isin(required.index)].copy()
    sales["feature_roll"] = sales["record_date"].dt.year - 1
    prior = pd.concat([complete_records(roll).assign(feature_roll=year)
                       for year, roll in rolls.items()]).reset_index()
    sales = sales.merge(prior, on=["property_id", "feature_roll"], how="inner", validate="many_to_one")
    sales = sales[sales["year_built"].lt(sales["record_date"].dt.year)].copy()
    # An identically priced/date-stamped transfer on several eligible parcels may cover more than one home.
    shared = sales.groupby(["record_date", "price_usd"])["property_id"].transform("nunique")
    sales = sales[shared.eq(1)].copy()
    counts["usable_candidate_sales"] = len(sales)
    counts["assessor_sale_date_agreement_within_14_days"] = int(sales["assessor_date_confirmed"].sum())
    counts["candidate_sales_by_year"] = {str(year): int(n) for year, n in
                                         sales["record_date"].dt.year.value_counts().sort_index().items()}
    sales = sales.assign(city_id=CITY, property_type="detached_single_family", target_type="sale",
                         currency="USD", price_unit="total", price_basis="completed_transaction",
                         source=SALES_SOURCE, synthetic=False)
    columns = [*ADDRESS_COLUMNS, "city_id", "property_type", "record_date", "target_type", "price_usd",
               "currency", "price_unit", "price_basis", "source", "synthetic", "assessor_date_confirmed",
               "feature_roll", *FEATURE_COLUMNS]
    return sales[columns].sort_values(["record_date", "property_id"]), counts


def build_records(raw: Path, processed: Path) -> dict:
    if sha256_of(raw / SALES_FILE) != SALES_SHA256:
        raise ValueError("The county sales file does not match the audited October 2026 snapshot")
    rolls = {year: read_roll(raw / ("house_roll_%d.csv" % year)) for year in range(FIRST_ROLL, LAST_ROLL + 1)}
    latest = rolls[LAST_ROLL]
    detached = detached_parcels(raw / "footprints.geojson", latest.index)
    addresses = read_addresses(raw / "addresses.csv", detached)
    houses = complete_records(latest)
    houses = houses[houses.index.isin(addresses["property_id"])]
    sales, counts = read_sales(raw / SALES_FILE, addresses, rolls, houses)
    checks, _ = validate_records(sales, city_ids=[CITY])
    if has_blocking_problem(checks):
        raise ValueError("San Francisco sales failed record validation: %s" % checks)
    counts.update({"detached_footprint_parcels": len(detached), "detached_parcels_with_address":
                   int(addresses["property_id"].nunique()), "complete_current_house_records": len(houses)})
    processed.mkdir(parents=True, exist_ok=True)
    addresses[list(ADDRESS_COLUMNS)].to_csv(processed / "san_francisco_parcels.csv", index=False)
    houses.to_csv(processed / "san_francisco_house_features.csv")
    sales.to_csv(processed / "san_francisco_sales.csv", index=False)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing public source files first.")
    args = parser.parse_args()
    raw = RAW_DIR / CITY
    raw.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_inputs(raw)
    counts = build_records(raw, PROCESSED_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "san_francisco_data.json").open("w", encoding="utf-8") as output:
        json.dump(counts, output, indent=2)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
