"""Facts about one house from King County's public building and parcel records.

These are what make two neighbouring houses get different prices: living
area, building grade, age, condition, rooms, lot size, view and waterfront.
They come from the county record for the confirmed parcel, so the user does
not type them, and the app knows them for any Seattle single-family parcel.

The records describe each building as it is recorded today, not as it was
on the day of each past sale. Training therefore drops sales of houses that
were built or renovated in or after the year of the sale (see
pipeline/train.py), because for those the record describes a different house.
"""
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "living_sqft",          # finished living area, square feet
    "grade",                # King County building grade, 1 (poor) to 13 (mansion), 7 is average
    "year_built",
    "year_renovated",       # 0 when never renovated
    "condition",            # 1 (poor) to 5 (very good)
    "bedrooms",
    "bathrooms",            # full = 1, three-quarter = 0.75, half = 0.5
    "stories",
    "basement_sqft",        # finished basement, square feet
    "buildings_on_parcel",
    "lot_sqft",
    "view_score",           # sum of the county's view ratings (mountains, water, skyline), 0 = no view
    "waterfront",           # 1 when the parcel is on the water
]

VIEW_COLUMNS = ["MtRainier", "Olympics", "Cascades", "Territorial", "SeattleSkyline", "PugetSound",
                "LakeWashington", "LakeSammamish", "SmallLakeRiverCreek", "OtherView"]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pin(df: pd.DataFrame) -> pd.Series:
    return df["Major"].astype(str).str.strip().str.zfill(6) + df["Minor"].astype(str).str.strip().str.zfill(4)


def read_king_county(raw_dir: Path) -> pd.DataFrame:
    """One row per parcel with FEATURE_COLUMNS, from the county's building and parcel files."""
    kc = Path(raw_dir) / "king_county"
    with zipfile.ZipFile(kc / "Residential Building.zip") as zf, zf.open("EXTR_ResBldg.csv") as f:
        b = pd.read_csv(f, encoding="latin-1", dtype=str,
                        usecols=["Major", "Minor", "BldgNbr", "SqFtTotLiving", "BldgGrade", "YrBuilt", "YrRenovated",
                                 "Condition", "Bedrooms", "BathFullCount", "Bath3qtrCount", "BathHalfCount",
                                 "Stories", "SqFtFinBasement"])
    b["property_id"] = _pin(b)
    buildings = b.groupby("property_id").size()
    b = b[_num(b["BldgNbr"]) == 1].drop_duplicates("property_id").set_index("property_id")
    out = pd.DataFrame(index=b.index)
    out["living_sqft"] = _num(b["SqFtTotLiving"])
    out["grade"] = _num(b["BldgGrade"])
    out["year_built"] = _num(b["YrBuilt"])
    out["year_renovated"] = _num(b["YrRenovated"]).fillna(0)
    out["condition"] = _num(b["Condition"])
    out["bedrooms"] = _num(b["Bedrooms"])
    out["bathrooms"] = (_num(b["BathFullCount"]).fillna(0) + 0.75 * _num(b["Bath3qtrCount"]).fillna(0)
                        + 0.5 * _num(b["BathHalfCount"]).fillna(0))
    out["stories"] = _num(b["Stories"])
    out["basement_sqft"] = _num(b["SqFtFinBasement"]).fillna(0)
    out["buildings_on_parcel"] = buildings.reindex(out.index)

    with zipfile.ZipFile(kc / "Parcel.zip") as zf, zf.open("EXTR_Parcel.csv") as f:
        p = pd.read_csv(f, encoding="latin-1", dtype=str, usecols=["Major", "Minor", "SqFtLot", "WfntLocation"] + VIEW_COLUMNS)
    p["property_id"] = _pin(p)
    p = p.drop_duplicates("property_id").set_index("property_id")
    parcel = pd.DataFrame({
        "lot_sqft": _num(p["SqFtLot"]),
        "view_score": sum(_num(p[c]).fillna(0) for c in VIEW_COLUMNS),
        "waterfront": (_num(p["WfntLocation"]).fillna(0) > 0).astype(int),
    })
    out = out.join(parcel, how="left")
    out.index.name = "property_id"
    return out[FEATURE_COLUMNS]


def load_table(path: Path) -> Optional[pd.DataFrame]:
    """The saved feature table, indexed by property_id, or None when the file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"property_id": str}).set_index("property_id")


def lookup(table: Optional[pd.DataFrame], property_id: Optional[str], feature_columns=FEATURE_COLUMNS) -> Optional[Dict]:
    """Features of one parcel, or None when unknown or incomplete."""
    if table is None or not property_id or property_id not in table.index:
        return None
    row = table.loc[property_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    values = {c: float(row[c]) for c in feature_columns}
    if not all(np.isfinite(v) for v in values.values()) or values["living_sqft"] <= 0:
        return None
    return values


def describe(f: Dict) -> str:
    """A short plain-English summary of the record, shown with the estimate."""
    parts = ["{:,.0f} sq ft living area".format(f["living_sqft"]),
             "built %d" % f["year_built"]]
    if f["year_renovated"] > 0:
        parts.append("renovated %d" % f["year_renovated"])
    parts += ["building grade %d of 13" % f["grade"], "condition %d of 5" % f["condition"],
              "%d bedrooms" % f["bedrooms"], "%g bathrooms" % f["bathrooms"],
              "{:,.0f} sq ft lot".format(f["lot_sqft"])]
    if f["view_score"] > 0:
        parts.append("has a view")
    if f["waterfront"] > 0:
        parts.append("waterfront")
    return ", ".join(parts)


def describe_san_francisco(f: Dict) -> str:
    return ("{living_sqft:,.0f} sq ft, built {year_built:.0f}, {rooms:.0f} rooms, "
            "{bedrooms:.0f} bedrooms, {bathrooms:g} bathrooms").format(**f)
