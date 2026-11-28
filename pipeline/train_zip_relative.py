"""A per-house model that can be used in any US city: price relative to the ZIP's typical value.

Usage:
    python -m pipeline.king_county_cities --download   # once, builds the test-city records
    python -m pipeline.train_zip_relative

Idea
----
The Seattle house model (pipeline/train.py) learns price from latitude,
longitude, month and the house record. Latitude and longitude mean nothing
outside Seattle, so that model cannot be moved to another city.

Prices depend strongly on location. Zillow already measures location for us:
it publishes the typical single-family value (ZHVI) of every US ZIP code,
every month. So this model does not predict the price. It predicts

    target = log(sale price) - log(Zillow value of the house's ZIP in the sale month)

that is, how much more or less this house is worth than a typical house in
its ZIP, from the house record only (living area, grade, age, condition,
rooms, lot, view, waterfront). No latitude, no longitude, no month.

    estimate = Zillow value of the ZIP this month * exp(predicted target)

The house record is measured against the typical house of the same ZIP
(median over every single-family parcel in the county record, no sale
prices): living area 1.3 times the ZIP median, built 20 years later than
the ZIP median, and so on. This matters: Seattle's typical house is old and
on a small lot, so a model on the raw record learns "large lot, newer house
= far above the ZIP value" and then overprices almost every suburban house,
where that is simply the typical house. The raw-record versions are kept in
the report to show this.

Input sets (served model chosen among the first two on Seattle validation):
    house_vs_zip_typical                 house record relative to the ZIP's typical house
    house_vs_zip_typical_and_zip_level   the same plus log(Zillow ZIP value)
    house_record, house_record_and_zip_level   raw record, comparison only

Prediction needs the ZIP's typical house, so a new city needs its assessor
records (which Persons 1 and 2 build) but not its sale prices.

Test
----
Train on Seattle only (date-ordered split, same as pipeline/train.py). Then
score the model on King County cities it never saw (Bellevue, Kent, ...;
see pipeline/king_county_cities.py). For a fair comparison every model is
scored on the same rows: the newest 20% of each test city's sales, which are
also later than every Seattle training sale.

Compared on those rows:
    Zillow ZIP value alone           the area average, what the app shows today outside Seattle
    Seattle house model (absolute)   pipeline/train.py's recipe, trained on Seattle, used out of city
    ZIP-relative model               this file, trained on Seattle
    Local house model (reference)    trained on that city's own older sales; an upper reference
                                     that needs local sale prices, which Portland may not have

Outputs: models/real/zip_relative.joblib, reports/zip_relative_evaluation.json,
docs/zip_relative_evaluation.md.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from core import house_features
from core.paths import PROCESSED_DIR, REAL_MODEL_DIR, REPORTS_DIR, ROOT
from pipeline import train
from pipeline.train import (INTERVAL_TARGET, MIN_CALIBRATION_ROWS, SEED, SPLIT_NAMES, assign_time_splits,
                            choose_on_validation, compare_errors, conformal_quantile, error_metrics,
                            house_candidates, sale_matrix, with_house_features)

MODEL_VERSION_PREFIX = "homelens-zip-relative"
ZIP_LEVEL = "log_zip_value"

# The typical house of a ZIP: medians over every single-family parcel in the county record for that
# ZIP (not over sales, so no sale price is used). A ZIP needs MIN_PARCELS_PER_ZIP parcels.
TYPICAL_COLUMNS = ("living_sqft", "lot_sqft", "grade", "year_built", "condition", "bedrooms", "bathrooms")
MIN_PARCELS_PER_ZIP = 30
RELATIVE_COLUMNS = (
    "rel_log_living_sqft",   # log(living area / ZIP median)
    "rel_log_lot_sqft",      # log(lot / ZIP median)
    "rel_grade",             # grade minus ZIP median grade
    "rel_year_built",        # year built minus ZIP median
    "rel_condition", "rel_bedrooms", "rel_bathrooms",
    "renovated",             # 1 when ever renovated
    "basement_share",        # finished basement / living area
    "stories", "buildings_on_parcel", "view_score", "waterfront",
)

# Raw house record: what a typical house looks like differs by city (Seattle's are older on smaller
# lots), so these are kept for comparison only. See docs/zip_relative_evaluation.md.
COMPARISON_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "house_record": tuple(house_features.FEATURE_COLUMNS),
    "house_record_and_zip_level": tuple(house_features.FEATURE_COLUMNS) + (ZIP_LEVEL,),
}
# House record measured against the ZIP's typical house. The served model is chosen among these.
SERVED_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "house_vs_zip_typical": RELATIVE_COLUMNS,
    "house_vs_zip_typical_and_zip_level": RELATIVE_COLUMNS + (ZIP_LEVEL,),
}
VARIANTS: Dict[str, Tuple[str, ...]] = {**COMPARISON_VARIANTS, **SERVED_VARIANTS}
MIN_CITY_SALES = 200       # a test city needs this many sales to be reported on its own
DOC_PATH = ROOT / "docs" / "zip_relative_evaluation.md"


# ------------------------------------------------------------------ the model

class ZipRelativeModel:
    """Predicts a dollar price from the house record and the ZIP's Zillow value.

    Fitted on rows with a 'zhvi_usd' column (the ZIP's value in the sale month)
    and the house record columns. predict() needs the same columns.
    """

    def __init__(self, regressor, feature_cols: Sequence[str]):
        self.regressor = regressor
        self.feature_cols = tuple(feature_cols)

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        return feature_matrix(df, self.feature_cols)

    def fit(self, df: pd.DataFrame, price: np.ndarray) -> "ZipRelativeModel":
        self.regressor.fit(self.matrix(df), relative_target(price, df["zhvi_usd"].to_numpy(dtype=float)))
        return self

    def predict_relative(self, df: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(self.matrix(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["zhvi_usd"].to_numpy(dtype=float) * np.exp(self.predict_relative(df))


def relative_target(price, zip_value) -> np.ndarray:
    """log(price / ZIP value): 0 for a typical house, 0.2 for one about 22% above it."""
    return np.log(np.asarray(price, dtype=float)) - np.log(np.asarray(zip_value, dtype=float))


def feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    columns = []
    for c in feature_cols:
        if c == ZIP_LEVEL:
            columns.append(np.log(df["zhvi_usd"].to_numpy(dtype=float)))
        else:
            columns.append(df[c].to_numpy(dtype=float))
    return np.column_stack(columns)


def regressor_candidates() -> Dict[str, Callable]:
    """Boosted trees on the log ratio. Same settings as the Seattle house model, without the log wrapper."""
    out: Dict[str, Callable] = {}
    for lr, leaf in ((0.05, 20), (0.03, 30), (0.05, 50)):
        out["gbm_lr%g_leaf%d" % (lr, leaf)] = lambda lr=lr, leaf=leaf: HistGradientBoostingRegressor(
            max_iter=800, learning_rate=lr, max_leaf_nodes=31, min_samples_leaf=leaf, random_state=SEED)
    return out


# ------------------------------------------------------------------ data prep

def attach_zip_value(sales: pd.DataFrame, zhvi_history: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Add 'zhvi_usd': the Zillow value of the sale's ZIP in the sale month. Rows without one are dropped."""
    df = sales.copy()
    month = pd.to_datetime(df["record_date"]).dt.strftime("%Y-%m")
    lookup = zhvi_history.assign(zip=zhvi_history["zip"].astype(str).str.zfill(5)).set_index(["zip", "month"])["zhvi_usd"]
    keys = pd.MultiIndex.from_arrays([df["zip"].astype(str).str.zfill(5), month])
    df["zhvi_usd"] = lookup.reindex(keys).to_numpy(dtype=float)
    have = np.isfinite(df["zhvi_usd"].to_numpy()) & (df["zhvi_usd"].to_numpy() > 0)
    counts = {"sales": int(len(df)), "with_a_zillow_zip_value_in_the_sale_month": int(have.sum())}
    return df[have].reset_index(drop=True), counts


def zip_typical_houses(features: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """Median house record per ZIP over every single-family parcel (index: zip, columns: TYPICAL_COLUMNS)."""
    zips = parcels.drop_duplicates("property_id").set_index("property_id")["zip"].astype(str).str.zfill(5)
    table = features[list(TYPICAL_COLUMNS)].join(zips, how="inner")
    grouped = table.groupby("zip")
    typical = grouped[list(TYPICAL_COLUMNS)].median()
    typical["parcels"] = grouped.size()
    return typical[typical["parcels"] >= MIN_PARCELS_PER_ZIP]


def add_relative_features(df: pd.DataFrame, typical: pd.DataFrame) -> pd.DataFrame:
    """Add RELATIVE_COLUMNS. Rows whose ZIP has no typical house are dropped."""
    t = typical.reindex(df["zip"].astype(str).str.zfill(5))
    keep = t["parcels"].notna().to_numpy()
    out = df[keep].copy()
    t = t[keep]
    v = lambda c: out[c].to_numpy(dtype=float)
    m = lambda c: t[c].to_numpy(dtype=float)
    out["rel_log_living_sqft"] = np.log(np.maximum(v("living_sqft"), 1) / np.maximum(m("living_sqft"), 1))
    out["rel_log_lot_sqft"] = np.log(np.maximum(v("lot_sqft"), 1) / np.maximum(m("lot_sqft"), 1))
    for c in ("grade", "year_built", "condition", "bedrooms", "bathrooms"):
        out["rel_" + c] = v(c) - m(c)
    out["renovated"] = (v("year_renovated") > 0).astype(float)
    out["basement_share"] = v("basement_sqft") / np.maximum(v("living_sqft"), 1)
    return out.reset_index(drop=True)


def prepare(sales: pd.DataFrame, features: pd.DataFrame, parcels: pd.DataFrame,
            zhvi_history: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """House record (with the built-after-sale rule of pipeline/train.py), the ZIP value, and the
    house record measured against the ZIP's typical house."""
    df, house_counts = with_house_features(sales, features)
    df, zip_counts = attach_zip_value(df, zhvi_history)
    df = add_relative_features(df, zip_typical_houses(features, parcels))
    return df, {**house_counts, "kept_with_zip_value": zip_counts["with_a_zillow_zip_value_in_the_sale_month"],
                "kept_with_a_typical_house_for_the_zip": int(len(df))}


def masks_of(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    split = df["split"].to_numpy()
    return {name: split == name for name in SPLIT_NAMES}


def split_summary(df: pd.DataFrame, masks: Dict[str, np.ndarray]) -> Dict:
    return {name: {"n": int(m.sum()),
                   "from": df.loc[m, "record_date"].min().strftime("%Y-%m-%d") if m.any() else None,
                   "to": df.loc[m, "record_date"].max().strftime("%Y-%m-%d") if m.any() else None}
            for name, m in masks.items()}


def scores(y, pred) -> Dict:
    out = error_metrics(y, pred)
    y = np.asarray(y, dtype=float)
    out["median_percent_error"] = round(100 * float(np.median(np.abs(np.asarray(pred) - y) / y)), 2) if len(y) else None
    return out


def versus(y, base, new, groups=None) -> Dict:
    """Row-by-row comparison of two models on the same rows (bootstrap, from pipeline/train.py)."""
    c = compare_errors(y, base, new, groups)
    c["verdict"] = c["verdict"].replace("the photo model", "the second model")
    return c


# ------------------------------------------------------------------ training

def fit_zip_relative(df: pd.DataFrame, masks: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
    """Choose input set and settings on the Seattle validation split, then refit on train + val."""
    y = df["price_usd"].to_numpy(dtype=float)
    val_mae: Dict[str, float] = {}
    for variant, cols in VARIANTS.items():
        for name, make in regressor_candidates().items():
            model = ZipRelativeModel(make(), cols).fit(df[masks["train"]], y[masks["train"]])
            val_mae[variant + "/" + name] = error_metrics(y[masks["val"]], model.predict(df[masks["val"]]))["mae"]
    fit = masks["train"] | masks["val"]
    fitted: Dict[str, ZipRelativeModel] = {}
    chosen_per_variant: Dict[str, str] = {}
    for variant, cols in VARIANTS.items():
        best = min((k for k in val_mae if k.startswith(variant + "/")), key=val_mae.get)
        chosen_per_variant[variant] = best.split("/", 1)[1]
        fitted[variant] = ZipRelativeModel(regressor_candidates()[chosen_per_variant[variant]](), cols).fit(
            df[fit], y[fit])
    # Raw-record variants are scored for comparison but never served (see COMPARISON_VARIANTS).
    chosen = min(SERVED_VARIANTS, key=lambda v: val_mae[v + "/" + chosen_per_variant[v]])
    report = {"validation_mae": val_mae, "settings_chosen_per_input_set": chosen_per_variant,
              "input_set_chosen_on_validation": chosen}
    return {"models": fitted, "chosen": chosen}, report


def fit_absolute_house_model(df: pd.DataFrame, masks: Dict[str, np.ndarray]) -> Tuple[object, str]:
    """pipeline/train.py's Seattle recipe: latitude, longitude, month and the house record, chosen on val."""
    cols = tuple(house_features.FEATURE_COLUMNS)
    X = sale_matrix(df, cols)
    y = df["price_usd"].to_numpy(dtype=float)
    chosen, _ = choose_on_validation(house_candidates(), X, y, masks)
    fit = masks["train"] | masks["val"]
    return house_candidates()[chosen]().fit(X[fit], y[fit]), chosen


def conformal_interval(model: ZipRelativeModel, df: pd.DataFrame, cal_mask: np.ndarray) -> Optional[Dict]:
    n_cal = int(cal_mask.sum())
    if n_cal < MIN_CALIBRATION_ROWS:
        return None
    cal = df[cal_mask]
    pred = model.predict(cal)
    q = conformal_quantile(np.abs(cal["price_usd"].to_numpy(dtype=float) - pred) / pred, INTERVAL_TARGET)
    return {"method": "split conformal on relative error, Seattle calibration split",
            "target_coverage": INTERVAL_TARGET, "relative_half_width": round(q, 4), "n_calibration": n_cal}


def coverage(model: ZipRelativeModel, df: pd.DataFrame, interval: Optional[Dict]) -> Optional[float]:
    if not interval or df.empty:
        return None
    pred = model.predict(df)
    return round(float(np.mean(np.abs(df["price_usd"].to_numpy(dtype=float) - pred) / pred
                               <= interval["relative_half_width"])), 4)


def evaluate_city(name: str, df: pd.DataFrame, zr: Dict, absolute, interval: Optional[Dict]) -> Dict:
    """Every model on the same rows: the newest 20% of this city's sales."""
    split_df, removed = assign_time_splits(df)
    masks = masks_of(split_df)
    report: Dict = {"city": name, "sales_with_record_and_zip_value": int(len(df)),
                    "splits": split_summary(split_df, masks)}
    test = split_df[masks["test"]]
    y = test["price_usd"].to_numpy(dtype=float)
    chosen = zr["models"][zr["chosen"]]
    preds = {"zillow_zip_value": test["zhvi_usd"].to_numpy(dtype=float)}
    for variant, model in zr["models"].items():
        preds["zip_relative_" + variant] = model.predict(test)
    preds["seattle_house_model_absolute"] = absolute.predict(sale_matrix(test, tuple(house_features.FEATURE_COLUMNS)))
    local_name = None
    if int((masks["train"] | masks["val"]).sum()) >= train.MIN_SALE_ROWS and masks["val"].any():
        local, local_name = fit_absolute_house_model(split_df, masks)
        preds["local_house_model_reference"] = local.predict(sale_matrix(test, tuple(house_features.FEATURE_COLUMNS)))
    report["test_rows"] = {k: scores(y, p) for k, p in preds.items()}
    report["test_median_sale_price"] = round(float(np.median(y)), 2) if len(y) else None
    report["local_reference_settings"] = local_name
    chosen_key = "zip_relative_" + zr["chosen"]
    report["zip_relative_vs_zillow"] = versus(y, preds["zillow_zip_value"], preds[chosen_key], test["zip"].to_numpy())
    report["zip_relative_vs_seattle_absolute"] = versus(y, preds["seattle_house_model_absolute"], preds[chosen_key],
                                                        test["zip"].to_numpy())
    report["interval_coverage_on_test_rows"] = coverage(chosen, test, interval)
    # More rows, every sale of the city. Earlier ones overlap Seattle's training dates, so this is secondary.
    report["all_sales"] = {"zillow_zip_value": scores(df["price_usd"], df["zhvi_usd"]),
                           chosen_key: scores(df["price_usd"], chosen.predict(df)),
                           "interval_coverage": coverage(chosen, df, interval)}
    return report


def train_zip_relative(seattle_sales: pd.DataFrame, seattle_features: pd.DataFrame, seattle_parcels: pd.DataFrame,
                       test_sales: pd.DataFrame, test_features: pd.DataFrame, test_parcels: pd.DataFrame,
                       zhvi_history: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Train on Seattle, test on Seattle's newest sales and on every test city. Returns (artifact, report)."""
    report: Dict = {"target": "log(sale price) - log(Zillow ZIP value in the sale month)",
                    "input_sets": {k: list(v) for k, v in VARIANTS.items()},
                    "served_input_sets": list(SERVED_VARIANTS),
                    "trained_on": "Seattle sales only"}
    sea, report["seattle_rows"] = prepare(seattle_sales, seattle_features, seattle_parcels, zhvi_history)
    if len(sea) < train.MIN_SALE_ROWS:
        raise ValueError("Too few Seattle sales with a house record and a ZIP value: %d" % len(sea))
    sea, removed = assign_time_splits(sea)
    masks = masks_of(sea)
    report["seattle_splits"] = split_summary(sea, masks)
    report["seattle_repeat_sale_rows_removed"] = removed

    zr, report["selection"] = fit_zip_relative(sea, masks)
    absolute, report["seattle_absolute_settings"] = fit_absolute_house_model(sea, masks)
    chosen = zr["models"][zr["chosen"]]
    interval = conformal_interval(chosen, sea, masks["cal"])
    report["interval"] = interval or {"status": "unavailable", "reason": "too few calibration rows"}

    test = sea[masks["test"]]
    y = test["price_usd"].to_numpy(dtype=float)
    preds = {"zillow_zip_value": test["zhvi_usd"].to_numpy(dtype=float),
             "seattle_house_model_absolute": absolute.predict(sale_matrix(test, tuple(house_features.FEATURE_COLUMNS)))}
    for variant, model in zr["models"].items():
        preds["zip_relative_" + variant] = model.predict(test)
    report["seattle_test"] = {k: scores(y, p) for k, p in preds.items()}
    report["seattle_test"]["interval_coverage"] = coverage(chosen, test, interval)

    tc, report["test_city_rows"] = prepare(test_sales, test_features, test_parcels, zhvi_history)
    report["test_cities"] = {}
    for name, city_df in sorted(tc.groupby("city")):
        if len(city_df) < MIN_CITY_SALES:
            report["test_cities"][name] = {"status": "too_few_sales", "sales": int(len(city_df))}
            continue
        report["test_cities"][name] = evaluate_city(name, city_df.reset_index(drop=True), zr, absolute, interval)
    report["all_test_cities"] = pooled(report["test_cities"])

    served = {}
    y_all = sea["price_usd"].to_numpy(dtype=float)
    settings = report["selection"]["settings_chosen_per_input_set"][zr["chosen"]]
    served[zr["chosen"]] = ZipRelativeModel(regressor_candidates()[settings](), VARIANTS[zr["chosen"]]).fit(sea, y_all)
    stamp = datetime.now(timezone.utc)
    artifact = {
        "model": served[zr["chosen"]],
        "input_set": zr["chosen"],
        "feature_cols": list(VARIANTS[zr["chosen"]]),
        "interval": interval,
        "model_version": "%s-%s" % (MODEL_VERSION_PREFIX, stamp.strftime("%Y%m%d")),
        "trained_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Served model refit on every Seattle sale. Test numbers describe the version fitted without "
                "Seattle's calibration and test sales.",
    }
    report["model_version"] = artifact["model_version"]
    report["trained_utc"] = artifact["trained_utc"]
    return artifact, report


def pooled(cities: Dict) -> Dict:
    """Row-weighted MAE over every reported test city, per model."""
    out: Dict[str, Dict] = {}
    for city in cities.values():
        for model, s in city.get("test_rows", {}).items():
            agg = out.setdefault(model, {"n": 0, "abs_error_sum": 0.0, "cities": 0})
            agg["n"] += s["n"]
            agg["abs_error_sum"] += s["mae"] * s["n"]
            agg["cities"] += 1
    return {m: {"n": a["n"], "cities": a["cities"], "mae": round(a["abs_error_sum"] / a["n"], 2)}
            for m, a in out.items() if a["n"]}


# ------------------------------------------------------------------ report

LABELS = {
    "zillow_zip_value": "Zillow ZIP value alone (area average)",
    "seattle_house_model_absolute": "Seattle house model, absolute (lat, lon, month, house record)",
    "zip_relative_house_record": "ZIP-relative, house record",
    "zip_relative_house_record_and_zip_level": "ZIP-relative, house record + ZIP value level",
    "zip_relative_house_vs_zip_typical": "ZIP-relative, house vs ZIP typical house",
    "zip_relative_house_vs_zip_typical_and_zip_level": "ZIP-relative, house vs ZIP typical house + ZIP value level",
    "local_house_model_reference": "Local house model (reference, trained on this city)",
}


def _money(v) -> str:
    return "" if v is None else "${:,.0f}".format(v)


def _table(rows: Dict[str, Dict]) -> List[str]:
    lines = ["| Model | Rows | MAE | RMSE | Median % error |", "|---|---:|---:|---:|---:|"]
    for key, s in rows.items():
        if not isinstance(s, dict) or "mae" not in s:
            continue
        lines.append("| %s | %d | %s | %s | %s%% |" % (LABELS.get(key, key), s["n"], _money(s["mae"]),
                                                     _money(s.get("rmse")), s.get("median_percent_error")))
    return lines


def write_markdown(report: Dict, path: Path = DOC_PATH) -> None:
    sel = report["selection"]
    sp = report["seattle_splits"]
    lines = [
        "# ZIP-relative model: evaluation report",
        "",
        "Generated by `python -m pipeline.train_zip_relative`. Do not edit by hand.",
        "",
        "- Model version: `%s`, trained (UTC) %s" % (report["model_version"], report["trained_utc"]),
        "- Target: %s. The estimate is the ZIP value times exp(prediction)." % report["target"],
        "- No latitude, longitude or month is used, so the model can be applied in any US ZIP Zillow covers.",
        "- Trained on Seattle sales only. Input set chosen on the Seattle validation split: `%s`."
        % sel["input_set_chosen_on_validation"],
        "",
        "## Seattle (same city, newest sales)",
        "",
        "Seattle sales with a house record and a ZIP value: %d. Date-ordered splits: train %d (%s to %s), "
        "val %d, cal %d, test %d (%s to %s)." % (
            report["seattle_rows"]["kept_with_a_typical_house_for_the_zip"], sp["train"]["n"], sp["train"]["from"], sp["train"]["to"],
            sp["val"]["n"], sp["cal"]["n"], sp["test"]["n"], sp["test"]["from"], sp["test"]["to"]),
        "",
    ]
    lines += _table(report["seattle_test"])
    interval = report.get("interval") or {}
    if "relative_half_width" in interval:
        lines += ["", "Prediction range: plus or minus %.1f%% (split conformal, target %d%%). Measured coverage on "
                  "the Seattle test split: %.1f%%." % (100 * interval["relative_half_width"],
                                                     100 * interval["target_coverage"],
                                                     100 * (report["seattle_test"]["interval_coverage"] or 0))]
    lines += ["", "## Other King County cities (never seen in training)", "",
              "Each city is scored on its newest 20% of sales, all later than every Seattle training sale. "
              "Every model is scored on the same rows. The local model is a reference only: it needs local "
              "sale prices, which is exactly what is missing in cities like Portland.", ""]
    lines += ["### All test cities together", "", "| Model | Cities | Rows | MAE |", "|---|---:|---:|---:|"]
    for key, s in report["all_test_cities"].items():
        lines.append("| %s | %d | %d | %s |" % (LABELS.get(key, key), s["cities"], s["n"], _money(s["mae"])))
    for name, city in report["test_cities"].items():
        lines += ["", "### %s" % name, ""]
        if city.get("status") == "too_few_sales":
            lines.append("Only %d sales, not reported." % city["sales"])
            continue
        t = city["splits"]["test"]
        lines += ["Test rows: %d sales, %s to %s. Median sale price %s." % (
            t["n"], t["from"], t["to"], _money(city["test_median_sale_price"])), ""]
        lines += _table(city["test_rows"])
        vz = city["zip_relative_vs_zillow"]
        va = city["zip_relative_vs_seattle_absolute"]
        lines += ["",
                  "ZIP-relative versus Zillow ZIP value: MAE lowered by %s, 95%% range %s to %s (whole ZIPs resampled): "
                  "%s." % (_money(vz["mae_lowered_by_usd"]), _money(vz["range_95_usd"][0]),
                           _money(vz["range_95_usd"][1]), vz["verdict"]),
                  "ZIP-relative versus the Seattle absolute model: MAE lowered by %s: %s." % (
                      _money(va["mae_lowered_by_usd"]), va["verdict"])]
        if city.get("interval_coverage_on_test_rows") is not None:
            lines.append("Coverage of the Seattle prediction range here: %.1f%% (target %d%%)." % (
                100 * city["interval_coverage_on_test_rows"], 100 * INTERVAL_TARGET))
    path.write_text("\n".join(lines).replace("$-", "-$") + "\n", encoding="utf-8")


# ------------------------------------------------------------------ main

def load_processed(processed: Path = PROCESSED_DIR) -> Dict[str, pd.DataFrame]:
    types = {"zip": str, "property_id": str}
    need = ["seattle_sales.csv", "seattle_house_features.csv", "seattle_parcels.csv", "test_city_sales.csv",
            "test_city_house_features.csv", "test_city_parcels.csv", "zhvi_history.csv"]
    missing = [n for n in need if not (processed / n).exists()]
    if missing:
        raise FileNotFoundError("Missing %s in %s. Run pipeline.build_dataset and pipeline.king_county_cities first."
                                % (", ".join(missing), processed))
    return {
        "seattle_sales": pd.read_csv(processed / "seattle_sales.csv", dtype=types),
        "seattle_features": house_features.load_table(processed / "seattle_house_features.csv"),
        "seattle_parcels": pd.read_csv(processed / "seattle_parcels.csv", dtype=types),
        "test_sales": pd.read_csv(processed / "test_city_sales.csv", dtype=types),
        "test_features": house_features.load_table(processed / "test_city_house_features.csv"),
        "test_parcels": pd.read_csv(processed / "test_city_parcels.csv", dtype=types),
        "zhvi_history": pd.read_csv(processed / "zhvi_history.csv", dtype={"zip": str}),
    }


def main() -> int:
    data = load_processed()
    artifact, report = train_zip_relative(data["seattle_sales"], data["seattle_features"], data["seattle_parcels"],
                                          data["test_sales"], data["test_features"], data["test_parcels"],
                                          data["zhvi_history"])
    REAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REAL_MODEL_DIR / "zip_relative.part"
    joblib.dump(artifact, tmp)
    tmp.replace(REAL_MODEL_DIR / "zip_relative.joblib")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "zip_relative_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report)
    print("Seattle test:")
    for k, s in report["seattle_test"].items():
        if isinstance(s, dict):
            print("  %-45s MAE %s  median %% error %s" % (k, _money(s["mae"]), s["median_percent_error"]))
    print("All test cities:")
    for k, s in report["all_test_cities"].items():
        print("  %-45s MAE %s  (%d rows)" % (k, _money(s["mae"]), s["n"]))
    print("Report written to %s" % DOC_PATH)
    return 0


if __name__ == "__main__":
    # Run through the imported module so the saved model refers to
    # pipeline.train_zip_relative.ZipRelativeModel, not __main__.ZipRelativeModel,
    # and can be loaded by the backend and other scripts.
    from pipeline import train_zip_relative as _module
    sys.exit(_module.main())
