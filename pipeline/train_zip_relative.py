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
its ZIP, from the house record only. No latitude, no longitude, no month.

    estimate = Zillow value of the ZIP this month * exp(predicted target)

The house record is measured against typical houses nearby, never used raw:
    vs the ZIP          median over every single-family parcel in the ZIP
    vs the neighbours   median over the NEIGHBORS_K nearest single-family parcels
Both come from the county's house records, never from sale prices. This
matters: Seattle's typical house is old and on a small lot, so a model on the
raw record learns "large lot, newer house = far above the ZIP value" and
overprices almost every suburban house, where that is simply the typical
house. The raw-record versions are kept in the report to show this.

Prediction needs typical houses, so a new city needs its assessor records
with parcel positions (which Persons 1 and 2 build) but not its sale prices.

Evaluation
----------
Train on Seattle only (date-ordered split, same as pipeline/train.py). Other
King County cities are split in two groups, fixed in TUNING_CITIES before
any of the choices below were tried:
    tuning cities       used to choose the input set and to size the prediction range
    final test cities   scored once, after every choice is made
Every city is scored on its newest 20% of sales, all later than every Seattle
training sale. Compared on the same rows:
    Zillow ZIP value alone           the area average, what the app shows today outside Seattle
    Seattle house model (absolute)   pipeline/train.py's recipe used out of city. It uses Seattle's
                                     latitude and longitude, so it says nothing for SF or Portland
    ZIP-relative model               this file
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
from sklearn.neighbors import NearestNeighbors

from core import house_features
from core.features import to_xyz_km
from core.paths import PROCESSED_DIR, REAL_MODEL_DIR, REPORTS_DIR, ROOT
from pipeline import train
from pipeline.train import (INTERVAL_TARGET, MIN_CALIBRATION_ROWS, SEED, SPLIT_NAMES, assign_time_splits,
                            choose_on_validation, compare_errors, conformal_quantile, error_metrics,
                            house_candidates, sale_matrix, with_house_features)

MODEL_VERSION_PREFIX = "homelens-zip-relative"
ZIP_LEVEL = "log_zip_value"

# Cities used for choices. Fixed before the changes they are used to choose between were tried.
# Every other test city with enough sales is a final test city, scored once.
TUNING_CITIES = ("Bellevue", "Kent", "Renton")

# Typical houses: medians of these columns over single-family parcels (county record, no sale prices).
TYPICAL_COLUMNS = ("living_sqft", "lot_sqft", "grade", "year_built", "condition", "bedrooms", "bathrooms")
MIN_PARCELS_PER_ZIP = 30
NEIGHBORS_K = 50


def _relative_names(prefix: str) -> Tuple[str, ...]:
    return ("%s_log_living_sqft" % prefix, "%s_log_lot_sqft" % prefix,
            *("%s_%s" % (prefix, c) for c in ("grade", "year_built", "condition", "bedrooms", "bathrooms")))


ZIP_RELATIVE = _relative_names("rel")         # house vs the ZIP's typical house
NEIGHBOR_RELATIVE = _relative_names("nb")     # house vs its NEIGHBORS_K nearest parcels
OTHER_COLUMNS = (
    "renovated",             # 1 when ever renovated
    "basement_share",        # finished basement / living area
    "stories", "buildings_on_parcel", "view_score", "waterfront",
)
RELATIVE_COLUMNS = ZIP_RELATIVE + OTHER_COLUMNS   # kept for code that imported the first version

# Raw house record: what a typical house looks like differs by city, so these are comparison only.
COMPARISON_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "house_record": tuple(house_features.FEATURE_COLUMNS),
    "house_record_and_zip_level": tuple(house_features.FEATURE_COLUMNS) + (ZIP_LEVEL,),
}
# The served model is chosen among these on the tuning cities.
SERVED_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "house_vs_zip_typical": ZIP_RELATIVE + OTHER_COLUMNS,
    "house_vs_zip_typical_and_zip_level": ZIP_RELATIVE + OTHER_COLUMNS + (ZIP_LEVEL,),
    "house_vs_neighbors": NEIGHBOR_RELATIVE + OTHER_COLUMNS,
    "house_vs_neighbors_and_zip_level": NEIGHBOR_RELATIVE + OTHER_COLUMNS + (ZIP_LEVEL,),
    "house_vs_zip_and_neighbors_and_zip_level": ZIP_RELATIVE + NEIGHBOR_RELATIVE + OTHER_COLUMNS + (ZIP_LEVEL,),
}
VARIANTS: Dict[str, Tuple[str, ...]] = {**COMPARISON_VARIANTS, **SERVED_VARIANTS}
MIN_CITY_SALES = 200       # a test city needs this many sales to be reported on its own
DOC_PATH = ROOT / "docs" / "zip_relative_evaluation.md"


# ------------------------------------------------------------------ the model

class ZipRelativeModel:
    """Predicts a dollar price from the house record and the ZIP's Zillow value.

    Fitted on rows with a 'zhvi_usd' column (the ZIP's value in the sale month)
    and the feature columns made by prepare(). predict() needs the same columns.
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


def _add_relative(out: pd.DataFrame, typical: Dict[str, np.ndarray], prefix: str) -> None:
    v = lambda c: out[c].to_numpy(dtype=float)
    out[prefix + "_log_living_sqft"] = np.log(np.maximum(v("living_sqft"), 1) / np.maximum(typical["living_sqft"], 1))
    out[prefix + "_log_lot_sqft"] = np.log(np.maximum(v("lot_sqft"), 1) / np.maximum(typical["lot_sqft"], 1))
    for c in ("grade", "year_built", "condition", "bedrooms", "bathrooms"):
        out[prefix + "_" + c] = v(c) - typical[c]


def add_relative_features(df: pd.DataFrame, typical: pd.DataFrame) -> pd.DataFrame:
    """Add ZIP_RELATIVE and OTHER_COLUMNS. Rows whose ZIP has no typical house are dropped."""
    t = typical.reindex(df["zip"].astype(str).str.zfill(5))
    keep = t["parcels"].notna().to_numpy()
    out = df[keep].copy()
    t = t[keep]
    _add_relative(out, {c: t[c].to_numpy(dtype=float) for c in TYPICAL_COLUMNS}, "rel")
    out["renovated"] = (out["year_renovated"].to_numpy(dtype=float) > 0).astype(float)
    out["basement_share"] = out["basement_sqft"].to_numpy(dtype=float) / np.maximum(out["living_sqft"].to_numpy(dtype=float), 1)
    return out.reset_index(drop=True)


def add_neighbor_features(df: pd.DataFrame, features: pd.DataFrame, parcels: pd.DataFrame,
                          k: int = NEIGHBORS_K) -> pd.DataFrame:
    """Add NEIGHBOR_RELATIVE: the house against the median of its k nearest single-family parcels.

    The house's own parcel is never one of its neighbours. Positions only pick the neighbours;
    they are not model inputs.
    """
    pos = parcels.drop_duplicates("property_id").set_index("property_id")[["lat", "lon"]]
    table = features[list(TYPICAL_COLUMNS)].join(pos, how="inner").dropna()
    if len(table) <= k:
        raise ValueError("Need more than %d parcels with a house record and a position" % k)
    index = NearestNeighbors(n_neighbors=k + 1).fit(to_xyz_km(table[["lat", "lon"]].to_numpy(dtype=float)))
    _, idx = index.kneighbors(to_xyz_km(df[["lat", "lon"]].to_numpy(dtype=float)))
    values = table[list(TYPICAL_COLUMNS)].to_numpy(dtype=float)[idx]           # (rows, k + 1, columns)
    is_self = table.index.to_numpy()[idx] == df["property_id"].to_numpy()[:, None]
    # Drop the house itself, or the farthest neighbour when the house is not in the table.
    drop = is_self.copy()
    drop[~is_self.any(axis=1), -1] = True
    values[drop] = np.nan
    medians = np.nanmedian(values, axis=1)
    out = df.copy()
    _add_relative(out, {c: medians[:, i] for i, c in enumerate(TYPICAL_COLUMNS)}, "nb")
    return out


def prepare(sales: pd.DataFrame, features: pd.DataFrame, parcels: pd.DataFrame,
            zhvi_history: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """House record (with the built-after-sale rule of pipeline/train.py), the ZIP value, and the
    house record measured against the ZIP's typical house and against its neighbours."""
    df, house_counts = with_house_features(sales, features)
    df, zip_counts = attach_zip_value(df, zhvi_history)
    df = add_relative_features(df, zip_typical_houses(features, parcels))
    df = add_neighbor_features(df, features, parcels)
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

def fit_zip_relative(df: pd.DataFrame, masks: Dict[str, np.ndarray]) -> Tuple[Dict[str, "ZipRelativeModel"], Dict]:
    """Per input set: choose tree settings on the Seattle validation split, then refit on train + val."""
    y = df["price_usd"].to_numpy(dtype=float)
    val_mae: Dict[str, float] = {}
    for variant, cols in VARIANTS.items():
        for name, make in regressor_candidates().items():
            model = ZipRelativeModel(make(), cols).fit(df[masks["train"]], y[masks["train"]])
            val_mae[variant + "/" + name] = error_metrics(y[masks["val"]], model.predict(df[masks["val"]]))["mae"]
    fit = masks["train"] | masks["val"]
    fitted: Dict[str, ZipRelativeModel] = {}
    settings: Dict[str, str] = {}
    for variant, cols in VARIANTS.items():
        best = min((k for k in val_mae if k.startswith(variant + "/")), key=val_mae.get)
        settings[variant] = best.split("/", 1)[1]
        fitted[variant] = ZipRelativeModel(regressor_candidates()[settings[variant]](), cols).fit(df[fit], y[fit])
    return fitted, {"seattle_validation_mae": val_mae, "settings_chosen_per_input_set": settings}


def fit_absolute_house_model(df: pd.DataFrame, masks: Dict[str, np.ndarray]) -> Tuple[object, str]:
    """pipeline/train.py's Seattle recipe: latitude, longitude, month and the house record, chosen on val."""
    cols = tuple(house_features.FEATURE_COLUMNS)
    X = sale_matrix(df, cols)
    y = df["price_usd"].to_numpy(dtype=float)
    chosen, _ = choose_on_validation(house_candidates(), X, y, masks)
    fit = masks["train"] | masks["val"]
    return house_candidates()[chosen]().fit(X[fit], y[fit]), chosen


def conformal_interval(prices: np.ndarray, pred: np.ndarray, source: str) -> Optional[Dict]:
    """Split-conformal range on relative error from calibration rows."""
    if len(prices) < MIN_CALIBRATION_ROWS:
        return None
    q = conformal_quantile(np.abs(np.asarray(prices, dtype=float) - pred) / pred, INTERVAL_TARGET)
    return {"method": "split conformal on relative error", "calibrated_on": source,
            "target_coverage": INTERVAL_TARGET, "relative_half_width": round(q, 4), "n_calibration": int(len(prices))}


def coverage(prices, pred, interval: Optional[Dict]) -> Optional[float]:
    if not interval or len(prices) == 0:
        return None
    prices = np.asarray(prices, dtype=float)
    return round(float(np.mean(np.abs(prices - pred) / pred <= interval["relative_half_width"])), 4)


def city_test_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """A city's sales with date-ordered splits. Its test split is the newest 20%."""
    split_df, _ = assign_time_splits(df)
    return split_df, masks_of(split_df)


def evaluate_city(name: str, role: str, df: pd.DataFrame, models: Dict[str, "ZipRelativeModel"], chosen: str,
                  absolute, intervals: Dict[str, Optional[Dict]]) -> Dict:
    """Every model on the same rows: the newest 20% of this city's sales."""
    split_df, masks = city_test_rows(df)
    report: Dict = {"city": name, "role": role, "sales_with_record_and_zip_value": int(len(df)),
                    "splits": split_summary(split_df, masks)}
    test = split_df[masks["test"]]
    y = test["price_usd"].to_numpy(dtype=float)
    preds = {"zillow_zip_value": test["zhvi_usd"].to_numpy(dtype=float)}
    for variant, model in models.items():
        preds["zip_relative_" + variant] = model.predict(test)
    preds["seattle_house_model_absolute"] = absolute.predict(sale_matrix(test, tuple(house_features.FEATURE_COLUMNS)))
    local_name = None
    if int((masks["train"] | masks["val"]).sum()) >= train.MIN_SALE_ROWS and masks["val"].any():
        local, local_name = fit_absolute_house_model(split_df, masks)
        preds["local_house_model_reference"] = local.predict(sale_matrix(test, tuple(house_features.FEATURE_COLUMNS)))
    report["test_rows"] = {k: scores(y, p) for k, p in preds.items()}
    report["test_median_sale_price"] = round(float(np.median(y)), 2) if len(y) else None
    report["local_reference_settings"] = local_name
    chosen_key = "zip_relative_" + chosen
    report["zip_relative_vs_zillow"] = versus(y, preds["zillow_zip_value"], preds[chosen_key], test["zip"].to_numpy())
    report["zip_relative_vs_seattle_absolute"] = versus(y, preds["seattle_house_model_absolute"], preds[chosen_key],
                                                        test["zip"].to_numpy())
    report["interval_coverage_on_test_rows"] = {k: coverage(y, preds[chosen_key], v) for k, v in intervals.items()}
    return report


def train_zip_relative(seattle_sales: pd.DataFrame, seattle_features: pd.DataFrame, seattle_parcels: pd.DataFrame,
                       test_sales: pd.DataFrame, test_features: pd.DataFrame, test_parcels: pd.DataFrame,
                       zhvi_history: pd.DataFrame,
                       tuning_cities: Sequence[str] = TUNING_CITIES) -> Tuple[Dict, Dict]:
    """Train on Seattle, choose on the tuning cities, score once on the final test cities.

    Returns (artifact, report).
    """
    report: Dict = {"target": "log(sale price) - log(Zillow ZIP value in the sale month)",
                    "input_sets": {k: list(v) for k, v in VARIANTS.items()},
                    "served_input_sets": list(SERVED_VARIANTS),
                    "neighbors_k": NEIGHBORS_K,
                    "trained_on": "Seattle sales only"}

    # 1. Seattle: date-ordered splits, models fitted on train + val.
    sea, report["seattle_rows"] = prepare(seattle_sales, seattle_features, seattle_parcels, zhvi_history)
    if len(sea) < train.MIN_SALE_ROWS:
        raise ValueError("Too few Seattle sales with a house record and a ZIP value: %d" % len(sea))
    sea, removed = assign_time_splits(sea)
    masks = masks_of(sea)
    report["seattle_splits"] = split_summary(sea, masks)
    report["seattle_repeat_sale_rows_removed"] = removed
    models, report["selection"] = fit_zip_relative(sea, masks)
    absolute, report["seattle_absolute_settings"] = fit_absolute_house_model(sea, masks)

    # 2. Test cities: which are for tuning and which are scored once.
    tc, report["test_city_rows"] = prepare(test_sales, test_features, test_parcels, zhvi_history)
    cities = {name: g.reset_index(drop=True) for name, g in sorted(tc.groupby("city"))}
    small = {name: int(len(g)) for name, g in cities.items() if len(g) < MIN_CITY_SALES}
    cities = {name: g for name, g in cities.items() if name not in small}
    tuning = [c for c in cities if c in set(tuning_cities)]
    final = [c for c in cities if c not in set(tuning_cities)]
    if not tuning or not final:
        raise ValueError("Need at least one tuning city and one final test city, have %s and %s" % (tuning, final))
    report["tuning_cities"], report["final_test_cities"], report["cities_with_too_few_sales"] = tuning, final, small

    # 3. Choose the input set on the tuning cities' newest sales (pooled MAE).
    tuning_rows = pd.concat([city_test_rows(cities[c])[0].pipe(lambda d: d[d["split"] == "test"]) for c in tuning],
                            ignore_index=True)
    y_tune = tuning_rows["price_usd"].to_numpy(dtype=float)
    tuning_mae = {v: error_metrics(y_tune, models[v].predict(tuning_rows))["mae"] for v in VARIANTS}
    chosen = min(SERVED_VARIANTS, key=tuning_mae.get)
    report["selection"].update({"tuning_mae_per_input_set": tuning_mae, "tuning_rows": int(len(tuning_rows)),
                                "input_set_chosen_on_tuning_cities": chosen})

    # 4. Prediction range. Seattle's calibration split (as in pipeline/train.py) versus the tuning cities,
    #    whose errors are those of a model used in a city it was not trained on.
    sea_cal = sea[masks["cal"]]
    intervals = {
        "seattle_calibration": conformal_interval(sea_cal["price_usd"].to_numpy(dtype=float),
                                                  models[chosen].predict(sea_cal), "Seattle calibration split"),
        "tuning_cities": conformal_interval(y_tune, models[chosen].predict(tuning_rows),
                                            "tuning cities' newest sales: " + ", ".join(tuning)),
    }
    report["intervals"] = intervals

    # 5. Seattle test split, then every city.
    test = sea[masks["test"]]
    y = test["price_usd"].to_numpy(dtype=float)
    preds = {"zillow_zip_value": test["zhvi_usd"].to_numpy(dtype=float),
             "seattle_house_model_absolute": absolute.predict(sale_matrix(test, tuple(house_features.FEATURE_COLUMNS)))}
    for variant, model in models.items():
        preds["zip_relative_" + variant] = model.predict(test)
    report["seattle_test"] = {k: scores(y, p) for k, p in preds.items()}
    report["seattle_test_interval_coverage"] = {k: coverage(y, preds["zip_relative_" + chosen], v)
                                                for k, v in intervals.items()}
    report["cities"] = {name: evaluate_city(name, "tuning" if name in tuning else "final_test", cities[name],
                                            models, chosen, absolute, intervals) for name in cities}
    report["all_tuning_cities"] = pooled({c: report["cities"][c] for c in tuning})
    report["all_final_test_cities"] = pooled({c: report["cities"][c] for c in final})
    report["all_final_test_cities_interval_coverage"] = pooled_coverage({c: report["cities"][c] for c in final})

    # 6. Served model: the chosen recipe refit on every Seattle sale, with the tuning-city range.
    settings = report["selection"]["settings_chosen_per_input_set"][chosen]
    served = ZipRelativeModel(regressor_candidates()[settings](), VARIANTS[chosen]).fit(
        sea, sea["price_usd"].to_numpy(dtype=float))
    stamp = datetime.now(timezone.utc)
    artifact = {
        "model": served,
        "input_set": chosen,
        "feature_cols": list(VARIANTS[chosen]),
        "neighbors_k": NEIGHBORS_K,
        "interval": intervals["tuning_cities"],
        "model_version": "%s-%s" % (MODEL_VERSION_PREFIX, stamp.strftime("%Y%m%d")),
        "trained_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Served model refit on every Seattle sale. Test numbers describe the version fitted without "
                "Seattle's calibration and test sales. Inputs are made by prepare().",
    }
    report["model_version"] = artifact["model_version"]
    report["trained_utc"] = artifact["trained_utc"]
    return artifact, report


def pooled(cities: Dict) -> Dict:
    """Row-weighted MAE over the given cities, per model."""
    out: Dict[str, Dict] = {}
    for city in cities.values():
        for model, s in city.get("test_rows", {}).items():
            agg = out.setdefault(model, {"n": 0, "abs_error_sum": 0.0, "cities": 0})
            agg["n"] += s["n"]
            agg["abs_error_sum"] += s["mae"] * s["n"]
            agg["cities"] += 1
    return {m: {"n": a["n"], "cities": a["cities"], "mae": round(a["abs_error_sum"] / a["n"], 2)}
            for m, a in out.items() if a["n"]}


def pooled_coverage(cities: Dict) -> Dict:
    """Row-weighted interval coverage of the chosen model over the given cities."""
    out: Dict[str, float] = {}
    for key in ("seattle_calibration", "tuning_cities"):
        hits = n = 0.0
        for city in cities.values():
            cov = city["interval_coverage_on_test_rows"].get(key)
            rows = city["splits"]["test"]["n"]
            if cov is not None:
                hits += cov * rows
                n += rows
        out[key] = round(hits / n, 4) if n else None
    return out


# ------------------------------------------------------------------ report

LABELS = {
    "zillow_zip_value": "Zillow ZIP value alone (area average)",
    "seattle_house_model_absolute": "Seattle house model, absolute (lat, lon, month, house record)",
    "zip_relative_house_record": "ZIP-relative, raw house record (comparison)",
    "zip_relative_house_record_and_zip_level": "ZIP-relative, raw house record + ZIP value level (comparison)",
    "zip_relative_house_vs_zip_typical": "ZIP-relative, vs ZIP typical house",
    "zip_relative_house_vs_zip_typical_and_zip_level": "ZIP-relative, vs ZIP typical house + ZIP value level",
    "zip_relative_house_vs_neighbors": "ZIP-relative, vs %d nearest houses" % NEIGHBORS_K,
    "zip_relative_house_vs_neighbors_and_zip_level": "ZIP-relative, vs %d nearest houses + ZIP value level"
                                                     % NEIGHBORS_K,
    "zip_relative_house_vs_zip_and_neighbors_and_zip_level": "ZIP-relative, vs ZIP and nearest houses + ZIP value level",
    "local_house_model_reference": "Local house model (reference, trained on this city)",
}


def _money(v) -> str:
    return "" if v is None else "${:,.0f}".format(v)


def _pct(v) -> str:
    return "" if v is None else "%.1f%%" % (100 * v)


def _label(key: str, chosen: str) -> str:
    text = LABELS.get(key, key)
    return "**%s (chosen)**" % text if key == "zip_relative_" + chosen else text


def _table(rows: Dict[str, Dict], chosen: str) -> List[str]:
    lines = ["| Model | Rows | MAE | RMSE | Median % error |", "|---|---:|---:|---:|---:|"]
    for key, s in rows.items():
        if not isinstance(s, dict) or "mae" not in s:
            continue
        lines.append("| %s | %d | %s | %s | %s%% |" % (_label(key, chosen), s["n"], _money(s["mae"]),
                                                     _money(s.get("rmse")), s.get("median_percent_error")))
    return lines


def _pooled_table(rows: Dict[str, Dict], chosen: str) -> List[str]:
    lines = ["| Model | Cities | Rows | MAE |", "|---|---:|---:|---:|"]
    for key, s in rows.items():
        lines.append("| %s | %d | %d | %s |" % (_label(key, chosen), s["cities"], s["n"], _money(s["mae"])))
    return lines


def _city_section(name: str, city: Dict, chosen: str) -> List[str]:
    t = city["splits"]["test"]
    lines = ["", "### %s" % name, "",
             "Test rows: %d sales, %s to %s. Median sale price %s." % (
                 t["n"], t["from"], t["to"], _money(city["test_median_sale_price"])), ""]
    lines += _table(city["test_rows"], chosen)
    vz, va = city["zip_relative_vs_zillow"], city["zip_relative_vs_seattle_absolute"]
    cov = city["interval_coverage_on_test_rows"]
    lines += ["",
              "Chosen model versus Zillow ZIP value: MAE lowered by %s, 95%% range %s to %s (whole ZIPs resampled): "
              "%s." % (_money(vz["mae_lowered_by_usd"]), _money(vz["range_95_usd"][0]),
                       _money(vz["range_95_usd"][1]), vz["verdict"]),
              "Chosen model versus the Seattle absolute model: MAE lowered by %s: %s." % (
                  _money(va["mae_lowered_by_usd"]), va["verdict"]),
              "Range coverage (target %d%%): %s with the Seattle-calibrated range, %s with the tuning-city range." % (
                  100 * INTERVAL_TARGET, _pct(cov.get("seattle_calibration")), _pct(cov.get("tuning_cities")))]
    return lines


def write_markdown(report: Dict, path: Path = DOC_PATH) -> None:
    sel = report["selection"]
    chosen = sel["input_set_chosen_on_tuning_cities"]
    sp = report["seattle_splits"]
    iv = report["intervals"]
    lines = [
        "# ZIP-relative model: evaluation report",
        "",
        "Generated by `python -m pipeline.train_zip_relative`. Do not edit by hand.",
        "",
        "- Model version: `%s`, trained (UTC) %s" % (report["model_version"], report["trained_utc"]),
        "- Target: %s. The estimate is the ZIP value times exp(prediction)." % report["target"],
        "- No latitude, longitude or month is a model input, so the model can be applied in any US ZIP Zillow "
        "covers, given the county's house records for that area.",
        "- Trained on Seattle sales only.",
        "- Tuning cities (used to choose the input set and to size the range): %s." % ", ".join(report["tuning_cities"]),
        "- Final test cities (scored once, after every choice): %s." % ", ".join(report["final_test_cities"]),
        "- Chosen input set: `%s`." % chosen,
        "",
        "Note on honesty: an earlier version of this report scored all seven cities together. Its result (the raw "
        "house record overprices suburban houses) led to measuring houses against typical houses. The tuning and "
        "final groups above were fixed after that and before the neighbour comparison, the input set choice and "
        "the range calibration were tried.",
        "",
        "## Final test cities (never used for any choice)",
        "",
    ]
    lines += _pooled_table(report["all_final_test_cities"], chosen)
    pc = report["all_final_test_cities_interval_coverage"]
    lines += ["", "Prediction range of the chosen model, target coverage %d%%:" % (100 * INTERVAL_TARGET), "",
              "| Range calibrated on | Half width | Coverage on final test cities |", "|---|---:|---:|"]
    for key, label in (("seattle_calibration", "Seattle calibration split"),
                       ("tuning_cities", "Tuning cities (served)")):
        if iv.get(key):
            lines.append("| %s | plus or minus %s | %s |" % (label, _pct(iv[key]["relative_half_width"]), _pct(pc.get(key))))
    lines += ["", "Each city is scored on its newest 20% of sales, all later than every Seattle training sale. Every "
              "model is scored on the same rows. The Seattle absolute model uses Seattle's latitude and longitude, "
              "so it is only meaningful near Seattle and is not an option for San Francisco or Portland. The local "
              "model is a reference only: it needs local sale prices."]
    for name in report["final_test_cities"]:
        lines += _city_section(name, report["cities"][name], chosen)

    lines += ["", "## Tuning cities (used for the choices above)", "",
              "Input set chosen by pooled MAE over the newest sales of these cities (%d rows):" % sel["tuning_rows"], "",
              "| Input set | Tuning MAE | Seattle validation MAE |", "|---|---:|---:|"]
    for v in VARIANTS:
        seattle_val = sel["seattle_validation_mae"][v + "/" + sel["settings_chosen_per_input_set"][v]]
        lines.append("| %s | %s | %s |" % (_label("zip_relative_" + v, chosen), _money(sel["tuning_mae_per_input_set"][v]),
                                          _money(seattle_val)))
    lines += [""] + _pooled_table(report["all_tuning_cities"], chosen)
    for name in report["tuning_cities"]:
        lines += _city_section(name, report["cities"][name], chosen)

    lines += ["", "## Seattle (same city, newest sales)", "",
              "Seattle sales with a house record and a ZIP value: %d. Date-ordered splits: train %d (%s to %s), "
              "val %d, cal %d, test %d (%s to %s)." % (
                  report["seattle_rows"]["kept_with_a_typical_house_for_the_zip"], sp["train"]["n"],
                  sp["train"]["from"], sp["train"]["to"], sp["val"]["n"], sp["cal"]["n"], sp["test"]["n"],
                  sp["test"]["from"], sp["test"]["to"]), ""]
    lines += _table(report["seattle_test"], chosen)
    sc = report["seattle_test_interval_coverage"]
    lines += ["", "Range coverage on the Seattle test split: %s with the Seattle-calibrated range, %s with the "
              "tuning-city range." % (_pct(sc.get("seattle_calibration")), _pct(sc.get("tuning_cities")))]
    if report.get("cities_with_too_few_sales"):
        lines += ["", "Cities with fewer than %d sales, not reported: %s." % (
            MIN_CITY_SALES, ", ".join("%s (%d)" % kv for kv in report["cities_with_too_few_sales"].items()))]
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
    sel = report["selection"]
    print("Chosen on tuning cities: %s" % sel["input_set_chosen_on_tuning_cities"])
    print("Tuning MAE per input set:")
    for k, v in sel["tuning_mae_per_input_set"].items():
        print("  %-45s %s" % (k, _money(v)))
    print("Final test cities (%s):" % ", ".join(report["final_test_cities"]))
    for k, s in report["all_final_test_cities"].items():
        print("  %-55s MAE %s  (%d rows)" % (k, _money(s["mae"]), s["n"]))
    print("Range half width: %s" % {k: (v or {}).get("relative_half_width") for k, v in report["intervals"].items()})
    print("Range coverage on final test cities: %s" % report["all_final_test_cities_interval_coverage"])
    print("Report written to %s" % DOC_PATH)
    return 0


if __name__ == "__main__":
    # Run through the imported module so the saved model refers to
    # pipeline.train_zip_relative.ZipRelativeModel, not __main__.ZipRelativeModel,
    # and can be loaded by the backend and other scripts.
    from pipeline import train_zip_relative as _module
    sys.exit(_module.main())
