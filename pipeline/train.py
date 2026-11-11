"""Train and evaluate every HomeLens model.

Usage:
    python -m pipeline.train              # real data from data/processed  -> models/real
    python -m pipeline.train --fixtures   # synthetic fixtures             -> models/demo

Three kinds of model are trained, each separately for its target:

1. Area models (sale value and monthly rent). One row per ZIP. Input is the
   latitude and longitude of a point, output is the typical single-family
   home value (or rent index) around it. Labels are Zillow ZIP values.
2. Seattle per-house sale model. One row per real sale. Input is latitude,
   longitude and month, output is a sale price.
3. Seattle photo model (only when matched photos exist). It predicts what
   the location model missed, from a photo embedding.

No model sees bedrooms, bathrooms, floor area or anything else the app
cannot know when a user asks for an estimate.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from core.embed import ENCODER_NAME
from core.features import FEATURE_VERSION, month_index_from_dates, to_xyz_km, xyz_only
from core.paths import DEMO_MODEL_DIR, FIXTURES_DIR, PROCESSED_DIR, REAL_MODEL_DIR, REPORTS_DIR

SEED = 0
AREA_K_VALUES = (2, 3, 4, 5, 6, 8, 10)
SALE_K_VALUES = (5, 10, 20, 30, 50)
SPLIT_NAMES = ("train", "val", "cal", "test")
SPLIT_FRACTIONS = (0.60, 0.10, 0.10, 0.20)
INTERVAL_TARGET = 0.80
MIN_CALIBRATION_ROWS = 100
MIN_AREA_ROWS = 20
MIN_SALE_ROWS = 200
MIN_IMAGE_FIT_ROWS = 150
MIN_IMAGE_EVAL_ROWS = 30
IMAGE_PCA_SIZES = (8, 16, 32)
IMAGE_RIDGE_ALPHAS = (1.0, 10.0, 100.0)
# The photo model must lower the error by at least this share before the app uses it.
# A smaller gain is treated as a tie, and a tie is not a reason to add a second model.
SERVE_MIN_IMPROVEMENT = 0.02
BOOTSTRAP_ROUNDS = 2000


# ---------------------------------------------------------------- helpers

def error_metrics(y_true, y_pred) -> Dict:
    """Mean absolute error and root mean squared error, in dollars."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "rmse": None}
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
    }


def with_log_target(regressor):
    """Train on log(price) and convert predictions back to dollars."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def file_hash(paths: List[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def conformal_quantile(scores: np.ndarray, target: float) -> float:
    """Split-conformal quantile: the score that at least `target` of new rows should stay under.

    With n calibration scores it is the k-th smallest one, where k = ceil((n + 1) * target).
    """
    ordered = np.sort(np.asarray(scores, dtype=float))
    n = len(ordered)
    k = min(n, math.ceil((n + 1) * target))
    return float(ordered[k - 1])


# ------------------------------------------------------------ area models

def compare_errors(y_true, base_pred, photo_pred, groups=None) -> Dict:
    """How much lower is the photo model's MAE, and could that be chance?

    The same rows are scored by both models, so the two errors are compared
    row by row. The rows are resampled BOOTSTRAP_ROUNDS times (whole groups
    at a time when groups are given, because rows of one group share a
    label) to get a 95% range for the difference in MAE.
    """
    y_true = np.asarray(y_true, dtype=float)
    gain = np.abs(np.asarray(base_pred, dtype=float) - y_true) - np.abs(np.asarray(photo_pred, dtype=float) - y_true)
    base_mae = float(np.mean(np.abs(np.asarray(base_pred, dtype=float) - y_true)))
    rng = np.random.default_rng(SEED)
    if groups is None:
        picks = rng.integers(0, len(gain), size=(BOOTSTRAP_ROUNDS, len(gain)))
        means = gain[picks].mean(axis=1)
    else:
        groups = np.asarray(groups)
        names = np.unique(groups)
        sums = np.array([gain[groups == g].sum() for g in names])
        counts = np.array([(groups == g).sum() for g in names])
        picks = rng.integers(0, len(names), size=(BOOTSTRAP_ROUNDS, len(names)))
        means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    if low > 0:
        verdict = "the photo model is better"
    elif high < 0:
        verdict = "the photo model is worse"
    else:
        verdict = "no measurable difference"
    return {
        "mae_lowered_by_usd": round(float(gain.mean()), 2),
        "mae_lowered_by_share": round(float(gain.mean()) / base_mae, 4) if base_mae > 0 else None,
        "range_95_usd": [round(float(low), 2), round(float(high), 2)],
        "resampled": "rows" if groups is None else "whole ZIPs",
        "verdict": verdict,
    }


def area_candidates() -> Dict[str, Callable]:
    """Candidate area models. Each takes [latitude, longitude] rows."""
    candidates: Dict[str, Callable] = {}
    for k in AREA_K_VALUES:
        candidates["knn_k%d" % k] = lambda k=k: with_log_target(Pipeline([
            ("xyz", FunctionTransformer(to_xyz_km)),
            ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance")),
        ]))
    # Trees split on one column at a time, so they can use raw latitude and longitude.
    candidates["gbm"] = lambda: with_log_target(
        HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                      min_samples_leaf=5, random_state=SEED))
    return candidates


def _cv_mae(make_model: Callable, X: np.ndarray, y: np.ndarray, n_splits: int) -> float:
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    errors = []
    for tr, te in folds.split(X):
        model = make_model().fit(X[tr], y[tr])
        errors.append(np.mean(np.abs(model.predict(X[te]) - y[te])))
    return float(np.mean(errors))


def _usable(candidates: Dict[str, Callable], smallest_train: int) -> Dict[str, Callable]:
    """Drop KNN candidates that ask for more neighbors than there are training rows."""
    out = {}
    for name, make in candidates.items():
        if name.startswith("knn_k") and int(name[5:]) > smallest_train:
            continue
        out[name] = make
    return out


def train_area_model(table: pd.DataFrame, value_col: str) -> Tuple[Optional[object], Dict]:
    """Train one area model with nested cross-validation.

    Every ZIP is held out once. The model used for a held-out fold is chosen
    using only the other folds, so the reported error is for the whole
    "choose a model, then fit it" procedure on ZIPs it never saw.
    """
    data = table.dropna(subset=[value_col, "lat", "lon"]).reset_index(drop=True)
    report: Dict = {"n_zips": int(len(data)), "label": value_col}
    if len(data) < MIN_AREA_ROWS:
        report["status"] = "insufficient_data"
        return None, report

    X = data[["lat", "lon"]].to_numpy(dtype=float)
    y = data[value_col].to_numpy(dtype=float)
    metro = data["metro_id"].to_numpy()
    n_outer = 5
    inner_splits = 4
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    names = ["baseline_metro_median", "location_knn", "location_gbm", "selected"]
    oof = {name: np.zeros(len(data)) for name in names}
    picks: List[str] = []

    for tr, te in outer.split(X):
        # Baseline: median of the training ZIPs in the same metro area.
        for m in np.unique(metro[te]):
            same = tr[metro[tr] == m]
            fallback = np.median(y[tr])
            oof["baseline_metro_median"][te[metro[te] == m]] = np.median(y[same]) if len(same) else fallback
        smallest = int(len(tr) * (inner_splits - 1) / inner_splits) - 1
        candidates = _usable(area_candidates(), smallest)
        inner = {name: _cv_mae(make, X[tr], y[tr], inner_splits) for name, make in candidates.items()}
        best_knn = min((n for n in inner if n.startswith("knn_k")), key=lambda n: inner[n])
        best_all = min(inner, key=lambda n: inner[n])
        picks.append(best_all)
        oof["location_knn"][te] = candidates[best_knn]().fit(X[tr], y[tr]).predict(X[te])
        oof["location_gbm"][te] = candidates["gbm"]().fit(X[tr], y[tr]).predict(X[te])
        oof["selected"][te] = oof["location_knn"][te] if best_all == best_knn else oof["location_gbm"][te]

    report["evaluation"] = "nested 5-fold cross-validation over ZIPs; each ZIP is predicted by a model that never saw it"
    report["overall"] = {name: error_metrics(y, oof[name]) for name in names}
    report["by_metro"] = {}
    for m in sorted(np.unique(metro)):
        mask = metro == m
        report["by_metro"][m] = {name: error_metrics(y[mask], oof[name][mask]) for name in names}
    report["by_city"] = {}
    city = data["supported_city_id"].fillna("").to_numpy()
    for c in sorted(set(city) - {""}):
        mask = city == c
        report["by_city"][c] = {name: error_metrics(y[mask], oof[name][mask]) for name in names}
    report["picks_per_outer_fold"] = picks

    # The served model: choose once more on all ZIPs, then fit on all ZIPs.
    smallest = int(len(data) * (n_outer - 1) / n_outer) - 1
    candidates = _usable(area_candidates(), smallest)
    scores = {name: _cv_mae(make, X, y, n_outer) for name, make in candidates.items()}
    final_name = min(scores, key=lambda n: scores[n])
    model = candidates[final_name]().fit(X, y)
    report["served_model"] = final_name
    report["status"] = "trained"
    return model, report


# ----------------------------------------------------- Seattle sale model

def sale_candidates() -> Dict[str, Callable]:
    """Candidate per-house models. Each takes [latitude, longitude, month index] rows."""
    candidates: Dict[str, Callable] = {}
    for k in SALE_K_VALUES:
        candidates["knn_k%d" % k] = lambda k=k: with_log_target(Pipeline([
            ("xyz", FunctionTransformer(xyz_only)),
            ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance")),
        ]))
    # Gradient boosting uses latitude, longitude and the month as they are.
    candidates["gbm"] = lambda: with_log_target(
        HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                      min_samples_leaf=20, random_state=SEED))
    return candidates


def assign_time_splits(sales: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Split sales by date into train, val, cal, test (oldest to newest).

    A house that sold more than once is kept only in the newest split it
    reaches, so the same house never appears on both sides of a split.
    Returns (table with a 'split' column, number of rows removed).
    """
    df = sales.copy()
    df["record_date"] = pd.to_datetime(df["record_date"])
    df = df.sort_values(["record_date", "property_id"]).reset_index(drop=True)
    n = len(df)
    edges = np.floor(np.cumsum(SPLIT_FRACTIONS) * n).astype(int)
    rank = np.searchsorted(edges, np.arange(n), side="right")
    rank = np.minimum(rank, len(SPLIT_NAMES) - 1)
    df["split_rank"] = rank
    newest = df.groupby("property_id")["split_rank"].transform("max")
    removed = int((df["split_rank"] < newest).sum())
    df = df[df["split_rank"] == newest].copy()
    df["split"] = df["split_rank"].map(dict(enumerate(SPLIT_NAMES)))
    return df.drop(columns=["split_rank"]).reset_index(drop=True), removed


def sale_matrix(df: pd.DataFrame) -> np.ndarray:
    t = month_index_from_dates(pd.to_datetime(df["record_date"]))
    return np.column_stack([df["lat"].to_numpy(dtype=float), df["lon"].to_numpy(dtype=float), t])


def _zip_median_predict(fit: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    city_median = fit["price_usd"].median()
    medians = fit.groupby("zip")["price_usd"].median()
    return target["zip"].map(medians).fillna(city_median).to_numpy(dtype=float)


def train_sale_model(sales: pd.DataFrame, zhvi_history: Optional[pd.DataFrame]) -> Tuple[Optional[Dict], Dict]:
    """Train the per-house sale model with a date-ordered split."""
    report: Dict = {"n_rows": int(len(sales))}
    if len(sales) < MIN_SALE_ROWS:
        report["status"] = "insufficient_data"
        return None, report

    df, removed = assign_time_splits(sales)
    X = sale_matrix(df)
    y = df["price_usd"].to_numpy(dtype=float)
    split = df["split"].to_numpy()
    masks = {name: split == name for name in SPLIT_NAMES}
    report["repeat_sale_rows_removed_to_keep_houses_in_one_split"] = removed
    report["splits"] = {
        name: {
            "n": int(masks[name].sum()),
            "from": df.loc[masks[name], "record_date"].min().strftime("%Y-%m-%d"),
            "to": df.loc[masks[name], "record_date"].max().strftime("%Y-%m-%d"),
        } for name in SPLIT_NAMES
    }
    overlap = set(df.loc[masks["train"] | masks["val"], "property_id"]) & set(df.loc[masks["test"], "property_id"])
    report["houses_in_both_training_and_test"] = len(overlap)

    # Step 1: choose the model on the validation split.
    candidates = sale_candidates()
    val_scores = {}
    for name, make in candidates.items():
        model = make().fit(X[masks["train"]], y[masks["train"]])
        val_scores[name] = error_metrics(y[masks["val"]], model.predict(X[masks["val"]]))
    chosen = min(val_scores, key=lambda n: val_scores[n]["mae"])
    report["validation"] = val_scores
    report["chosen_on_validation"] = chosen

    # Step 2: refit on train + val, measure once on the test split.
    fit_mask = masks["train"] | masks["val"]
    fit_df = df[fit_mask]
    test_df = df[masks["test"]]
    model = candidates[chosen]().fit(X[fit_mask], y[fit_mask])
    test_pred = model.predict(X[masks["test"]])
    y_test = y[masks["test"]]
    test = {
        "baseline_city_median": error_metrics(y_test, np.full(len(y_test), fit_df["price_usd"].median())),
        "baseline_zip_median": error_metrics(y_test, _zip_median_predict(fit_df, test_df)),
        "location_only": error_metrics(y_test, test_pred),
    }
    for name in ("knn_k%d" % SALE_K_VALUES[2], "gbm"):
        if name != chosen:
            other = candidates[name]().fit(X[fit_mask], y[fit_mask])
            test["location_" + name + "_not_chosen"] = error_metrics(y_test, other.predict(X[masks["test"]]))
    if zhvi_history is not None and len(zhvi_history):
        lookup = zhvi_history.set_index(["zip", "month"])["zhvi_usd"]
        keys = list(zip(test_df["zip"], test_df["record_date"].dt.strftime("%Y-%m")))
        zillow = np.array([lookup.get(k, np.nan) for k in keys], dtype=float)
        have = ~np.isnan(zillow)
        test["baseline_zillow_zip_value_same_month"] = error_metrics(y_test[have], zillow[have])
        test["location_only_on_those_rows"] = error_metrics(y_test[have], test_pred[have])
    report["test"] = test
    report["test_median_sale_price"] = round(float(np.median(y_test)), 2)

    # Step 3: prediction range from the calibration split, coverage measured on test.
    interval = None
    n_cal = int(masks["cal"].sum())
    if n_cal >= MIN_CALIBRATION_ROWS:
        cal_pred = model.predict(X[masks["cal"]])
        q = conformal_quantile(np.abs(y[masks["cal"]] - cal_pred) / cal_pred, INTERVAL_TARGET)
        covered = np.abs(y_test - test_pred) / test_pred <= q
        interval = {
            "method": "split conformal on relative error",
            "target_coverage": INTERVAL_TARGET,
            "relative_half_width": round(q, 4),
            "n_calibration": n_cal,
            "n_test": int(len(y_test)),
            "measured_coverage_on_test": round(float(np.mean(covered)), 4),
        }
    report["interval"] = interval or {"status": "unavailable", "reason": "too few calibration rows"}

    # Step 4: the served model is the chosen recipe refit on every row.
    served = candidates[chosen]().fit(X, y)
    report["served_model_note"] = ("The served model uses the chosen recipe refit on all rows, including the test rows. "
                                   "The test numbers above describe the version trained without them.")
    report["status"] = "trained"
    bundle = {
        "model": served,
        "time_index": float(X[:, 2].max()),
        "interval": interval,
        "_eval": {"df": df, "X": X, "y": y, "masks": masks, "chosen": chosen},
    }
    return bundle, report


# ------------------------------------------------------- Seattle photo model

def image_candidates() -> Dict[str, Callable]:
    """Candidate photo models: shrink the 512 numbers with PCA, then a ridge regression."""
    out: Dict[str, Callable] = {}
    for size in IMAGE_PCA_SIZES:
        for alpha in IMAGE_RIDGE_ALPHAS:
            out["pca%d_ridge%g" % (size, alpha)] = lambda size=size, alpha=alpha: Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=size, random_state=SEED)),
                ("ridge", Ridge(alpha=alpha)),
            ])
    return out


def out_of_fold_location(make_model: Callable, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Location-only predictions for training rows, each made by a model that did not see that house."""
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def match_images(df: pd.DataFrame, matches: pd.DataFrame, refs: np.ndarray) -> np.ndarray:
    """Return, for each row of df, the index of its photo embedding or -1.

    A photo is used for one house only (the nearest one), so the same photo
    can never sit in both the training and the test split.
    """
    m = matches.dropna(subset=["image_ref"]).copy()
    m["image_ref"] = m["image_ref"].astype(str)
    m = m.sort_values("image_distance_m").drop_duplicates("image_ref").drop_duplicates("property_id")
    position = {ref: i for i, ref in enumerate(refs.astype(str))}
    ref_of = dict(zip(m["property_id"].astype(str), m["image_ref"]))
    return np.array([position.get(ref_of.get(str(pid), ""), -1) for pid in df["property_id"]], dtype=int)


def train_image_model(sale_bundle: Dict, matches: pd.DataFrame, refs: np.ndarray,
                      embeddings: np.ndarray) -> Tuple[Optional[Dict], Dict]:
    """Train the photo model on the houses that have a matched photo.

    The photo model predicts the log of (true price / location-only price),
    which is the part of the price that location could not explain.
    """
    ev = sale_bundle["_eval"]
    df, X, y, masks, chosen = ev["df"], ev["X"], ev["y"], ev["masks"], ev["chosen"]
    make_location = sale_candidates()[chosen]
    emb_index = match_images(df, matches, refs)
    has_image = emb_index >= 0
    fit_mask = masks["train"] | masks["val"]
    report: Dict = {
        "encoder": ENCODER_NAME,
        "rows_with_photo": {name: int((masks[name] & has_image).sum()) for name in SPLIT_NAMES},
        "rows_total": {name: int(masks[name].sum()) for name in SPLIT_NAMES},
    }
    n_fit = int((fit_mask & has_image).sum())
    n_cal = int((masks["cal"] & has_image).sum())
    n_test = int((masks["test"] & has_image).sum())
    if n_fit < MIN_IMAGE_FIT_ROWS or n_cal < MIN_IMAGE_EVAL_ROWS or n_test < MIN_IMAGE_EVAL_ROWS:
        report["status"] = "insufficient_data"
        report["reason"] = "too few houses with a matched photo"
        return None, report

    groups = df["property_id"].to_numpy()
    fit_idx = np.where(fit_mask)[0]
    oof = out_of_fold_location(make_location, X[fit_idx], y[fit_idx], groups[fit_idx])
    img_in_fit = has_image[fit_idx]
    E_fit = embeddings[emb_index[fit_idx][img_in_fit]]
    y_fit = y[fit_idx][img_in_fit]
    base_fit = oof[img_in_fit]
    resid_fit = np.log(y_fit) - np.log(base_fit)
    g_fit = groups[fit_idx][img_in_fit]

    # Choose PCA size and ridge strength with grouped cross-validation on the fit rows.
    scores = {}
    for name, make in image_candidates().items():
        errors = []
        for tr, te in GroupKFold(n_splits=5).split(E_fit, resid_fit, g_fit):
            if make().named_steps["pca"].n_components >= min(len(tr), E_fit.shape[1]):
                errors = []
                break
            r_hat = make().fit(E_fit[tr], resid_fit[tr]).predict(E_fit[te])
            errors.append(np.mean(np.abs(base_fit[te] * np.exp(r_hat) - y_fit[te])))
        if errors:
            scores[name] = float(np.mean(errors))
    if not scores:
        report["status"] = "insufficient_data"
        report["reason"] = "too few rows for the smallest photo model"
        return None, report
    chosen_image = min(scores, key=lambda n: scores[n])
    report["chosen_photo_model"] = chosen_image
    report["cross_validation_mae_on_fit_rows"] = {
        "location_only": round(float(np.mean(np.abs(base_fit - y_fit))), 2),
        "location_plus_photo": round(scores[chosen_image], 2),
    }
    residual_model = image_candidates()[chosen_image]().fit(E_fit, resid_fit)
    location_model = make_location().fit(X[fit_mask], y[fit_mask])

    def both_predictions(mask):
        rows = np.where(mask & has_image)[0]
        base = location_model.predict(X[rows])
        photo = base * np.exp(residual_model.predict(embeddings[emb_index[rows]]))
        return y[rows], base, photo

    y_cal, base_cal, photo_cal = both_predictions(masks["cal"])
    y_test, base_test, photo_test = both_predictions(masks["test"])
    cal = {"location_only": error_metrics(y_cal, base_cal), "location_plus_photo": error_metrics(y_cal, photo_cal)}
    test = {"location_only": error_metrics(y_test, base_test), "location_plus_photo": error_metrics(y_test, photo_test)}
    report["calibration_rows_with_photo"] = cal
    report["test_rows_with_photo"] = test
    cal_gain = 1.0 - cal["location_plus_photo"]["mae"] / cal["location_only"]["mae"]
    serve = cal_gain >= SERVE_MIN_IMPROVEMENT
    report["calibration_mae_lowered_by_share"] = round(float(cal_gain), 4)
    report["serve_photo_model"] = bool(serve)
    report["serve_rule"] = ("the photo model is served only if it lowered the error on the calibration split "
                            "by at least %.0f%%" % (100 * SERVE_MIN_IMPROVEMENT))
    report["test_comparison"] = compare_errors(y_test, base_test, photo_test)

    interval = None
    if n_cal >= MIN_CALIBRATION_ROWS:
        q = conformal_quantile(np.abs(y_cal - photo_cal) / photo_cal, INTERVAL_TARGET)
        interval = {
            "method": "split conformal on relative error",
            "target_coverage": INTERVAL_TARGET,
            "relative_half_width": round(q, 4),
            "n_calibration": n_cal,
            "n_test": n_test,
            "measured_coverage_on_test": round(float(np.mean(np.abs(y_test - photo_test) / photo_test <= q)), 4),
        }
    report["interval"] = interval or {"status": "unavailable", "reason": "too few calibration rows with a photo"}

    # Served version: residual model refit on every house with a photo.
    oof_all = out_of_fold_location(make_location, X, y, groups)
    served = image_candidates()[chosen_image]().fit(
        embeddings[emb_index[has_image]], np.log(y[has_image]) - np.log(oof_all[has_image]))
    report["status"] = "trained"
    return {"residual_model": served, "interval": interval, "serve": bool(serve)}, report


# ------------------------------------------------------ area photo experiment

MIN_AREA_PHOTO_POINTS = 150
MIN_AREA_PHOTO_ZIPS = 15


def area_photo_experiment(zips: pd.DataFrame, points: pd.DataFrame, refs: np.ndarray, embeddings: np.ndarray,
                          recipes: Dict[str, str]) -> Dict:
    """Do street photos say anything about an area's typical value beyond its location?

    Rows are photos taken near random spots in the supported cities. The
    label of a photo is the Zillow value of the ZIP it is in. Every ZIP is
    held out in turn (5 folds, grouped by ZIP). The location-only number is
    the area model's prediction when it has never seen that ZIP. The photo
    model predicts what that location-only number missed.

    This is an experiment only. Its model is never served, because inside
    the app the area model already knows the value of the house's own ZIP.
    """
    report: Dict = {
        "description": ("Photos near random spots in each supported city ZIP, labeled with the ZIP's Zillow value. "
                        "5 folds grouped by ZIP: every number is for ZIPs the models never saw. "
                        "This experiment is not served by the app."),
    }
    position = {str(ref): i for i, ref in enumerate(refs)}
    pts = points.dropna(subset=["image_ref"]).copy()
    pts["image_ref"] = pts["image_ref"].astype(str)
    pts = pts[pts["image_ref"].isin(position)].drop_duplicates("image_ref").reset_index(drop=True)
    report["n_photos"] = int(len(pts))
    report["n_zips"] = int(pts["zip"].nunique())
    if len(pts) < MIN_AREA_PHOTO_POINTS or pts["zip"].nunique() < MIN_AREA_PHOTO_ZIPS:
        report["status"] = "insufficient_data"
        report["reason"] = "too few photos or ZIPs"
        return report

    candidates = area_candidates()
    report["targets"] = {}
    for title, column, recipe_key in (("Typical home value", "zhvi_usd", "sale"), ("Monthly rent index", "zori_usd_month", "rent")):
        table = zips.dropna(subset=[column, "lat", "lon"]).reset_index(drop=True)
        value_of = dict(zip(table["zip"], table[column]))
        p = pts[pts["zip"].isin(value_of)].reset_index(drop=True)
        if p["zip"].nunique() < MIN_AREA_PHOTO_ZIPS:
            continue
        y = p["zip"].map(value_of).to_numpy(dtype=float)
        groups = p["zip"].to_numpy()
        X = p[["lat", "lon"]].to_numpy(dtype=float)
        E = embeddings[[position[r] for r in p["image_ref"]]]
        make_area = candidates[recipes[recipe_key]]
        table_X = table[["lat", "lon"]].to_numpy(dtype=float)
        table_y = table[column].to_numpy(dtype=float)
        table_zip = table["zip"].to_numpy()

        def base_without(hidden: set, rows: np.ndarray) -> np.ndarray:
            keep = ~np.isin(table_zip, list(hidden))
            return make_area().fit(table_X[keep], table_y[keep]).predict(rows)

        base = np.zeros(len(p))
        photo = np.zeros(len(p))
        for tr, te in GroupKFold(n_splits=5).split(E, y, groups):
            test_zips = set(groups[te])
            base[te] = base_without(test_zips, X[te])
            # Location-only numbers for the training photos, each with its own ZIP hidden too.
            base_tr = np.zeros(len(tr))
            for z in np.unique(groups[tr]):
                rows = groups[tr] == z
                base_tr[rows] = base_without(test_zips | {z}, X[tr][rows])
            resid_tr = np.log(y[tr]) - np.log(base_tr)
            scores = {}
            for name, make in image_candidates().items():
                errors = []
                for a, b in GroupKFold(n_splits=4).split(E[tr], resid_tr, groups[tr]):
                    if make().named_steps["pca"].n_components >= len(a):
                        errors = []
                        break
                    r_hat = make().fit(E[tr][a], resid_tr[a]).predict(E[tr][b])
                    errors.append(np.mean(np.abs(base_tr[b] * np.exp(r_hat) - y[tr][b])))
                if errors:
                    scores[name] = float(np.mean(errors))
            best = min(scores, key=lambda n: scores[n])
            photo[te] = base[te] * np.exp(image_candidates()[best]().fit(E[tr], resid_tr).predict(E[te]))

        block = {
            "overall": {"location_only": error_metrics(y, base), "location_plus_photo": error_metrics(y, photo)},
            "by_city": {},
            "n_zips": int(p["zip"].nunique()),
        }
        for city in sorted(p["city_id"].dropna().unique()):
            mask = (p["city_id"] == city).to_numpy()
            block["by_city"][city] = {"location_only": error_metrics(y[mask], base[mask]),
                                      "location_plus_photo": error_metrics(y[mask], photo[mask])}
        block["comparison"] = compare_errors(y, base, photo, groups)
        report["targets"][title] = block
    report["status"] = "trained" if report["targets"] else "insufficient_data"
    return report


# -------------------------------------------------------------- main flow

def load_inputs(fixtures: bool) -> Dict:
    if fixtures:
        base = FIXTURES_DIR
        paths = {
            "zip": base / "synthetic_zip_values.csv",
            "sales": base / "synthetic_seattle_sales.csv",
            "ratio": base / "synthetic_metro_rent_ratio.json",
            "history": base / "does_not_exist.csv",
            "matches": base / "synthetic_image_matches.csv",
            "embeddings": base / "synthetic_image_embeddings.npz",
        }
    else:
        base = PROCESSED_DIR
        paths = {
            "zip": base / "zip_values.csv",
            "sales": base / "seattle_sales.csv",
            "ratio": base / "metro_rent_ratio.json",
            "history": base / "zhvi_history.csv",
            "matches": base / "image_matches.csv",
            "embeddings": base / "image_embeddings.npz",
        }
    paths["area_points"] = base / "area_image_points.csv"
    out: Dict = {"paths": paths}
    out["zip"] = pd.read_csv(paths["zip"], dtype={"zip": str}, keep_default_na=False,
                             na_values={"zhvi_usd": [""], "zori_usd_month": [""], "lat": [""], "lon": [""]})
    out["sales"] = pd.read_csv(paths["sales"], dtype={"zip": str, "property_id": str}) if paths["sales"].exists() else None
    out["history"] = pd.read_csv(paths["history"], dtype={"zip": str}) if paths["history"].exists() else None
    with open(paths["ratio"], "r", encoding="utf-8") as f:
        out["ratio"] = json.load(f)
    out["matches"] = None
    out["embeddings"] = None
    if paths["matches"].exists() and paths["embeddings"].exists():
        out["matches"] = pd.read_csv(paths["matches"], dtype={"property_id": str, "image_ref": str})
        npz = np.load(paths["embeddings"], allow_pickle=False)
        out["embeddings"] = (npz["image_ref"], npz["embedding"].astype(np.float64))
    # How many searched houses and spots had a usable photo (written by pipeline.fetch_images).
    out["coverage"] = None
    coverage_path = base / "image_coverage.json"
    if not fixtures and coverage_path.exists():
        with open(coverage_path, "r", encoding="utf-8") as f:
            out["coverage"] = json.load(f)
    out["area_points"] = None
    if paths["area_points"].exists() and paths["embeddings"].exists():
        out["area_points"] = pd.read_csv(paths["area_points"], dtype={"zip": str, "image_ref": str})
        if out["embeddings"] is None:
            npz = np.load(paths["embeddings"], allow_pickle=False)
            out["embeddings"] = (npz["image_ref"], npz["embedding"].astype(np.float64))
    return out


def train_all(fixtures: bool, out_dir: Optional[Path] = None) -> Dict:
    """Train everything and write bundle.joblib and manifest.json to out_dir."""
    inputs = load_inputs(fixtures)
    out_dir = out_dir or (DEMO_MODEL_DIR if fixtures else REAL_MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = inputs["zip"]
    metrics: Dict = {}

    print("Training area sale model")
    area_sale, metrics["area_sale"] = train_area_model(zips, "zhvi_usd")
    print("Training area rent model")
    area_rent, metrics["area_rent"] = train_area_model(zips, "zori_usd_month")

    sale_bundle = None
    metrics["seattle_sale"] = {"status": "no_data"}
    if inputs["sales"] is not None:
        print("Training Seattle per-house sale model")
        sale_bundle, metrics["seattle_sale"] = train_sale_model(inputs["sales"], inputs["history"])

    image_bundle = None
    metrics["seattle_photo"] = {"status": "not_trained", "reason": "no matched photos have been downloaded"}
    if sale_bundle is not None and inputs["embeddings"] is not None and inputs["matches"] is not None:
        print("Training Seattle photo model")
        refs, emb = inputs["embeddings"]
        image_bundle, metrics["seattle_photo"] = train_image_model(sale_bundle, inputs["matches"], refs, emb)

    if inputs["area_points"] is not None and area_sale is not None and area_rent is not None:
        print("Running the area photo experiment")
        refs, emb = inputs["embeddings"]
        recipes = {"sale": metrics["area_sale"]["served_model"], "rent": metrics["area_rent"]["served_model"]}
        metrics["area_photo"] = area_photo_experiment(zips, inputs["area_points"], refs, emb, recipes)

    trained_at = datetime.now(timezone.utc)
    kind = "demo" if fixtures else "real"
    version = "homelens-%s-%s-%s" % (kind, trained_at.strftime("%Y%m%d"),
                                     file_hash([inputs["paths"][k] for k in ("zip", "sales", "embeddings")]))
    sales = inputs["sales"]
    manifest = {
        "model_version": version,
        "kind": kind,
        "synthetic_training_data": bool(fixtures),
        "feature_version": FEATURE_VERSION,
        "trained_at_utc": trained_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_period": {
            "area_sale_month": str(zips["zhvi_month"].iloc[0]),
            "area_rent_month": str(zips["zori_month"].iloc[0]),
            "seattle_sales_from": None if sales is None else str(sales["record_date"].min()),
            "seattle_sales_to": None if sales is None else str(sales["record_date"].max()),
        },
        "metro_rent_ratio": inputs["ratio"],
        "photo_coverage": inputs["coverage"],
        "interval_target": INTERVAL_TARGET,
        "metrics": metrics,
    }
    if sale_bundle is not None:
        sale_bundle = {k: v for k, v in sale_bundle.items() if k != "_eval"}
    bundle = {"area_sale": area_sale, "area_rent": area_rent, "seattle_sale": sale_bundle,
              "seattle_photo": image_bundle, "manifest": manifest}
    joblib.dump(bundle, out_dir / "bundle.joblib")
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if not fixtures:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    print("Saved %s to %s" % (version, out_dir))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Train HomeLens models.")
    parser.add_argument("--fixtures", action="store_true", help="train demo models on the synthetic fixtures")
    args = parser.parse_args()
    train_all(fixtures=args.fixtures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
