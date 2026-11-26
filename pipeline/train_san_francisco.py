"""Train a San Francisco candidate for review without replacing the live bundle."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.paths import PROCESSED_DIR, RAW_DIR, REAL_MODEL_DIR, REPORTS_DIR
from pipeline.build_dataset import read_zillow_zip_file
from pipeline.embed_images import embeddings_path
from pipeline.san_francisco import FEATURE_COLUMNS, SALES_SOURCE
from pipeline.train import error_metrics, sale_candidates, train_photo_models, train_sale_model


def _zillow_history(zips: pd.Series, raw: Path = RAW_DIR) -> pd.DataFrame:
    path = raw / "zillow" / "Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
    values, months = read_zillow_zip_file(path)
    history = values.loc[values["zip"].isin(zips), ["zip", *months]].melt(
        id_vars=["zip"], var_name="month", value_name="zhvi_usd")
    history["month"] = history["month"].str[:7]
    return history.dropna(subset=["zhvi_usd"])


def _comparison(actual, prediction) -> dict:
    return {**error_metrics(actual, prediction),
            "median_percent_error": round(100 * float(np.median(np.abs(prediction - actual) / actual)), 2)}


def train_candidate(processed: Path, model_dir: Path) -> dict:
    csv_types = {"property_id": str, "zip": str}
    sales = pd.read_csv(processed / "san_francisco_sales.csv", dtype=csv_types)
    if sales.empty or not sales["synthetic"].eq(False).all():
        raise ValueError("San Francisco training needs real sales")
    houses = pd.read_csv(processed / "san_francisco_house_features.csv", dtype=csv_types).set_index("property_id")
    parcels = pd.read_csv(processed / "san_francisco_parcels.csv", dtype=csv_types)
    history = _zillow_history(sales["zip"])
    area = joblib.load(model_dir / "bundle.joblib")["area_sale"]
    candidate, report = train_sale_model(sales, history, FEATURE_COLUMNS)
    if candidate is None:
        raise ValueError("Too few completed sales to train and test a San Francisco model")

    evaluation = candidate.pop("_eval")
    # Keep the actual evaluated model for review, with test sales out of the fit.
    candidate["model"] = evaluation["model"]
    report["served_model_note"] = "The preview uses the model fitted on the training and validation sales."
    rows = evaluation["df"][evaluation["masks"]["test"]]
    y = evaluation["y"][evaluation["masks"]["test"]]
    predictions = dict(evaluation["test_predictions"])
    if report["location_only_model_on_same_rows"] == "gbm":
        predictions["location_and_date"] = predictions.pop("location_only_same_rows")
    else:
        # The selected nearest-neighbor comparison ignores time. Also measure the date-aware recipe.
        masks = evaluation["masks"]
        fit = masks["train"] | masks["val"]
        location_and_date = evaluation["X"][:, :3]
        dated_model = sale_candidates()["gbm"]().fit(location_and_date[fit], evaluation["y"][fit])
        predictions["location_and_date"] = dated_model.predict(location_and_date[masks["test"]])
    predictions["current_area_estimate"] = area.predict(rows[["lat", "lon"]].to_numpy(dtype=float))
    report["area_comparison_note"] = (
        "The current area estimate uses the current deployed ZIP value model. "
        "The separate same-month ZIP baseline uses historical Zillow values.")
    report["held_out_comparison"] = {name: _comparison(y, pred) for name, pred in predictions.items()}
    dates_verified = rows["assessor_date_confirmed"].to_numpy(dtype=bool)
    report["held_out_assessor_date_confirmed"] = (
        {name: _comparison(y[dates_verified], pred[dates_verified]) for name, pred in predictions.items()}
        if dates_verified.any() else {})
    report["current_house_records"] = len(houses)
    report["price_source"] = SALES_SOURCE
    report["assessor_date_confirmed"] = int(sales["assessor_date_confirmed"].sum())

    photo = None
    matches_path = processed / "san_francisco_photos" / "image_matches.csv"
    photo_report = {"status": "not_trained", "reason": "No San Francisco photo matches and embeddings were supplied."}
    if matches_path.is_file():
        from core.embed import ENCODERS

        encoders = {}
        for encoder in ENCODERS:
            path = embeddings_path(encoder, matches_path.parent)
            if path.is_file():
                with np.load(path, allow_pickle=False) as values:
                    encoders[encoder] = (values["image_ref"], values["embedding"].astype(np.float64))
        if encoders:
            matches = pd.read_csv(matches_path, dtype={"property_id": str, "image_ref": str})
            photo, photo_report = train_photo_models({**candidate, "_eval": evaluation}, matches, encoders)
            if photo is not None:
                photo["base"] = "location_plus_house_record"
    report["photo"] = photo_report

    artifact = {
        "house": candidate, "features": houses, "parcels": parcels, "photo": photo,
        "photo_report": photo_report, "price_source": SALES_SOURCE,
        "sales_from": report["splits"]["train"]["from"], "sales_to": report["splits"]["val"]["to"],
    }
    path = model_dir / "san_francisco.joblib"
    tmp = path.with_suffix(".part")
    joblib.dump(artifact, tmp)
    tmp.replace(path)
    return report


def main() -> int:
    report = train_candidate(PROCESSED_DIR, REAL_MODEL_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "san_francisco_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["held_out_comparison"], indent=2))
    print("Evaluation saved to %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
