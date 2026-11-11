"""Training: leakage guards, metrics, and that saved models predict the same thing twice."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from core.features import to_xyz_km
from core.geo import haversine_m
from pipeline import train


def synthetic_sales():
    return pd.read_csv(train.FIXTURES_DIR / "synthetic_seattle_sales.csv", dtype={"zip": str, "property_id": str})


def test_error_metrics_small_example():
    m = train.error_metrics([100.0, 200.0], [110.0, 170.0])
    assert m["n"] == 2
    assert m["mae"] == 20.0                      # (10 + 30) / 2
    assert m["rmse"] == pytest.approx(22.36, abs=0.01)   # sqrt((100 + 900) / 2)


def test_conformal_quantile_is_conservative():
    scores = np.arange(1, 101) / 100.0           # 0.01 ... 1.00
    q = train.conformal_quantile(scores, 0.80)
    assert q == pytest.approx(0.81)              # ceil(101 * 0.8) = 81st smallest score
    assert np.mean(scores <= q) >= 0.80


def test_xyz_distance_matches_ground_distance():
    a = (47.6561, -122.3560)
    b = (47.6565, -122.3568)
    xyz = to_xyz_km(np.array([a, b]))
    straight_m = float(np.linalg.norm(xyz[0] - xyz[1]) * 1000)
    assert straight_m == pytest.approx(float(haversine_m(a[0], a[1], b[0], b[1])), rel=1e-3)


def test_time_split_is_in_date_order_and_keeps_each_house_in_one_split():
    df, removed = train.assign_time_splits(synthetic_sales())
    assert removed > 0                           # the fixtures contain repeat sales on purpose
    newest = {name: df.loc[df["split"] == name, "record_date"].max() for name in train.SPLIT_NAMES}
    oldest = {name: df.loc[df["split"] == name, "record_date"].min() for name in train.SPLIT_NAMES}
    assert newest["train"] <= oldest["val"] <= newest["val"] <= oldest["cal"] <= newest["cal"] <= oldest["test"]
    splits_per_house = df.groupby("property_id")["split"].nunique()
    assert splits_per_house.max() == 1
    train_houses = set(df.loc[df["split"].isin(["train", "val", "cal"]), "property_id"])
    test_houses = set(df.loc[df["split"] == "test", "property_id"])
    assert train_houses.isdisjoint(test_houses)


def test_image_matching_uses_each_photo_for_one_house_only():
    df = pd.DataFrame({"property_id": ["A", "B", "C"]})
    matches = pd.DataFrame({
        "property_id": ["A", "B", "C"],
        "image_ref": ["img1", "img1", "img2"],        # A and B were matched to the same photo
        "image_distance_m": [20.0, 8.0, 15.0],
    })
    index = train.match_images(df, matches, np.array(["img1", "img2"]))
    assert list(index) == [-1, 0, 1]                  # the nearer house (B) keeps img1, A gets none


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo_models")
    manifest = train.train_all(fixtures=True, out_dir=out)
    return out, manifest


def test_fixture_training_is_labeled_synthetic(demo_run):
    out, manifest = demo_run
    assert manifest["synthetic_training_data"] is True
    assert manifest["kind"] == "demo"
    assert manifest["model_version"].startswith("homelens-demo-")
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["model_version"] == manifest["model_version"]


def test_no_house_is_in_training_and_test(demo_run):
    _, manifest = demo_run
    assert manifest["metrics"]["seattle_sale"]["houses_in_both_training_and_test"] == 0


def test_saved_models_load_and_predict_consistently(demo_run):
    out, _ = demo_run
    first = joblib.load(out / "bundle.joblib")
    second = joblib.load(out / "bundle.joblib")
    point = np.array([[47.66, -122.34]])
    a = first["area_sale"].predict(point)[0]
    assert a == second["area_sale"].predict(point)[0]
    assert a == first["area_sale"].predict(point)[0]          # same input, same output
    assert 50000 < a < 30000000
    sale = first["seattle_sale"]
    row = np.array([[47.66, -122.34, sale["time_index"]]])
    assert sale["model"].predict(row)[0] == second["seattle_sale"]["model"].predict(row)[0]
    rent = first["area_rent"].predict(point)[0]
    assert 300 < rent < 50000


def test_too_little_data_reports_insufficient_instead_of_training():
    tiny = pd.DataFrame({"lat": [47.6, 47.7], "lon": [-122.3, -122.4], "zhvi_usd": [1.0e6, 1.1e6],
                         "metro_id": ["seattle", "seattle"], "supported_city_id": ["seattle", "seattle"]})
    model, report = train.train_area_model(tiny, "zhvi_usd")
    assert model is None
    assert report["status"] == "insufficient_data"
    bundle, report = train.train_sale_model(synthetic_sales().head(50), None)
    assert bundle is None
    assert report["status"] == "insufficient_data"


def test_photo_model_trains_on_fixtures_and_never_shares_a_photo_across_splits(demo_run):
    _, manifest = demo_run
    photo = manifest["metrics"]["seattle_photo"]
    assert photo["status"] == "trained"
    assert photo["rows_with_photo"]["test"] >= 30
    assert set(photo["test_rows_with_photo"]) == {"location_only", "location_plus_photo"}
    assert isinstance(photo["serve_photo_model"], bool)
    assert photo["interval"].get("status") == "unavailable"       # only 40 calibration rows: no range is claimed


def test_area_photo_experiment_groups_by_zip_and_reports_both_models():
    zips = pd.read_csv(train.FIXTURES_DIR / "synthetic_zip_values.csv", dtype={"zip": str}, keep_default_na=False)
    city_zips = zips[zips["supported_city_id"] != ""].reset_index(drop=True)
    rng = np.random.default_rng(0)
    rows, embeddings = [], []
    for z in city_zips.itertuples():
        for i in range(10):
            rows.append({"zip": z.zip, "city_id": z.supported_city_id, "image_ref": "%s-%d" % (z.zip, i),
                         "lat": z.lat + rng.normal(0, 0.002), "lon": z.lon + rng.normal(0, 0.002)})
            signal = np.log(z.zhvi_usd)
            embeddings.append(np.concatenate([[signal], rng.normal(0, 1, 63)]))
    points = pd.DataFrame(rows)
    refs = points["image_ref"].to_numpy()
    report = train.area_photo_experiment(zips, points, refs, np.array(embeddings, dtype=np.float32),
                                         {"sale": "knn_k3", "rent": "knn_k3"})
    assert report["status"] == "trained"
    assert report["n_photos"] == len(points) and report["n_zips"] == len(city_zips)
    block = report["targets"]["Typical home value"]
    assert block["overall"]["location_only"]["n"] == len(points)
    assert set(block["by_city"]) == {"portland", "san_francisco", "seattle"}
    assert "not served" in report["description"]
    tiny = train.area_photo_experiment(zips, points.head(20), refs, np.array(embeddings, dtype=np.float32),
                                       {"sale": "knn_k3", "rent": "knn_k3"})
    assert tiny["status"] == "insufficient_data"
