"""The San Francisco preview uses a confirmed parcel and falls back when facts are missing."""
import joblib
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.tests.conftest import Constant, estimate_body
from core.paths import FIXTURES_DIR
from pipeline.san_francisco import FEATURE_COLUMNS


def test_san_francisco_preview_and_fallback(tmp_path, geocoder):
    joblib.dump({
        "manifest": {"synthetic_training_data": False, "model_version": "test",
                     "data_period": {"area_sale_month": "2026-01", "area_rent_month": "2026-01"}},
        "area_sale": Constant(1_250_000), "area_rent": Constant(4_000),
    }, tmp_path / "bundle.joblib")
    coords = {"lat": 37.76, "lng": -122.484}
    addresses = ["100 Test Ave, San Francisco, CA 94122", "100 Other Ave, San Francisco, CA 94122"]
    parcels = pd.DataFrame([
        {"property_id": key, "address": address, "zip": "94122", "lat": coords["lat"], "lon": coords["lng"]}
        for key, address in zip(("sf-0001001", "sf-0001002"), addresses)
    ])
    values = [1_500, 1930, 3, 2, 6, 1, 2_500]
    features = pd.DataFrame([values, [*values[:4], float("nan"), *values[5:]]],
                            columns=FEATURE_COLUMNS, index=parcels["property_id"])
    joblib.dump({
        "house": {"model": Constant(1_750_000), "feature_cols": FEATURE_COLUMNS, "time_index": 1, "interval": None},
        "features": features, "parcels": parcels, "photo": None, "price_source": "Test sales",
        "sales_from": "2024-01-01", "sales_to": "2025-12-31",
    }, tmp_path / "san_francisco.joblib")
    settings = Settings(model_dir=tmp_path, zip_table_path=FIXTURES_DIR / "synthetic_zip_values.csv",
                        parcels_path=None, house_features_path=None)
    with TestClient(create_app(settings=settings, geocoder=geocoder)) as app:
        body = estimate_body(coords, address=addresses[0], property_id="sf-0001001")
        assert app.post("/estimate", json=body).json()["estimates"][0]["basis"] == "area_level_model"

    settings.san_francisco_preview = True
    with TestClient(create_app(settings=settings, geocoder=geocoder)) as app:
        found = app.post("/resolve", json={"address": "100 Test Avenue, San Francisco, CA 94122"}).json()
        assert found["candidates"][0]["property_id"] == "sf-0001001"
        assert not geocoder.calls
        result = app.post("/estimate", json=body).json()
        assert result["prediction_mode"] == "location_plus_house_record"
        assert result["estimates"][0]["amount_usd"] == 1_750_000
        assert len(result["estimates"]) == 4
        assert result["image"]["status"] == "no_photo_model"

        wrong_street = app.post("/estimate", json={**body, "address": addresses[1]}).json()
        incomplete = app.post("/estimate", json={**body, "address": addresses[1], "property_id": "sf-0001002"}).json()
        for fallback in (wrong_street, incomplete):
            assert fallback["prediction_mode"] == "location_only"
            assert fallback["estimates"][0]["basis"] == "area_level_model"
            assert fallback["estimates"][0]["amount_usd"] == 1_250_000
