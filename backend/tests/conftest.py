"""Shared test setup: a demo backend built on the synthetic fixtures, with no network access."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.cities import Locator
from backend.app.config import Settings
from backend.app.estimator import Estimator
from backend.app.geocode import GeocodeError, GeocodeResult
from backend.app.main import create_app
from core import furniture
from core.geo import load_cities
from core.imagery import NoImageryProvider
from core.paths import FIXTURES_DIR
from pipeline import train

EXTENSION_ORIGIN = "chrome-extension://" + "a" * 32
SEATTLE_HOUSE = {"lat": 47.6730, "lng": -122.3430}
PORTLAND_HOUSE = {"lat": 45.5150, "lng": -122.6000}
BELLEVUE = {"lat": 47.6180, "lng": -122.2050}        # in the Seattle metro, outside the city
NEW_YORK = {"lat": 40.7128, "lng": -74.0060}


class FakeGeocoder:
    """Stands in for the Census geocoder so tests never touch the network."""

    def __init__(self):
        self.fail = False
        self.calls = []

    def lookup(self, address):
        self.calls.append(address)
        if self.fail:
            raise GeocodeError("address lookup timed out")
        if "nowhere" in address.lower():
            return None
        return GeocodeResult(address="12 Test Ave, Seattle, WA", lat=47.6731, lon=-122.3431,
                             zip="98103", city="SEATTLE", state="WA", source="Test geocoder")


class Constant:
    """A stand-in model that always predicts the same number."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


@pytest.fixture(scope="session")
def demo_model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo_models")
    train.train_all(fixtures=True, out_dir=out)
    return out


@pytest.fixture(scope="session")
def cities():
    return load_cities()


@pytest.fixture(scope="session")
def zip_table():
    return pd.read_csv(FIXTURES_DIR / "synthetic_zip_values.csv", dtype={"zip": str}, keep_default_na=False)


@pytest.fixture()
def demo_settings(demo_model_dir):
    return Settings(demo=True, model_dir=demo_model_dir, zip_table_path=FIXTURES_DIR / "synthetic_zip_values.csv",
                    parcels_path=None, requests_per_minute=1000)


@pytest.fixture()
def geocoder():
    return FakeGeocoder()


@pytest.fixture()
def package():
    return furniture.load_package()


@pytest.fixture()
def demo_estimator(demo_settings, package):
    return Estimator.from_dir(demo_settings.model_dir, NoImageryProvider(), package, demo=True)


@pytest.fixture()
def client(demo_settings, demo_estimator, geocoder):
    app = create_app(settings=demo_settings, estimator=demo_estimator, geocoder=geocoder)
    return TestClient(app)


def estimate_body(point, **changes):
    body = {"lat": point["lat"], "lng": point["lng"], "property_type": "detached_single_family",
            "client_request_id": "req-1"}
    body.update(changes)
    return body
