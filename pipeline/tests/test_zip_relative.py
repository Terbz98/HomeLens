"""ZIP-relative model: target, ZIP value lookup, test-city records, and that it transfers to an unseen city.

The data here is SYNTHETIC. Prices are built as (ZIP value) x (a factor that depends on how the house
compares with the typical house of its ZIP), which is the assumption the model makes. The test city's
houses are bigger than Seattle's, as in the real suburbs, so a model on the raw house record overprices
them, and a model that learned Seattle's latitude and longitude learns nothing there.
"""
import numpy as np
import pandas as pd
import pytest

from core import house_features
from pipeline import king_county_cities as kcc
from pipeline import train_zip_relative as zr

MONTHS = pd.period_range("2023-12", "2026-10", freq="M").strftime("%Y-%m").tolist()


def _zips(prefix: str, n: int, lat0: float, lon0: float, low: float, high: float, rng):
    return pd.DataFrame({
        "zip": ["%s%02d" % (prefix, i) for i in range(n)],
        "base": np.exp(rng.uniform(np.log(low), np.log(high), n)),
        "lat": lat0 + rng.uniform(-0.05, 0.05, n),
        "lon": lon0 + rng.uniform(-0.05, 0.05, n),
    })


def _houses(zips: pd.DataFrame, n: int, city: str, id_prefix: str, rng, size: float = 1.0):
    z = zips.sample(n, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    sqft = (size * rng.uniform(900, 4000, n)).round()
    grade = rng.integers(5, 12, n).astype(float)
    feats = pd.DataFrame({
        "living_sqft": sqft, "grade": grade, "year_built": rng.integers(1910, 2015, n).astype(float),
        "year_renovated": 0.0, "condition": rng.integers(2, 6, n).astype(float),
        "bedrooms": np.clip((sqft / 700).round(), 1, 6), "bathrooms": np.clip((sqft / 1000).round(), 1, 4),
        "stories": rng.integers(1, 3, n).astype(float), "basement_sqft": 0.0, "buildings_on_parcel": 1.0,
        "lot_sqft": rng.uniform(3000, 9000, n).round(), "view_score": 0.0, "waterfront": 0,
    })
    feats.index = ["%s%06d" % (id_prefix, i) for i in range(n)]
    feats.index.name = "property_id"
    house_factor = 0.45 * np.log(sqft / (size * 2450)) + 0.08 * (grade - 7)
    month_i = rng.integers(0, len(MONTHS), n)
    trend = 1 + 0.004 * month_i
    price = z["base"] * trend * np.exp(house_factor + rng.normal(0, 0.06, n))
    dates = [MONTHS[i] + "-15" for i in month_i]
    sales = pd.DataFrame({
        "property_id": feats.index, "zip": z["zip"], "city_id": city.lower(), "city": city,
        "lat": z["lat"] + rng.normal(0, 0.005, n), "lon": z["lon"] + rng.normal(0, 0.005, n),
        "record_date": dates, "price_usd": price.round(),
    })
    history = pd.DataFrame([{"zip": r.zip, "month": m, "zhvi_usd": r.base * (1 + 0.004 * i)}
                            for r in zips.itertuples() for i, m in enumerate(MONTHS)])
    parcels = sales[["property_id", "zip"]]
    return sales, feats[house_features.FEATURE_COLUMNS], parcels, history


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(1)
    sea_z = _zips("981", 20, 47.62, -122.33, 600_000, 1_400_000, rng)
    # A test city far from Seattle, mostly priced outside Seattle's range: Seattle's lat/lon learn nothing here.
    far_z = _zips("980", 10, 47.38, -122.23, 380_000, 700_000, rng)
    sea_sales, sea_feats, sea_parcels, sea_hist = _houses(sea_z, 1500, "Seattle", "S", rng)
    far_sales, far_feats, far_parcels, far_hist = _houses(far_z, 700, "Kent", "K", rng, size=1.4)
    return {"sea": (sea_sales, sea_feats, sea_parcels), "far": (far_sales, far_feats, far_parcels),
            "history": pd.concat([sea_hist, far_hist], ignore_index=True)}


def test_relative_target_is_log_ratio():
    assert zr.relative_target([110.0, 100.0], [100.0, 100.0]) == pytest.approx([np.log(1.1), 0.0])


def test_attach_zip_value_uses_the_sale_month_and_drops_unknown_zips():
    sales = pd.DataFrame({"zip": ["98101", "98101", "99999"], "record_date": ["2025-01-03", "2025-02-27", "2025-01-03"],
                          "price_usd": [1.0, 2.0, 3.0]})
    history = pd.DataFrame({"zip": ["98101", "98101"], "month": ["2025-01", "2025-02"], "zhvi_usd": [500.0, 510.0]})
    out, counts = zr.attach_zip_value(sales, history)
    assert out["zhvi_usd"].tolist() == [500.0, 510.0]
    assert counts == {"sales": 3, "with_a_zillow_zip_value_in_the_sale_month": 2}


def test_model_inputs_never_include_location_or_date():
    for cols in zr.VARIANTS.values():
        assert not {"lat", "lon", "record_date", "zip"} & set(cols)


def test_typical_house_is_the_zip_median_over_parcels_and_needs_enough_parcels():
    feats = pd.DataFrame({c: 1.0 for c in house_features.FEATURE_COLUMNS},
                         index=["p%d" % i for i in range(31)] + ["q0"])
    feats["living_sqft"] = [1000.0] * 15 + [2000.0] * 16 + [5000.0]
    parcels = pd.DataFrame({"property_id": feats.index, "zip": ["98101"] * 31 + ["98102"]})
    typical = zr.zip_typical_houses(feats, parcels)
    assert typical.index.tolist() == ["98101"]           # 98102 has one parcel only
    assert typical.loc["98101", "living_sqft"] == 2000.0
    sales = feats.loc[["p0", "q0"]].reset_index(names="property_id").assign(zip=["98101", "98102"])
    rel = zr.add_relative_features(sales, typical)
    assert rel["property_id"].tolist() == ["p0"]
    assert rel["rel_log_living_sqft"].iloc[0] == pytest.approx(np.log(0.5))


def test_city_id_and_test_city_parcels(tmp_path):
    assert kcc.city_id("Federal Way") == "federal_way"
    path = tmp_path / "parcels.csv"
    pd.DataFrame({
        "PIN": ["123", "123", "456", "789"], "ADDR_FULL": ["1 MAIN ST", "1 MAIN ST UNIT B", "2 OAK AVE", "3 ELM"],
        "ZIP5": ["98004", "98004", "98032", None], "LAT": [47.6, 47.6, 47.4, 47.5], "LON": [-122.2, -122.2, -122.2, -122.3],
        "PRIMARY_ADDR": [1, 0, 1, 1], "CTYNAME": ["Bellevue", "Bellevue", "Kent", "Kent"],
    }).to_csv(path, index=False)
    p = kcc.read_test_city_parcels(path)
    assert p["property_id"].tolist() == ["0000000123", "0000000456"]   # duplicate PIN kept once, no-ZIP row dropped
    assert p["city_id"].tolist() == ["bellevue", "kent"]
    assert p["address"].iloc[1].endswith(", Kent, WA 98032")


def test_trained_on_seattle_transfers_to_an_unseen_city(synthetic, tmp_path):
    artifact, report = zr.train_zip_relative(*synthetic["sea"], *synthetic["far"], synthetic["history"])
    assert artifact["input_set"] in zr.SERVED_VARIANTS
    kent = report["test_cities"]["Kent"]["test_rows"]
    chosen = "zip_relative_" + artifact["input_set"]
    # Kent's test rows are later than every Seattle training sale.
    assert report["test_cities"]["Kent"]["splits"]["test"]["from"] > report["seattle_splits"]["val"]["to"]
    assert kent[chosen]["mae"] < 0.6 * kent["zillow_zip_value"]["mae"]
    assert kent[chosen]["mae"] < 0.6 * kent["seattle_house_model_absolute"]["mae"]
    # The raw house record does not transfer: the bigger suburban houses look expensive to it.
    assert kent[chosen]["mae"] < 0.6 * kent["zip_relative_house_record"]["mae"]
    assert report["test_cities"]["Kent"]["zip_relative_vs_zillow"]["verdict"] == "the second model is better"
    assert 0.6 <= report["test_cities"]["Kent"]["interval_coverage_on_test_rows"] <= 1.0
    assert "local_house_model_reference" in kent
    assert report["all_test_cities"][chosen]["n"] == kent[chosen]["n"]

    # The served model predicts from the house record and the ZIP value only.
    far_sales, far_feats, far_parcels = synthetic["far"]
    row = far_sales.head(3).merge(far_feats, left_on="property_id", right_index=True)
    row["zhvi_usd"] = 500_000.0
    row = zr.add_relative_features(row, zr.zip_typical_houses(far_feats, far_parcels))
    moved = row.assign(lat=0.0, lon=0.0, record_date="1999-01-01")
    assert artifact["model"].predict(row) == pytest.approx(artifact["model"].predict(moved))

    doc = tmp_path / "report.md"
    zr.write_markdown(report, doc)
    text = doc.read_text(encoding="utf-8")
    assert "### Kent" in text and "Zillow ZIP value alone" in text and chr(8212) not in text
