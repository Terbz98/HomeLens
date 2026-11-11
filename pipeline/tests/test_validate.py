"""Data validation checks on small hand-made tables."""
import pandas as pd

from pipeline import schema
from pipeline.validate import has_blocking_problem, validate_area_table, validate_records


def good_row(**changes):
    row = {
        "property_id": "P1", "address": "1 Test St, Seattle, WA 98103", "zip": "98103", "city_id": "seattle",
        "lat": 47.66, "lon": -122.34, "property_type": "detached_single_family", "record_date": "2025-06-01",
        "target_type": "sale", "price_usd": 900000.0, "currency": "USD", "price_unit": "total",
        "price_basis": "completed_transaction", "source": "test", "synthetic": True,
    }
    row.update(changes)
    return row


def flagged(results, check):
    return [r for r in results if r["check"] == check][0]["rows_flagged"]


def test_clean_rows_pass_every_check():
    df = pd.DataFrame([good_row(), good_row(property_id="P2", price_usd=700000.0)])
    results, cleaned = validate_records(df)
    assert not has_blocking_problem(results)
    assert len(cleaned) == 2


def test_missing_required_column_blocks_everything():
    df = pd.DataFrame([good_row()]).drop(columns=["price_usd"])
    results, cleaned = validate_records(df)
    assert has_blocking_problem(results)
    assert len(cleaned) == 0


def test_each_problem_is_flagged_and_removed():
    rows = [
        good_row(property_id="ok"),
        good_row(property_id="no_price", price_usd=None),
        good_row(property_id="eur", currency="EUR"),
        good_row(property_id="unit", price_unit="per_month"),
        good_row(property_id="coords", lat=0.0, lon=0.0),
        good_row(property_id="lat_range", lat=123.0),
        good_row(property_id="cheap", price_usd=5.0),
        good_row(property_id="huge", price_usd=9e9),
        good_row(property_id="future", record_date="2999-01-01"),
        good_row(property_id="bad_date", record_date="not a date"),
        good_row(property_id="condo", property_type="condo"),
        good_row(property_id="city", city_id="los_angeles"),
        good_row(property_id="rent_ok", target_type="rent", price_unit="per_month", price_usd=3000.0,
                 price_basis="asking"),
        good_row(property_id="rent_as_total", target_type="rent", price_unit="total", price_usd=3000.0),
    ]
    results, cleaned = validate_records(pd.DataFrame(rows))
    assert flagged(results, "missing_values") == 1
    assert flagged(results, "currency_not_usd") == 1
    assert flagged(results, "unit_mismatch") == 2
    assert flagged(results, "invalid_coordinates") == 2
    assert flagged(results, "implausible_price") == 2
    assert flagged(results, "invalid_or_future_date") == 2
    assert flagged(results, "unsupported_property_type") == 1
    assert flagged(results, "unsupported_city") == 1
    assert sorted(cleaned["property_id"]) == ["ok", "rent_ok"]


def test_duplicates_removed_but_repeat_sales_kept():
    rows = [
        good_row(property_id="A"),
        good_row(property_id="A"),                                   # exact duplicate
        good_row(property_id="A", record_date="2026-01-15", price_usd=950000.0),   # a later resale
    ]
    results, cleaned = validate_records(pd.DataFrame(rows))
    assert flagged(results, "duplicate_rows") == 1
    assert flagged(results, "repeated_properties") == 2
    assert len(cleaned) == 2


def test_far_away_photo_is_dropped_but_price_row_kept():
    rows = [
        good_row(property_id="near", image_ref="img1", image_license="CC BY-SA 4.0", image_distance_m=12.0),
        good_row(property_id="far", image_ref="img2", image_license="CC BY-SA 4.0",
                 image_distance_m=schema.MAX_IMAGE_DISTANCE_M + 100),
        good_row(property_id="nolicense", image_ref="img3", image_distance_m=10.0),
    ]
    results, cleaned = validate_records(pd.DataFrame(rows))
    assert flagged(results, "image_property_mismatch") == 1
    assert flagged(results, "image_without_license") == 1
    by_id = cleaned.set_index("property_id")
    assert by_id.loc["near", "image_ref"] == "img1"
    assert pd.isna(by_id.loc["far", "image_ref"])
    assert pd.isna(by_id.loc["nolicense", "image_ref"])
    assert len(cleaned) == 3


def test_area_table_checks():
    table = pd.DataFrame([
        {"zip": "98103", "city": "Seattle", "state": "WA", "metro": "m", "metro_id": "seattle",
         "supported_city_id": "seattle", "lat": 47.6, "lon": -122.3, "zhvi_usd": 900000.0, "zhvi_month": "2026-01",
         "zori_usd_month": 2500.0, "zori_month": "2026-01", "synthetic": True},
        {"zip": "98103", "city": "Seattle", "state": "WA", "metro": "m", "metro_id": "seattle",
         "supported_city_id": "seattle", "lat": 47.6, "lon": -122.3, "zhvi_usd": 12.0, "zhvi_month": "2026-01",
         "zori_usd_month": None, "zori_month": "2026-01", "synthetic": True},
    ])
    results = validate_area_table(table)
    assert flagged(results, "duplicate_zip") == 1
    assert flagged(results, "implausible_home_value") == 1
    assert flagged(results, "no_rent_value") == 1
    assert has_blocking_problem(results)
