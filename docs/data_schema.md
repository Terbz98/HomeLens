# Data schema and validation

HomeLens keeps two kinds of table apart on purpose.

## 1. Records table (individual properties)

One row is one priced event for one property. This is the import schema for real sale or rental
data. `data/processed/seattle_sales.csv` and `data/fixtures/synthetic_seattle_sales.csv` use it.

| Column | Required | Meaning |
|---|---|---|
| `property_id` | yes | Stable ID of the house, for example a parcel number |
| `address` | yes | Street address |
| `zip` | yes | 5 digit ZIP code |
| `city_id` | yes | `seattle`, `san_francisco` or `portland` |
| `lat`, `lon` | yes | Position of the house |
| `property_type` | yes | Must be `detached_single_family` |
| `record_date` | yes | `YYYY-MM-DD` sale or listing date |
| `target_type` | yes | `sale` or `rent` |
| `price_usd` | yes | Sale price, or rent for one month |
| `currency` | yes | Must be `USD` |
| `price_unit` | yes | `total` for sales, `per_month` for rents |
| `price_basis` | yes | `completed_transaction` or `asking` |
| `source` | yes | Where the row came from |
| `synthetic` | yes | `True` only for made-up fixture rows |
| `furnished` | no | `yes`, `no` or `unknown`. Verified status only |
| `lease_months` | no | Rentals: lease length |
| `utilities_included` | no | Rentals: `yes`, `no` or `unknown` |
| `image_ref`, `image_provider`, `image_license`, `image_captured_at`, `image_distance_m` | no | The matched exterior photo and where it came from |

Bedrooms, bathrooms and floor area are deliberately not in the schema. The app cannot know them
when a user asks for an estimate, so no model may use them.

### Validation command

```bash
python -m pipeline.validate path/to/records.csv --clean-out cleaned.csv --report-out report.json
```

| Check | Blocking | What it catches |
|---|---|---|
| `required_columns` | yes | A required column is missing |
| `missing_values` | yes | An empty required field |
| `currency_not_usd` | yes | Any currency other than USD |
| `unit_mismatch` | yes | A sale not marked `total`, or a rent not marked `per_month` |
| `unknown_target_type`, `unknown_price_basis` | yes | Values outside the allowed lists |
| `invalid_coordinates` | yes | Not a number, out of range, or (0, 0) |
| `unsupported_city`, `unsupported_property_type` | yes | Outside the supported scope |
| `implausible_price` | yes | Sale outside $50,000 to $30,000,000, rent outside $300 to $50,000 per month |
| `invalid_or_future_date` | yes | Unreadable or future dates |
| `duplicate_rows` | yes | Same house, date, target and price. The first copy is kept |
| `repeated_properties` | no | The same house more than once. Kept, and never split across training and testing |
| `unknown_furnished_value` | yes | Furnishing status outside `yes`, `no`, `unknown` |
| `image_property_mismatch` | no | Photo more than 40 m from the house. The photo is dropped, the price row is kept |
| `image_without_license` | no | Photo with no license recorded. The photo is dropped |

Rows that fail a blocking check are left out of the cleaned file, and the command exits with code 1.

## 2. Area table (ZIP level typical values)

`data/processed/zip_values.csv`. One row per ZIP in the three metro areas.

| Column | Meaning |
|---|---|
| `zip`, `city`, `state`, `metro`, `metro_id` | The ZIP and where Zillow files it |
| `supported_city_id` | City id when the ZIP is inside a supported city, otherwise empty |
| `lat`, `lon` | ZIP center point from the US Census Gazetteer |
| `zhvi_usd`, `zhvi_month` | Zillow Home Value Index, single-family homes, newest month |
| `zori_usd_month`, `zori_month` | Zillow Observed Rent Index, all home types, newest month |
| `synthetic` | `True` only for fixtures |

These are area averages. They are never copied into the records table and never described as the
price of an individual house.

## How the real tables are built

`python -m pipeline.build_dataset` reads the raw downloads and prints every check.

King County sales are filtered to ordinary open-market sales of one improved single-family parcel:
last 3 years, one parcel per sale, price above zero, principal use residential, property class
residential improved, property type single-family household, warranty deed or statutory warranty
deed, no sale reason, no sale warning. The rows are then joined to Seattle parcels whose present use
is single family. Buyer and seller names are never loaded. The counts at each step are saved in
`data/processed/validation_report.json`.

## Provenance

`python -m pipeline.download` writes `data/raw/provenance.json`: for every file the source link,
retrieval time, size, SHA-256 hash and license note. Photo downloads write
`data/processed/image_search_log.csv`: provider, photo id, contributor, capture date, license and the
distance between the camera and the house.
