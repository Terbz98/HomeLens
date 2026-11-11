# HomeLens

HomeLens is a Chrome extension for Google Maps. You select a house on the map, confirm it, and a side
panel shows four estimates:

1. Purchase price without furniture (US dollars)
2. Purchase price with furniture (US dollars)
3. Monthly rent without furniture (US dollars per month)
4. Monthly rent with furniture (US dollars per month)

It is a student prototype for an Introduction to Machine Learning course. The numbers are estimates
of current market prices. They are not appraisals, forecasts or guarantees.

**Supported:** detached single-family homes in Seattle, San Francisco and Portland. Anything else is
reported as unsupported.

**You never enter** bedrooms, bathrooms, floor area or interior details. The only inputs are the
location of the house and, where a licensed one exists and a photo model is served, a street photo.

## What each number really is

| Estimate | Seattle | San Francisco, Portland |
|---|---|---|
| Buy, without furniture | Per-house model trained on real King County sale records | Area-level: typical single-family home value near that spot (Zillow ZIP values) |
| Rent, without furniture | Area-level: Zillow ZIP rent index, scaled to single-family homes | same |
| Buy, with furniture | The number above plus the assumed resale value of one fixed furniture package | same |
| Rent, with furniture | The number above plus a scenario amount per month | same |

The side panel labels every number as a per-house model, an area-level figure, or a scenario. The
furnished numbers are assumptions, not predictions: a photo of the outside cannot show furniture.
Measured accuracy is in [docs/evaluation.md](docs/evaluation.md) and explained in
[docs/model_card.md](docs/model_card.md).

## Parts

```
extension/   Chrome extension (Manifest V3, plain JavaScript, no build step)
backend/     FastAPI server that makes the estimates
pipeline/    data download, validation, training and evaluation scripts
core/        code shared by the backend and the pipeline (features, furniture, photo providers)
config/      supported cities, data sources, furniture package
data/        fixtures/ holds small SYNTHETIC test data. Real data is downloaded, never committed
docs/        architecture, licensing, data schema, model card, walkthrough, proposal script
```

## 1. Install

You need Python 3.9 or newer, Google Chrome 116 or newer, and Node.js 20 or newer (only to run the
extension tests).

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

## 2. Get data and train (real predictions)

One command downloads the public data (about 290 MB), checks it, trains the models and writes the
evaluation report. It takes a few minutes.

```bash
python -m pipeline.run_all
```

The separate steps, if you want to run them one at a time:

```bash
python -m pipeline.download
```

```bash
python -m pipeline.build_dataset
```

```bash
python -m pipeline.train
```

```bash
python -m pipeline.evaluate
```

Trained models are saved in `models/real/`. The report is written to `docs/evaluation.md`.

### Optional: street photos

Photos come from Mapillary, not from Google (see [docs/licensing.md](docs/licensing.md)). Put a free
Mapillary client token in `.env` as `MAPILLARY_TOKEN=...`, then:

```bash
python -m pipeline.fetch_images --houses 3000 --area-points 12
```

```bash
python -m pipeline.embed_images
```

```bash
python -m pipeline.train
```

```bash
python -m pipeline.evaluate
```

The photo model is used by the app only if it lowers the error on held-out sales by at least 2%.
In the current run it did not (the difference was 0.04%), so the app gives location-only estimates.
The full result is in [docs/evaluation.md](docs/evaluation.md) and [docs/model_card.md](docs/model_card.md).

## 3. Start the backend

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Check it: open http://127.0.0.1:8000/health. The typed API description is at
http://127.0.0.1:8000/docs.

### Demo mode (no downloads)

Demo mode serves models trained on the small synthetic fixtures in `data/fixtures`. Every response
and the side panel are stamped "Demo mode". Demo numbers are made up and say nothing about real houses.

```bash
python -m pipeline.train --fixtures
```

```bash
HOMELENS_DEMO=1 uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## 4. Load the extension in Chrome

1. Open `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and choose the `extension` folder of this project.
4. Click the puzzle icon in the toolbar and pin **HomeLens**.
5. Reload any Google Maps tab that was already open.

## 5. Use it on Google Maps

1. Open https://www.google.com/maps and click a house, or search for its address, so that the
   address panel is showing.
2. Click the HomeLens toolbar icon. The side panel opens.
3. Check the address under **Property**. If you were in Street View with no address selected,
   HomeLens lists the nearest houses (Seattle) or asks for the address. It never picks one for you.
4. Choose **Detached single-family house**.
5. Click **Estimate**.

If the address is not detected, type it (with city and state) or paste a map pin as
`latitude, longitude`.

## How to tell demo from real predictions

| You see | Meaning |
|---|---|
| Purple **Demo mode** pill and banner, tags "Demo, not real" | Synthetic data. Not a valuation. |
| Tag **Trained model** + **Per-house model** | Predicted by a model trained on real individual sales |
| Tag **Trained model** + **Area-level figure** | A typical value for the area, not for this house |
| Tag **Scenario** + **Assumption added** | Unfurnished number plus a fixed assumed furniture amount |
| "Range ... to ..." under a price | A range checked on held-out sales, with its measured coverage |
| "A reliable range is unavailable" | No valid way to check a range for that number |

`GET /health` also reports `"mode": "real"` or `"mode": "demo"`, and every estimate carries
`is_demo`, `result_kind` and `model_version`.

## Import your own data

Individual sale or rental records can be added in the schema described in
[docs/data_schema.md](docs/data_schema.md). Check a file with:

```bash
python -m pipeline.validate path/to/records.csv --clean-out cleaned.csv
```

The command lists every check and exits with an error if a blocking problem is found.

## Tests

```bash
python -m pytest
```

```bash
cd extension && npm test && npm run check
```

## Credentials and permissions

| What | Needed for | Where it goes |
|---|---|---|
| Mapillary client token (free) | Street photos. Optional. | `MAPILLARY_TOKEN` in `.env`, on the server only |
| Nothing else | Zillow, Census and King County data are public downloads | |

The extension holds no secrets. Chrome permissions it asks for:

| Permission | Why |
|---|---|
| `sidePanel` | To show the HomeLens panel next to the page |
| Content script on `https://www.google.com/maps/*` only | To read the Google Maps page address (URL), which contains the selected address. It reads nothing else on the page and runs on no other site. |
| Host access to `http://127.0.0.1:8000/*` | To ask the local backend for estimates |

It does not ask for `tabs`, `history`, `activeTab`, `storage` or access to all sites.

## Known limitations

- San Francisco and Portland buy prices, and all rents, are area-level figures. Free per-house labels
  were only available for Seattle.
- Location alone is a weak predictor of one house's price. See the measured error in
  [docs/evaluation.md](docs/evaluation.md).
- Property type is confirmed by you. HomeLens does not detect it.
- The photo model did not beat the location-only model, so every real estimate is location-only.
  Street photo coverage is also uneven: 56.8% of sampled Seattle houses had a usable photo.
- Google Street View images are never read, stored or sent to a model. That part of the original
  idea is blocked by Google's terms. See [docs/licensing.md](docs/licensing.md).
- The backend is meant to run on your own computer. It has no accounts and no HTTPS.

## Troubleshooting

| Problem | Fix |
|---|---|
| Panel says **Backend unavailable** | Start the backend (step 3) and click **Try again** |
| Panel says **This tab is not Google Maps** on a Maps tab | Reload the Maps tab once after loading the extension |
| Panel says **Insufficient data** | Train the models (step 2) or start in demo mode |
| Typed address is not found | Include house number, street, city and state. The US Census lookup does not answer from some networks outside the United States, so OpenStreetMap is tried first. |
| "encountered in matmul" warnings while training on a Mac | A known false warning from NumPy on macOS. The results are unaffected. |
| Download of a Zillow file fails | Open the link printed by the script in a browser and save the file to the path it prints |

## Data and credits

Home value and rent indexes: Zillow Research (ZHVI and ZORI). Sale records and parcel positions: King
County, Washington. ZIP center points: US Census Bureau. Address lookup: OpenStreetMap Nominatim, data
(c) OpenStreetMap contributors, and the US Census Bureau geocoder. Street photos: Mapillary
contributors, CC BY-SA 4.0. Details in [docs/licensing.md](docs/licensing.md).
