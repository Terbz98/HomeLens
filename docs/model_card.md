# Model card: HomeLens

Every dollar figure below is copied from [evaluation.md](evaluation.md), which is generated from the
training run. If the two ever disagree, `evaluation.md` is right. A test
(`pipeline/tests/test_docs.py`) fails when a figure here is not found there.

## What the models are for

Estimating the current market purchase price and monthly rent of a detached single-family home in
Seattle, San Francisco or Portland, from its location and, where available, a licensed street photo.

Not for: appraisals, lending or insurance decisions, price forecasts, other property types, other
cities, or anything where a wrong number has real consequences.

## Inputs

| Input | Used |
|---|---|
| Latitude and longitude of the confirmed house | Yes |
| Month (for the Seattle sale model) | Yes |
| Licensed street photo of the exterior (Mapillary) | Only where a photo exists and the photo model is served |
| Bedrooms, bathrooms, floor area, interior condition | No. The app cannot know them |
| Listing text, price tags or watermarks in images | No |
| Anything about the people who live there | No |

## The models

| Model | Label | Rows | Method |
|---|---|---|---|
| Area sale model | Zillow ZIP home value, single-family, 2026-09 | 430 ZIPs in the three metro areas | Distance-weighted nearest neighbors on location |
| Area rent model | Zillow ZIP rent index, all home types, 2026-09 | 328 ZIPs | Distance-weighted nearest neighbors on location |
| Seattle per-house sale model | Completed sale price, King County record | 9,652 sales, 2023-11-05 to 2026-11-04 | Distance-weighted nearest neighbors on location |
| Seattle photo model | Same sales, for houses with a matched photo | See evaluation.md | Fixed ResNet18 embedding, PCA, ridge regression on what the location model missed |

Each model is compared against a simple baseline and against gradient boosting on the same inputs.
The choice between candidates is made on validation data, never on the test data.

## Results

### Seattle per-house sale price (test split: the newest 1,931 sales, median price $1,125,000)

| Model | MAE | RMSE |
|---|---:|---:|
| Baseline: city median | $517,088 | $937,696 |
| Baseline: ZIP median of past sales | $444,860 | $843,839 |
| Location only (chosen model) | $363,904 | $692,018 |

On the 1,658 test sales where Zillow has a ZIP value for the sale month, using that ZIP value as the
prediction gives MAE $494,251, against $373,052 for the location-only model on the same rows.

Plain reading: location alone gets a typical Seattle house wrong by about a third of its price. It
beats the baselines, but it cannot tell a large renovated house from a small worn one next door.

### Area models (each ZIP predicted by a model that never saw it)

| City ZIPs | Home value: baseline MAE | Home value: model MAE | Rent: baseline MAE | Rent: model MAE |
|---|---:|---:|---:|---:|
| Seattle (23) | $269,274 | $114,710 | $251 | $294 |
| San Francisco (20) | $794,034 | $478,867 | $1,354 | $514 |
| Portland (27 for value, 26 for rent) | $146,279 | $99,802 | $213 | $193 |

The baseline is the median of the metro area. For Seattle rent the model is worse than that baseline.
These are errors against a ZIP's typical value, not against individual houses. There are few ZIPs per
city, so the figures are rough.

### Does the exterior photo help?

See the "Seattle photo model" and "Area photo experiment" sections of [evaluation.md](evaluation.md).
Until photos have been downloaded and the photo model has been trained, that section says
"Not trained", and the app gives location-only estimates.

## Prediction ranges

- **Seattle buy price:** split conformal range. The relative errors on a calibration split (962
  sales) set the width, and coverage is measured on the test split. The current width, target and
  measured coverage are in evaluation.md under "Prediction range".
- **San Francisco and Portland buy price, and all rents:** no range. These are area-level figures
  and there are no per-house labels to check a range against. The app says a reliable range is
  unavailable.
- **Furnished numbers:** no range. They add an assumption, not a prediction.

No confidence percentages are invented. Every coverage number shown in the app was measured.

## Leakage controls

- Seattle sales are split by date: oldest for training, then validation, calibration, and the newest
  for testing.
- A house that sold more than once is kept only in the newest split it reaches, so no house is on
  both sides. The evaluation reports the overlap count, which is 0.
- Area models are evaluated with every ZIP held out in turn.
- Scaling, PCA and every other fitted step are fitted on training rows only.
- One photo is used for one house only, so the same photo cannot be in training and testing.
- Only sale date and location go into the model. Nothing recorded after the sale is used.

## Furnished estimates

A photo of the outside cannot show furniture, and the interior size is unknown. HomeLens therefore
uses one fixed package for every house, defined in `config/furniture_package.json`.

| Item | Value | Basis |
|---|---:|---|
| Package cost when new | $6,250 | Sum of assumed item prices |
| Added to the purchase price | $2,188 | 35% assumed resale fraction |
| Added to monthly rent | $174 | Cost when new spread over 36 months |

All three are assumptions set by the project author and can be edited. None is a measured premium.
The data sources contain no verified furnished and unfurnished rental records, so no furnishing
effect was learned from data, and the furnished rent is labeled a scenario estimate.

## Single-family rent adjustment

Zillow publishes a ZIP-level rent index only for all home types together. The app multiplies it by
the metro-wide ratio of the single-family rent index to the all-homes rent index (the ratios are in
evaluation.md). This assumes the ratio is the same in every ZIP of a metro, which is a simplification.

## Known limitations

- Per-house labels exist for Seattle only. Elsewhere the buy price is an area-level figure.
- Rent is area-level everywhere.
- Location alone has a large error for individual houses.
- The property type is confirmed by the user, not detected.
- Street photos are often dashcam frames looking along the road, may be years old, and may show the
  wrong house. The app shows the photo it used and lets the user reject it.
- The served Seattle model is refit on all rows, including the test rows, so its own error is not
  measured directly. The reported error is for the version trained without the test rows.
- Sale filters drop unusual sales (related parties, estates, quit claims), so the model reflects
  ordinary open-market sales only.
