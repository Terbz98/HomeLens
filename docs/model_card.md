# Model card: HomeLens

Every dollar figure below is copied from [evaluation.md](evaluation.md), which is generated from the
training run. If the two ever disagree, `evaluation.md` is right. A test
(`pipeline/tests/test_docs.py`) fails when a figure here is not found there.

## What the models are for

Estimating the current market purchase price and monthly rent of a detached single-family home in
Seattle, San Francisco or Portland, from its location. A licensed street photo of the exterior was
tested as a second input. It did not lower the error, so the app does not use it (see "Does the
exterior photo help?" below).

Not for: appraisals, lending or insurance decisions, price forecasts, other property types, other
cities, or anything where a wrong number has real consequences.

## Inputs

| Input | Used |
|---|---|
| Latitude and longitude of the confirmed house | Yes |
| Month (for the Seattle sale model) | Yes |
| Licensed street photo of the exterior (Mapillary) | Tested, currently not used by the app (it did not lower the error) |
| Bedrooms, bathrooms, floor area, interior condition | No. The app cannot know them |
| Listing text, price tags or watermarks in images | No |
| Anything about the people who live there | No |

## The models

| Model | Label | Rows | Method |
|---|---|---|---|
| Area sale model | Zillow ZIP home value, single-family, 2026-09 | 430 ZIPs in the three metro areas | Distance-weighted nearest neighbors on location |
| Area rent model | Zillow ZIP rent index, all home types, 2026-09 | 328 ZIPs | Distance-weighted nearest neighbors on location |
| Seattle per-house sale model | Completed sale price, King County record | 9,652 sales, 2023-11-05 to 2026-11-04 | Distance-weighted nearest neighbors on location |
| Seattle photo model (trained, not used by the app) | Same sales, for houses with a matched photo | 1,705 sales | Fixed ResNet18 embedding, PCA, ridge regression on what the location model missed |

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

Measured answer: no, not with this setup. The app therefore gives location-only estimates.

**Coverage.** Mapillary had a usable photo (taken 3 to 40 m away and, when the direction is known,
pointing toward the house) for 1,703 of 2,997 sampled sold Seattle houses, which is 56.8%. For random
spots near ZIP centers the share was 73.3% in San Francisco, 50.9% in Portland and 44.2% in Seattle.

**Seattle, per house.** On the 327 test sales that have a photo:

| Model | MAE | RMSE |
|---|---:|---:|
| Location only | $403,927 | $879,130 |
| Location plus photo | $403,728 | $880,442 |

The photo lowered the MAE by $200, which is 0.05%. To see whether that could be chance, the 327 test
rows were resampled 2,000 times. The change in MAE ranged from $4,698 higher to $5,211 lower (95% of
the resamples), so zero is well inside the range: no measurable difference. The RMSE got slightly worse.

The model selection also points the same way. Out of nine photo models it chose the smallest and most
heavily shrunk one (`pca8_ridge100`), which is what happens when the photo numbers carry almost no
usable signal.

**Rule for using the photo model in the app.** It is used only if it lowers the MAE on the calibration
split by at least 2%. It lowered it by 0.04%, so it is not used. In the first version of the code the
rule was "any lower error". That rule would have switched the photo model on because of a difference
far inside the noise, so it was tightened after this result was seen. The rule looks only at the
calibration split, not at the test split.

**Area experiment (not used by the app).** 463 photos near random spots in the three cities, each
labeled with its ZIP's Zillow value, with whole ZIPs held out. Typical home value: location only
$285,597, location plus photo $277,606. Resampling whole ZIPs gives a range from $1,285 higher to
$18,427 lower, so again no measurable difference. Monthly rent: $319 against $323, no measurable
difference.

**Likely reasons (not tested, the author's reading).** Most Mapillary photos are dashcam frames that
look along the road, not at the house. Many are several years old. The encoder is a general ImageNet
network that was never tuned on houses. Only 1,208 training sales have a photo. And much of a house's
price comes from its interior size and condition, which no street photo shows.

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
  wrong house. The photo model is not used by the app. If a later version uses one, the app shows the
  photo and lets the user reject it (this path runs in demo mode today).
- The photo result is for one encoder, one photo source and one city. It does not show that the
  exterior of a house carries no price information.
- The served Seattle model is refit on all rows, including the test rows, so its own error is not
  measured directly. The reported error is for the version trained without the test rows.
- Sale filters drop unusual sales (related parties, estates, quit claims), so the model reflects
  ordinary open-market sales only.
