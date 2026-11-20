# Model card: HomeLens

Every dollar figure below is copied from [evaluation.md](evaluation.md), which is generated from the
training run. If the two ever disagree, `evaluation.md` is right. A test
(`pipeline/tests/test_docs.py`) fails when a figure here is not found there.

## What the models are for

Estimating the current market purchase price and monthly rent of a detached single-family home in
Seattle, San Francisco or Portland. In Seattle the buy price uses the house's location, the county's
public record of the building (living area, grade, age, condition, rooms, lot, view) and, where one
exists, a licensed street photo. Elsewhere it uses location only.

Not for: appraisals, lending or insurance decisions, price forecasts, other property types, other
cities, or anything where a wrong number has real consequences.

## Inputs

| Input | Used |
|---|---|
| Latitude and longitude of the confirmed house | Yes |
| Month (for the Seattle sale models) | Yes |
| King County record of the house: living area, building grade, year built and renovated, condition, bedrooms, bathrooms, stories, finished basement, number of buildings, lot size, view, waterfront | Seattle, when the confirmed house matches a county parcel. The user never types these |
| Licensed street photo of the exterior (Mapillary) | Seattle, when a photo exists. Its measured effect is shown next to the estimate |
| Listing text, price tags or watermarks in images | No |
| Anything about the people who live there | No |

## The models

| Model | Label | Rows | Method |
|---|---|---|---|
| Area sale model | Zillow ZIP home value, single-family, 2026-09 | 430 ZIPs in the three metro areas | Distance-weighted nearest neighbors on location |
| Area rent model | Zillow ZIP rent index, all home types, 2026-09 | 328 ZIPs | Distance-weighted nearest neighbors on location |
| Seattle location-only sale model (used when no county record matches) | Completed sale price, King County record | 9,652 sales, 2023-11-05 to 2026-11-04 | Distance-weighted nearest neighbors on location |
| Seattle house-record sale model (used by the app) | Same sales | 9,475 sales | Gradient boosted trees on location, month and the house record |
| Seattle photo model (used by the app on top of the house-record model) | Same sales, houses with a matched photo | 5,479 sales | Fixed ConvNeXt-Tiny photo embedding, PCA, ridge regression on what the house-record model missed |

Each model is compared against simple baselines. The choice between candidates is made on validation
data or by cross-validation on training rows, never on the test data.

## Results

### Seattle buy price with the house record (test split: the newest 1,896 sales, median price $1,116,500)

| Model | MAE | RMSE |
|---|---:|---:|
| Baseline: city median | $511,161 | $906,039 |
| Baseline: ZIP median of past sales | $439,296 | $808,971 |
| Location only, same rows | $359,220 | $662,583 |
| Location plus house record (chosen model) | $163,388 | $313,307 |

The median error is 9.2% of the sale price, against 20.6% for location only. On the 1,629 test sales
where Zillow has a ZIP value for the sale month, that ZIP value is off by $486,713 on average and the
house-record model by $162,883. The 80% price range is plus or minus 19.3%, and it held the real
price for 81.3% of the test sales.

177 sales were left out because the house was built or renovated in or after the year it sold: the
county record shows today's building, which is not the one that was sold.

Plain reading: two neighbouring houses now get different prices, because a larger, newer or higher
grade house is priced higher. The record is what the location-only model was missing.

### Seattle buy price, location only (test split: the newest 1,931 sales, median price $1,125,000)

| Model | MAE | RMSE |
|---|---:|---:|
| Baseline: city median | $517,088 | $937,696 |
| Baseline: ZIP median of past sales | $444,860 | $843,839 |
| Location only (chosen model) | $363,904 | $692,018 |

This model is used only when the confirmed house cannot be matched to a county parcel.

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

Short answer: a photo carries a small real signal beyond location, but almost nothing beyond the
county record.

**Coverage.** Mapillary had a usable photo (taken 3 to 40 m away and, when the direction is known,
pointing toward the house) for 5,465 of 9,487 sold Seattle houses, which is 57.6%. For random spots
near ZIP centers the share was 73.3% in San Francisco, 50.9% in Portland and 44.2% in Seattle.

**Three ways to turn a photo into numbers were compared:** ResNet18 and ConvNeXt-Tiny (pretrained
image networks, used unchanged), and Harris corner features, the corner and edge measurements of DSP
Lab 7 counted in a 3 x 3 grid (`core/harris.py`). The encoder is chosen by cross-validation on the
training rows only.

**Photo on top of location only** (the professor's question), 1,095 test sales with a photo:

| Encoder | Without photo | With photo | Result |
|---|---:|---:|---|
| ResNet18 | $403,273 | $401,286 | No measurable difference |
| ConvNeXt-Tiny (chosen) | $403,273 | $400,472 | The photo model is better |
| Harris corners (DSP Lab 7) | $403,273 | $401,817 | The photo model is better |

ConvNeXt lowered the MAE by $2,801. Resampling the test rows 2,000 times gives a range from $384 to
$5,286 lower, so the gain is real, but it is under 1% of the error. The Harris features also gave a
small real gain.

**Photo on top of the house record** (what the app uses), 1,080 test sales with a photo:

| Encoder | Without photo | With photo | Result |
|---|---:|---:|---|
| ResNet18 | $175,673 | $176,741 | No measurable difference |
| ConvNeXt-Tiny (chosen) | $175,673 | $175,336 | No measurable difference |
| Harris corners (DSP Lab 7) | $175,673 | $177,869 | The photo model is worse |

Once the model knows the living area, grade and age, the photo adds nothing measurable. A plausible
reading (the author's, not tested): what the photo can show, such as size and how new and well kept
a house looks, is already in the record.

**How the app uses it.** The photo model is used whenever a photo is found (setting
`HOMELENS_PHOTO_MODEL=on`, the default), and the panel shows the photo and this test result next to
the estimate. With `HOMELENS_PHOTO_MODEL=auto` it is used only if it lowered the calibration error by
at least 2%. It lowered it by 0.62%, so in that mode it would not be used.

**Area experiment (not used by the app).** 463 photos near random spots in the three cities, each
labeled with its ZIP's Zillow value, with whole ZIPs held out. Typical home value: location only
$285,597, location plus photo $277,606. Resampling whole ZIPs gives a range from $1,285 higher to
$18,427 lower, so no measurable difference.

**Photo date.** A photo and a sale can be years apart. Across 5,479 Seattle sales with a dated photo,
the median gap is 5.4 years, only 14.5% of photos were taken within 3 years of the sale, and 347 were
taken after the sale. Checked with 5-fold cross-validation grouped by house (on top of the
house-record model):

| Photos taken | Sales | Without photo | With photo | Result |
|---|---:|---:|---:|---|
| Within 3 years of the sale | 796 | $149,803 | $149,840 | No measurable difference |
| 3 to 5 years | 1,495 | $157,413 | $156,243 | No measurable difference |
| More than 5 years | 3,188 | $142,107 | $141,631 | No measurable difference |
| Within 3 years, photo model trained on those photos only | 796 | $149,803 | $150,424 | No measurable difference |

Recent photos did not help more than old ones.

**Likely reasons the photo adds little (not tested, the author's reading).** Most Mapillary photos
are dashcam frames that look along the road, not at the house. The image networks were never tuned
on houses. Much of a house's price comes from its interior, which no street photo shows.

## Prediction ranges

- **Seattle buy price:** split conformal range. The relative errors on a calibration split set the
  width, and coverage is measured on the test split. Each Seattle model has its own range (house
  record: plus or minus 19.3%; with a photo: plus or minus 18.8%; location only: plus or minus
  38.0%). The details are in evaluation.md under "Prediction range".
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
- The county house record shows today's building. Sales of houses built or renovated in or after
  the sale year are dropped, so the record never describes a building that did not exist at the sale.
  Small changes the county did not record as a renovation can still differ from the sold house.

## Furnished estimates

A photo of the outside cannot show furniture, and the county record does not list it. HomeLens therefore
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
- Location alone has a large error for individual houses. Outside Seattle no house record is used.
- The county record can be out of date, and the app does not inspect the house.
- The property type is confirmed by the user, not detected.
- Street photos are often dashcam frames looking along the road, may be years old, and may show the
  wrong house. The app shows the photo it used and lets the user reject it.
- The photo result is for three encoders, one photo source and one city. It does not show that the
  exterior of a house carries no price information.
- The served Seattle model is refit on all rows, including the test rows, so its own error is not
  measured directly. The reported error is for the version trained without the test rows.
- Sale filters drop unusual sales (related parties, estates, quit claims), so the model reflects
  ordinary open-market sales only.
