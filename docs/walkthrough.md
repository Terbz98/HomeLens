# Demonstration walkthrough

About five minutes. It uses real trained models. For a no-download version, start the backend in
demo mode instead (see the README): every number will then be stamped "Demo mode".

## Before you start

1. Train once: `python -m pipeline.run_all`
2. Start the backend: `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`
3. Load the `extension` folder in `chrome://extensions` (Developer mode, Load unpacked) and pin it.

## Steps

| Step | Do this | You should see |
|---|---|---|
| 1 | Open Google Maps and click the HomeLens icon | Side panel opens. Status **Select a house** |
| 2 | Search for `209 N 41st St, Seattle, WA` and select it | Status **Confirm property**. The address is listed with "Seattle: supported city" and "Matched to a single-family parcel" |
| 3 | Choose **Detached single-family house**, click **Estimate** | Status **Location-only estimate**, because the photo model is not in use (**Estimate ready** appears only when a photo was used). Four price cards |
| 4 | Read the Buy cards | "Without furniture" has a range and the tags "Trained model" and "Per-house model". "With furniture" is $2,188 higher and tagged "Scenario" |
| 5 | Read the Rent cards | Explicit "/month". Tagged "Area-level figure". No range, with the reason written out |
| 6 | Open **Furniture assumptions** and **About this estimate** | The fixed package, the model version, the estimate date and the data period |
| 7 | Open Street View on the same street without selecting an address | Status **Confirm property** with the three nearest houses. None is preselected. The old result is gone |
| 8 | Search for an address in San Francisco or Portland and estimate | Buy price tagged "Area-level figure", no range |
| 9 | Search for an address in Bellevue, WA | Status **Unsupported location** and no numbers |
| 10 | Back on a Seattle house, choose **Something else** as the property type | Status **Unsupported property** and no numbers |
| 11 | Stop the backend and click **Estimate again** on a house that has a result | Status **Backend unavailable**. The last estimate for that same house stays, with **Try again** |
| 12 | Select a different house while the backend is still stopped | The old numbers disappear. They are never shown under the new address |

## Points to make while showing it

- The extension reads only the Google Maps page address. It never touches Street View pictures.
- Every number says what kind of number it is.
- The range under the Seattle price was checked on sales the model never saw.
- The furnished numbers are an assumption, and the panel says so.
