# Architecture and data flow

## The pieces

```
 Google Maps tab                Chrome extension                     Your computer
+----------------+   URL only  +---------------------+   JSON     +-----------------------+
| google.com/maps| ----------> | content.js          |            | FastAPI backend       |
| (you browse    |             |   reads the page    |            |  /health /cities      |
|  and pick a    |             |   address (URL)     |            |  /resolve /estimate   |
|  house here)   |             | side panel          | <--------> |  loads trained models |
+----------------+             |   parse URL         |  127.0.0.1 |  furniture scenario   |
                               |   confirm property  |    :8000   +-----------+-----------+
                               |   show 4 estimates  |                        |
                               +---------------------+              fixed hosts only
                                                                              |
                                              +-------------------------------+----------------+
                                              | OpenStreetMap / US Census: address lookup      |
                                              | Mapillary: licensed street photo (optional)    |
                                              +------------------------------------------------+

 Training pipeline (run by hand, offline)
 download -> build_dataset + validate -> [fetch_images -> embed_images] -> train -> evaluate
   Zillow CSVs, Census ZIP centers, King County sales and parcels      models/real/bundle.joblib
```

## What happens when you ask for an estimate

1. **You select a house on Google Maps.** The content script (`extension/content.js`) notices the
   page address (URL) changed and sends that URL to the side panel. It reads nothing else.
2. **The side panel parses the URL** (`extension/lib/maps_url.js`). A Google Maps URL can hold two
   positions: the camera position after `@`, and the selected place's own position in `!3d...!4d...`.
   Only an address or the place position identifies a house. A Street View camera position is
   treated as "somewhere near a house".
3. **Resolve.** The panel sends the address or position to `POST /resolve`. The backend answers with
   candidate properties. For Seattle it matches against the public list of single-family parcels.
   For a bare map position it returns the nearest houses and asks you to pick. Nothing is preselected
   unless the backend is sure.
4. **You confirm** the house and that it is a detached single-family home.
5. **Estimate.** The panel sends the confirmed position to `POST /estimate` with a request id.
6. **The backend** (`backend/app/estimator.py`) checks the city, runs the models, optionally looks
   for a licensed street photo, adds the furniture scenario, rejects impossible numbers, and returns
   the four estimates with labels, ranges where valid, model version and data period.
7. **The panel shows the result**, but only if it belongs to the house that is confirmed right now
   and carries the request id that is still current (`extension/lib/state.js`).

## Why it is built this way

| Decision | Reason |
|---|---|
| The extension reads only the URL | It needs the least permission, and it never touches map or Street View imagery |
| Secrets stay on the backend | The Mapillary token is never in the extension |
| Plain JavaScript, no build step | The extension folder can be loaded as it is and read line by line |
| State rules are a pure function | The "never show a stale result" rules can be tested without a browser |
| Files instead of a database | CSV, JSON and one joblib file are enough for a prototype |
| Backend and pipeline share `core/` | Features are built the same way in training and at prediction time |
| Demo and real models in separate folders | A synthetic model can never be served as a real one. The backend refuses to start a mismatch |

## Stale results and several tabs

- Every request gets an id (`r1`, `r2`, ...). An answer is accepted only if its id is the pending one.
  The backend echoes the id back and the panel checks it.
- Each selection has a key (the address, or the rounded Street View position). When the key changes,
  candidates and results are cleared before anything else happens.
- Panning, zooming or turning the Street View camera does not change the key, so results stay.
- When you switch tabs, the panel asks the new front tab for its URL. A different house clears the
  result. Messages from tabs that are not in front are ignored.
- If the backend goes down after a good estimate, the last estimate for the same house stays visible
  with an error message and a **Try again** button.

## Backend safety

- Input is validated by typed models (`backend/app/schemas.py`). Unknown fields are refused.
- Errors use one structure: `{"error": {"code", "message", "fields"}}`. Internal details are not sent.
- Request bodies are capped at 10 kB. Each client gets 30 requests per minute on the POST endpoints.
- `/estimate` has a 25 second limit. Outbound calls have their own shorter limits.
- The backend only contacts fixed hosts: `nominatim.openstreetmap.org`, `geocoding.geo.census.gov`,
  `graph.mapillary.com`, and Mapillary's image hosts for a thumbnail link that came from Mapillary.
  It never fetches a link supplied by a user.
- Cross-origin access is limited to Chrome extension pages, and can be pinned to one extension id
  with `HOMELENS_EXTENSION_ID`.
- Logs hold the path, status code and time. They hold no addresses, coordinates or request bodies.

## Model versions

`python -m pipeline.train` writes `models/real/bundle.joblib` and `models/real/manifest.json`. The
manifest records the model version (date plus a hash of the training tables), the feature version,
the data period, and every evaluation number. `docs/evaluation.md` is generated from that manifest.
