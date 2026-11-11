# Data and image licensing notes

This file records what each source allows, what HomeLens does with it, and what was actually checked.
Terms were read on 2026-11-11. This is a student's reading of the terms, not legal advice.

## Summary

| Source | Used for | Redistributed in this repository | Status |
|---|---|---|---|
| Zillow Research (ZHVI, ZORI) | Area-level labels | No | Allowed with attribution |
| King County Assessor sales | Per-house sale labels, Seattle | No | Public record, non-commercial use, names dropped |
| King County GIS parcels | House positions and addresses, Seattle | No | Public open data, owner fields never requested |
| US Census ZCTA Gazetteer | ZIP center points | No | Public domain |
| OpenStreetMap Nominatim | Address lookup at run time | No | Allowed with attribution, light use |
| US Census geocoder | Address lookup at run time | No | Public service |
| Mapillary | Exterior photos for training and prediction | No | CC BY-SA 4.0, attribution shown |
| Google Maps and Street View imagery | **Not used** | No | **Blocked by Google's terms** |
| torchvision ResNet18 weights | Fixed photo encoder | No | See below |

## Google Maps and Street View: what is blocked and why

The original idea was to analyze the house exterior that the user sees in Street View. That is not
implemented, on purpose.

- **Google Maps Platform Terms of Service** (last modified August 25, 2026), section 3.2.3
  "Restrictions Against Misusing the Services": (a) no scraping, which names bulk download of
  Street View images; (b) no caching; (c) no creating content from Google Maps Content, which
  names building an index from Street View imagery in item (v) and using Google Maps Content to
  train, test, validate or fine-tune machine learning models in item (vii).
- **Google Maps / Google Earth Additional Terms of Service** (last modified January 27, 2026),
  section 2 "Prohibited Conduct": users may not copy the content or mass download it.

What HomeLens does instead:

- Google Maps stays the place where the user browses and selects a house.
- The extension reads only the page address (URL), which holds the address the user selected. It does
  not read pixels, take screenshots, capture the tab, call hidden Google endpoints or use a proxy.
- Exterior photos come from a separately licensed provider (Mapillary), matched to the confirmed
  house position.
- When no licensed photo is used, the result is labeled "Location-only estimate" and the panel says
  the visible exterior was not analyzed.

Still open: the consumer terms also restrict creating a new product or service based on Google
Maps. HomeLens only reads the URL of a page the user opened, and it is not published or distributed.
Publishing it in the Chrome Web Store would need this question answered by Google or a lawyer first.

## Zillow Research data

- Files: ZIP-level ZHVI for single-family homes, ZIP-level and metro-level ZORI.
- Zillow states that its downloadable housing data is free for public use with clear attribution
  to Zillow. Attribution is shown in the side panel and in the README.
- Zillow's Terms of Use forbid scraping listing pages with automated tools. HomeLens does not do
  that. It downloads the published research CSV files, once each, from Zillow's public file host.
  No listing, photo or Zestimate is collected.
- The CSV files and the trained models are not committed, so nothing derived from Zillow data is
  redistributed here.
- Not verified: the Zillow Research data page itself refused automated reading (HTTP 403), so the
  attribution wording was read from Zillow Group's public data page and from search results, not
  from that page. If you rely on this beyond coursework, read the page in a browser.

## King County, Washington

- **Assessor data download** (Real Property Sales): the download page shows a disclaimer that the
  files are public records and that Washington law, RCW 42.56.070(9), prohibits using lists of
  individuals for commercial purposes. HomeLens is non-commercial coursework, and buyer and seller
  names are never loaded (`pipeline/build_dataset.py`, `read_sales`).
- **GIS Open Data** (parcels with address): only parcel number, address, ZIP and coordinates are
  requested, and only for Seattle single-family parcels. Owner and taxpayer fields are never requested.
- Neither file is committed.

## US Census Bureau

The ZCTA Gazetteer file and the geocoding service are US government works in the public domain.

## OpenStreetMap Nominatim

Used at run time to turn a typed address into a position. Data (c) OpenStreetMap contributors, Open
Database License. The usage policy asks for a clear User-Agent and at most one request per second.
The backend sends an identifying User-Agent and waits one second between requests. Results are not
stored.

## Mapillary photos

- Mapillary states that all images on its platform are shared under CC BY-SA. HomeLens records and
  shows, for each photo: provider, photo id, contributor name, capture date and license. The side
  panel links back to the photo on Mapillary.
- Photos are downloaded to `data/images/` for training and are not committed.
- The API token is free, stays in `.env` on the server, and is sent in a request header, never in a URL.
- Open question: whether a model trained on CC BY-SA photos is itself a "share-alike" derivative is
  not settled. The trained models are not distributed, which avoids the question for this project.

## Model weights

The photo encoder is torchvision's ResNet18 with `IMAGENET1K_V1` weights, used unchanged as a
feature extractor. torchvision is BSD-3-Clause. The weights were trained on ImageNet, whose images
are provided for non-commercial research and educational use. That fits this coursework. A
commercial version would need an encoder with clearly commercial terms.

## What is in this repository

Source code, documentation, the furniture assumption file, and synthetic fixtures generated by this
project. No downloaded data, no photos, no trained models, no secrets.
