"""Find and download licensed street photos for training. Needs MAPILLARY_TOKEN in .env.

Usage:
    python -m pipeline.fetch_images --houses 3000        # photos of sold Seattle houses
    python -m pipeline.fetch_images --area-points 12     # photos near random spots in each city ZIP
    python -m pipeline.fetch_images --houses 3000 --area-points 12

Photos come from Mapillary (CC BY-SA 4.0). Nothing is taken from Google
Maps or Street View. For every photo the script records where it came from:
provider, photo id, contributor, capture date, license, and how far the
camera was from the house. Photos are saved under data/images, which is
not committed to git.

The script can be stopped and started again. Houses and spots that were
already searched are skipped.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from core.geo import load_cities
from core.imagery import ImageryError, ImageryProvider, get_provider
from core.paths import IMAGES_DIR, PROCESSED_DIR, ROOT

SEED = 0
WORKERS = 4
AREA_POINT_RADIUS_M = 600.0
MATCH_COLUMNS = ["kind", "property_id", "zip", "city_id", "lat", "lon", "status", "image_ref", "image_provider",
                 "image_license", "image_creator", "image_captured_at", "image_distance_m", "view_angle_diff_deg",
                 "is_pano", "image_lat", "image_lon", "file"]


def search_one(provider: ImageryProvider, target: Dict, image_dir: Path) -> Dict:
    """Search for one house or spot. Returns a row for the match log."""
    row = {c: "" for c in MATCH_COLUMNS}
    row.update({k: target[k] for k in ("kind", "property_id", "zip", "city_id", "lat", "lon")})
    try:
        image = provider.find_best(target["lat"], target["lon"])
        if image is None:
            row["status"] = "no_photo"
            return row
        path = image_dir / ("%s.jpg" % image.image_id)
        if not path.exists():
            path.write_bytes(provider.download(image))
    except ImageryError as exc:
        row["status"] = "error: %s" % exc
        return row
    row.update({
        "status": "found", "image_ref": image.image_id, "image_provider": image.provider,
        "image_license": image.license, "image_creator": image.creator or "",
        "image_captured_at": image.captured_at or "", "image_distance_m": image.distance_m,
        "view_angle_diff_deg": "" if image.view_angle_diff_deg is None else image.view_angle_diff_deg,
        "is_pano": image.is_pano, "image_lat": image.lat, "image_lon": image.lon,
        "file": str(path.relative_to(image_dir.parent)),
    })
    return row


def house_targets(sales: pd.DataFrame, count: int) -> List[Dict]:
    """A random sample of sold houses, one target per house."""
    houses = sales.drop_duplicates("property_id")
    houses = houses.sample(n=min(count, len(houses)), random_state=SEED)
    return [{"kind": "house", "property_id": str(r.property_id), "zip": str(r.zip), "city_id": r.city_id,
             "lat": float(r.lat), "lon": float(r.lon)} for r in houses.itertuples()]


def area_targets(zips: pd.DataFrame, per_zip: int) -> List[Dict]:
    """Random spots within AREA_POINT_RADIUS_M of each supported city ZIP center.

    These are spots, not houses. They are only used to measure photo
    coverage and for the area photo experiment.
    """
    rng = np.random.default_rng(SEED)
    out = []
    city_zips = zips[zips["supported_city_id"].fillna("") != ""]
    for r in city_zips.itertuples():
        for i in range(per_zip):
            distance = AREA_POINT_RADIUS_M * np.sqrt(rng.uniform())
            angle = rng.uniform(0, 2 * np.pi)
            dlat = distance * np.cos(angle) / 111320.0
            dlon = distance * np.sin(angle) / (111320.0 * np.cos(np.radians(r.lat)))
            out.append({"kind": "area_point", "property_id": "%s-%02d" % (r.zip, i), "zip": str(r.zip),
                        "city_id": r.supported_city_id, "lat": round(r.lat + dlat, 6), "lon": round(r.lon + dlon, 6)})
    return out


def run(provider: ImageryProvider, targets: List[Dict], log_path: Path, image_dir: Path) -> pd.DataFrame:
    """Search every target not already in the log. Appends to the log as it goes."""
    image_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if log_path.exists():
        old = pd.read_csv(log_path, dtype=str)
        # Errors are retried on the next run. Found and no_photo are final.
        done = set(old.loc[~old["status"].str.startswith("error"), "property_id"])
    todo = [t for t in targets if t["property_id"] not in done]
    print("%d targets, %d already searched, %d to do" % (len(targets), len(targets) - len(todo), len(todo)))
    new_file = not log_path.exists()
    started = time.monotonic()
    with open(log_path, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        writer = csv.DictWriter(f, fieldnames=MATCH_COLUMNS)
        if new_file:
            writer.writeheader()
        for i, row in enumerate(pool.map(lambda t: search_one(provider, t, image_dir), todo), start=1):
            writer.writerow(row)
            if i % 200 == 0:
                f.flush()
                print("  %d / %d searched (%.0f s)" % (i, len(todo), time.monotonic() - started))
    return pd.read_csv(log_path, dtype={"property_id": str, "zip": str, "image_ref": str})


def coverage(log: pd.DataFrame) -> Dict:
    """Share of searched houses and spots that have a usable photo, per city."""
    out: Dict = {}
    final = log[~log["status"].astype(str).str.startswith("error")].drop_duplicates("property_id", keep="last")
    for (kind, city), group in final.groupby(["kind", "city_id"]):
        found = int((group["status"] == "found").sum())
        out.setdefault(kind, {})[city] = {"searched": int(len(group)), "with_photo": found,
                                          "share": round(found / len(group), 4) if len(group) else None}
    return out


def write_match_files(log: pd.DataFrame, processed_dir: Path) -> None:
    """Split the log into the two files the training step reads."""
    found = log[log["status"] == "found"].drop_duplicates("property_id", keep="last")
    found[found["kind"] == "house"].to_csv(processed_dir / "image_matches.csv", index=False)
    found[found["kind"] == "area_point"].to_csv(processed_dir / "area_image_points.csv", index=False)
    with open(processed_dir / "image_coverage.json", "w", encoding="utf-8") as f:
        json.dump(coverage(log), f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download licensed street photos for HomeLens.")
    parser.add_argument("--houses", type=int, default=0, help="how many sold Seattle houses to search")
    parser.add_argument("--area-points", type=int, default=0, help="random spots to search per city ZIP")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    provider = get_provider(os.environ.get("MAPILLARY_TOKEN", ""))
    if not provider.configured:
        print("MAPILLARY_TOKEN is empty. Put your Mapillary client token in the .env file first.")
        return 1
    if args.houses <= 0 and args.area_points <= 0:
        print("Nothing to do. Pass --houses N and/or --area-points N.")
        return 1
    load_cities()
    targets: List[Dict] = []
    if args.houses > 0:
        sales = pd.read_csv(PROCESSED_DIR / "seattle_sales.csv", dtype={"property_id": str, "zip": str})
        targets += house_targets(sales, args.houses)
    if args.area_points > 0:
        zips = pd.read_csv(PROCESSED_DIR / "zip_values.csv", dtype={"zip": str}, keep_default_na=False,
                           na_values={"lat": [""], "lon": [""]})
        targets += area_targets(zips, args.area_points)
    log = run(provider, targets, PROCESSED_DIR / "image_search_log.csv", IMAGES_DIR / "mapillary")
    write_match_files(log, PROCESSED_DIR)
    print(json.dumps(coverage(log), indent=2))
    last = log.drop_duplicates("property_id", keep="last")       # a retried search has more than one row
    errors = int(last["status"].astype(str).str.startswith("error").sum())
    if errors:
        print("%d searches failed. Run the same command again to retry them." % errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
