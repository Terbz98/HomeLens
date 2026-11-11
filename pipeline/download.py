"""Download the raw data files listed in config/data_sources.json.

Usage:
    python -m pipeline.download            # download anything missing
    python -m pipeline.download --force    # download everything again

Each file gets a provenance record (source link, date, size, SHA-256 hash,
license note) in data/raw/provenance.json. Nothing here scrapes a website:
every link is a file or public data service that its publisher offers for
download.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from core.paths import CONFIG_DIR, RAW_DIR

USER_AGENT = "Mozilla/5.0 (HomeLens student project; data download)"
PARCEL_PAGE_SIZE = 2000


def load_sources(config_dir: Optional[Path] = None) -> Dict:
    with open((config_dir or CONFIG_DIR) / "data_sources.json", "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_provenance(raw_dir: Path, source_id: str, entry: Dict) -> None:
    path = raw_dir / "provenance.json"
    records = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    records[source_id] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


def provenance_entry(path: Path, url: str, what: str, license_note: str, downloaded: bool) -> Dict:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "url": url,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "retrieved_at_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "downloaded_by_script": downloaded,
        "what": what,
        "license": license_note,
    }


def download_file(url: str, dest: Path, timeout_s: int = 900) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s) as response, open(tmp, "wb") as out:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
    tmp.replace(dest)


def download_sources(raw_dir: Path = RAW_DIR, force: bool = False) -> bool:
    """Download every plain file. Returns False if any download failed."""
    ok = True
    for source in load_sources()["sources"]:
        dest = raw_dir / source["dest"]
        downloaded = False
        if force or not dest.exists():
            print("downloading %s" % source["id"])
            try:
                download_file(source["url"], dest)
                downloaded = True
            except (urllib.error.URLError, OSError) as exc:
                ok = False
                print("  FAILED: %s" % exc)
                print("  Manual fallback: open this link in a browser and save the file as")
                print("  %s" % dest)
                print("  %s" % source["url"])
                continue
        else:
            print("already have %s" % source["id"])
        record_provenance(raw_dir, source["id"], provenance_entry(dest, source["url"], source["what"], source["license"], downloaded))
    return ok


def fetch_parcel_page(query_url: str, where: str, fields: str, offset: int, timeout_s: int = 60) -> Dict:
    params = {
        "where": where,
        "outFields": fields,
        "returnGeometry": "false",
        "orderByFields": "OBJECTID",
        "resultOffset": str(offset),
        "resultRecordCount": str(PARCEL_PAGE_SIZE),
        "f": "json",
    }
    url = query_url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def download_parcels(raw_dir: Path = RAW_DIR, force: bool = False) -> bool:
    """Page through the King County parcel layer and save Seattle single-family parcels."""
    spec = load_sources()["king_county_parcels"]
    dest = raw_dir / spec["dest"]
    if dest.exists() and not force:
        print("already have %s" % spec["id"])
        record_provenance(raw_dir, spec["id"], provenance_entry(dest, spec["query_url"], spec["what"], spec["license"], False))
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = spec["fields"].split(",")
    offset = 0
    rows = 0
    tmp = dest.with_suffix(".part")
    print("downloading %s (about 65 pages)" % spec["id"])
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            while True:
                page = None
                for attempt in range(3):
                    try:
                        page = fetch_parcel_page(spec["query_url"], spec["where"], spec["fields"], offset)
                        break
                    except (urllib.error.URLError, OSError, ValueError):
                        time.sleep(2 + attempt * 3)
                if page is None or "error" in page:
                    raise OSError("parcel service did not answer at offset %d" % offset)
                features = page.get("features", [])
                for feature in features:
                    attrs = feature["attributes"]
                    writer.writerow([attrs.get(name) for name in fields])
                rows += len(features)
                if not features or not page.get("exceededTransferLimit"):
                    break
                offset += len(features)
                if offset % 20000 == 0:
                    print("  %d parcels so far" % offset)
    except OSError as exc:
        print("  FAILED: %s" % exc)
        return False
    tmp.replace(dest)
    print("  saved %d parcels" % rows)
    record_provenance(raw_dir, spec["id"], provenance_entry(dest, spec["query_url"], spec["what"], spec["license"], True))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HomeLens raw data.")
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args()
    ok_files = download_sources(force=args.force)
    ok_parcels = download_parcels(force=args.force)
    if not (ok_files and ok_parcels):
        print("Some downloads failed. See the messages above.")
        return 1
    print("All raw data is in %s" % RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
