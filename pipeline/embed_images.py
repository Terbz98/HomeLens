"""Turn downloaded photos into embeddings (512 numbers each) with a fixed pretrained encoder.

Usage:
    python -m pipeline.embed_images

Reads data/processed/image_matches.csv and area_image_points.csv, embeds
every photo not embedded yet, and saves data/processed/image_embeddings.npz
with two arrays: image_ref (photo ids) and embedding (one row per photo).
"""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.embed import EMBEDDING_SIZE, embed_images
from core.paths import DATA_DIR, PROCESSED_DIR

BATCH = 32


def embed_files(files: Dict[str, Path], out_path: Path) -> int:
    """Embed {image_ref: file path}. Already embedded refs in out_path are kept. Returns how many were added."""
    refs: List[str] = []
    rows = np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
    if out_path.exists():
        old = np.load(out_path, allow_pickle=False)
        refs = [str(r) for r in old["image_ref"]]
        rows = old["embedding"].astype(np.float32)
    known = set(refs)
    todo = [(ref, path) for ref, path in files.items() if ref not in known and path.exists()]
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        good_refs, data = [], []
        for ref, path in chunk:
            good_refs.append(ref)
            data.append(path.read_bytes())
        try:
            emb = embed_images(data)
        except Exception:
            # One unreadable file should not lose the batch: do them one by one.
            emb_rows, kept = [], []
            for ref, blob in zip(good_refs, data):
                try:
                    emb_rows.append(embed_images([blob])[0])
                    kept.append(ref)
                except Exception:
                    print("  skipped unreadable photo %s" % ref)
            good_refs = kept
            emb = np.array(emb_rows, dtype=np.float32).reshape(-1, EMBEDDING_SIZE)
        refs += good_refs
        rows = np.vstack([rows, emb])
        if (start // BATCH) % 10 == 0:
            print("  %d / %d photos embedded" % (min(start + BATCH, len(todo)), len(todo)))
    np.savez_compressed(out_path, image_ref=np.array(refs, dtype=str), embedding=rows.astype(np.float32))
    return len(todo)


def main() -> int:
    files: Dict[str, Path] = {}
    for name in ("image_matches.csv", "area_image_points.csv"):
        path = PROCESSED_DIR / name
        if path.exists():
            table = pd.read_csv(path, dtype={"image_ref": str})
            for row in table.itertuples():
                files[str(row.image_ref)] = DATA_DIR / "images" / str(row.file)
    if not files:
        print("No photos found. Run python -m pipeline.fetch_images first.")
        return 1
    added = embed_files(files, PROCESSED_DIR / "image_embeddings.npz")
    print("%d photos embedded, %d total listed" % (added, len(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
