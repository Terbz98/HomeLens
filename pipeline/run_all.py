"""Run the whole real-data pipeline in order.

Usage:
    python -m pipeline.run_all

Steps: download raw data, build and validate datasets, train, write the
evaluation report. Photo steps run only when MAPILLARY_TOKEN is set and
are started separately (see README).
"""
import sys

from pipeline import build_dataset, download, evaluate, train


def main() -> int:
    if not (download.download_sources() and download.download_parcels()):
        print("Download step failed. Fix it (see the messages above) and run again.")
        return 1
    code = build_dataset.main()
    if code != 0:
        return code
    train.train_all(fixtures=False)
    sys.argv = [sys.argv[0]]
    return evaluate.main()


if __name__ == "__main__":
    sys.exit(main())
