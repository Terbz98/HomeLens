"""Write a readable evaluation report from the numbers saved by training.

Usage:
    python -m pipeline.evaluate              # reads models/real/manifest.json -> docs/evaluation.md
    python -m pipeline.evaluate --fixtures   # reads models/demo/manifest.json -> prints only

Every number in the report is copied from the manifest. Nothing is typed by hand.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from core.paths import DEMO_MODEL_DIR, REAL_MODEL_DIR, ROOT

CITY_NAMES = {"seattle": "Seattle", "san_francisco": "San Francisco", "portland": "Portland"}
MODEL_NAMES = {
    "baseline_metro_median": "Baseline: metro median",
    "location_knn": "Location only: nearest neighbors",
    "location_gbm": "Location only: gradient boosting",
    "selected": "Selected procedure",
    "baseline_city_median": "Baseline: city median",
    "baseline_zip_median": "Baseline: ZIP median of past sales",
    "baseline_zillow_zip_value_same_month": "Baseline: Zillow ZIP value, same month",
    "location_only": "Location only (chosen model)",
    "location_only_on_those_rows": "Location only (chosen model), same rows as the Zillow baseline",
    "location_plus_photo": "Location plus photo",
}


def money(value) -> str:
    if value is None:
        return "n/a"
    return "${:,.0f}".format(value)


def _label(name: str) -> str:
    if name in MODEL_NAMES:
        return MODEL_NAMES[name]
    if name.startswith("location_") and name.endswith("_not_chosen"):
        return "Location only: %s (not chosen)" % name[len("location_"):-len("_not_chosen")]
    return name


def _table(rows: Dict[str, Dict]) -> List[str]:
    lines = ["| Model | Rows | MAE | RMSE |", "|---|---:|---:|---:|"]
    for name, m in rows.items():
        lines.append("| %s | %d | %s | %s |" % (_label(name), m["n"], money(m["mae"]), money(m["rmse"])))
    return lines


def _comparison(c: Dict) -> str:
    """One sentence: how much the photo changed the MAE and whether that could be chance."""
    change = c["mae_lowered_by_usd"]
    word = "lowered" if change >= 0 else "raised"
    low, high = c["range_95_usd"]
    return ("The photo %s the MAE by %s (%.2f%%). Resampling %s gives a 95%% range for the change in MAE "
            "from %s to %s, so the result is: %s." % (
                word, money(abs(change)), 100 * abs(c["mae_lowered_by_share"]), c["resampled"],
                lower_or_higher(low), lower_or_higher(high), c["verdict"]))


def lower_or_higher(value) -> str:
    return "%s %s" % (money(abs(value)), "lower" if value >= 0 else "higher")


def area_section(title: str, unit: str, report: Dict) -> List[str]:
    lines = ["## %s" % title, ""]
    if report.get("status") != "trained":
        return lines + ["Not trained: %s." % report.get("status", "unknown"), ""]
    lines += [
        "Label: Zillow ZIP level value (%s). One row per ZIP, %d ZIPs across the three metro areas." % (unit, report["n_zips"]),
        "Evaluation: %s." % report["evaluation"],
        "Model fitted on all ZIPs for the app: `%s`." % report["served_model"],
        "",
        "All ZIPs:",
        "",
    ] + _table(report["overall"]) + [""]
    for city, rows in report["by_city"].items():
        lines += ["%s (ZIPs inside the supported city only):" % CITY_NAMES.get(city, city), ""] + _table(rows) + [""]
    lines += [
        "These errors are against a held-out ZIP's typical value. They are not errors against the price of "
        "an individual house, because no per-house label exists for this model.",
        "",
    ]
    return lines


def sale_section(report: Dict) -> List[str]:
    lines = ["## Seattle per-house sale price", ""]
    if report.get("status") != "trained":
        return lines + ["Not trained: %s." % report.get("status", "unknown"), ""]
    s = report["splits"]
    lines += [
        "Label: completed sale price of a single-family house (King County public record). %d sales." % report["n_rows"],
        "",
        "| Split | Rows | From | To |",
        "|---|---:|---|---|",
    ]
    for name in ("train", "val", "cal", "test"):
        lines.append("| %s | %d | %s | %s |" % (name, s[name]["n"], s[name]["from"], s[name]["to"]))
    lines += [
        "",
        "Splits are in date order. %d older rows of houses that sold again later were removed, so no house is in "
        "two splits. Houses that appear in both training and test: %d." % (
            report["repeat_sale_rows_removed_to_keep_houses_in_one_split"], report["houses_in_both_training_and_test"]),
        "",
        "Model chosen on the validation split: `%s`. Validation MAE per candidate: %s." % (
            report["chosen_on_validation"],
            ", ".join("%s %s" % (k, money(v["mae"])) for k, v in report["validation"].items())),
        "",
        "Test split (median sale price %s):" % money(report["test_median_sale_price"]),
        "",
    ] + _table(report["test"]) + [""]
    interval = report["interval"]
    if "relative_half_width" in interval:
        lines += [
            "Prediction range: %s, plus or minus %.1f%% of the estimate. Target coverage %.0f%%, measured coverage on "
            "the test split %.1f%% (%d calibration rows, %d test rows)." % (
                interval["method"], 100 * interval["relative_half_width"], 100 * interval["target_coverage"],
                100 * interval["measured_coverage_on_test"], interval["n_calibration"], interval["n_test"]),
            "",
        ]
    else:
        lines += ["Prediction range: unavailable (%s)." % interval.get("reason", ""), ""]
    lines += [report["served_model_note"], ""]
    return lines


def photo_section(report: Dict) -> List[str]:
    lines = ["## Seattle photo model (location plus exterior photo)", ""]
    status = report.get("status")
    if status != "trained":
        return lines + ["Not trained: %s. %s" % (status, report.get("reason", "")), ""]
    lines += [
        "Encoder: %s, used as a fixed feature extractor." % report["encoder"],
        "Sales of houses with a matched photo per split: %s (all sales: %s)." % (
            ", ".join("%s %d" % kv for kv in report["rows_with_photo"].items()),
            ", ".join("%s %d" % kv for kv in report["rows_total"].items())),
        "Chosen photo model: `%s`." % report["chosen_photo_model"],
        "",
        "Test split, only houses that have a photo:",
        "",
    ] + _table(report["test_rows_with_photo"]) + [
        "",
        "Did the photo help on the test split? %s" % _comparison(report["test_comparison"]),
        "",
        "On the calibration split the photo lowered the MAE by %.2f%%." % (100 * report["calibration_mae_lowered_by_share"]),
        "Is the photo model served by the app? %s (%s)." % (
            "Yes" if report["serve_photo_model"] else "No", report["serve_rule"]),
        "",
    ]
    interval = report["interval"]
    if "relative_half_width" in interval:
        lines += [
            "Prediction range with a photo: plus or minus %.1f%%, measured coverage on test %.1f%% (target %.0f%%)." % (
                100 * interval["relative_half_width"], 100 * interval["measured_coverage_on_test"],
                100 * interval["target_coverage"]),
            "",
        ]
    else:
        lines += ["Prediction range with a photo: unavailable (%s)." % interval.get("reason", ""), ""]
    return lines


def coverage_section(coverage: Dict) -> List[str]:
    lines = ["## Street photo coverage (Mapillary)", "",
             "A place counts as covered when a photo exists that was taken 3 to 40 m away and, if its direction "
             "is known, points within 70 degrees of the place.", "",
             "| What was searched | City | Searched | With a usable photo | Share |", "|---|---|---:|---:|---:|"]
    names = {"house": "Sold houses (random sample)", "area_point": "Random spots near ZIP centers"}
    for kind in ("house", "area_point"):
        for city, c in sorted(coverage.get(kind, {}).items()):
            lines.append("| %s | %s | %d | %d | %.1f%% |" % (
                names[kind], CITY_NAMES.get(city, city), c["searched"], c["with_photo"], 100 * c["share"]))
    return lines + [""]


def area_photo_section(report: Dict) -> List[str]:
    lines = ["## Area photo experiment (San Francisco and Portland)", ""]
    status = report.get("status")
    if status != "trained":
        return lines + ["Not run: %s. %s" % (status, report.get("reason", "")), ""]
    lines += [report["description"], ""]
    for target, block in report["targets"].items():
        lines += ["%s:" % target, ""] + _table(block["overall"]) + ["", "Did the photo help? %s" % _comparison(block["comparison"]), ""]
        for city, rows in block["by_city"].items():
            lines += ["%s:" % CITY_NAMES.get(city, city), ""] + _table(rows) + [""]
    return lines


def render(manifest: Dict) -> str:
    metrics = manifest["metrics"]
    period = manifest["data_period"]
    lines = [
        "# HomeLens evaluation report",
        "",
        "Generated by `python -m pipeline.evaluate` from `manifest.json`. Do not edit by hand.",
        "",
        "- Model version: `%s`" % manifest["model_version"],
        "- Trained (UTC): %s" % manifest["trained_at_utc"],
        "- Training data: %s" % ("SYNTHETIC fixtures. These numbers say nothing about real houses."
                                 if manifest["synthetic_training_data"] else "real data (Zillow Research, King County)"),
        "- Zillow home value month: %s. Zillow rent month: %s." % (period["area_sale_month"], period["area_rent_month"]),
        "- Seattle sales: %s to %s." % (period["seattle_sales_from"], period["seattle_sales_to"]),
        "",
        "MAE is the mean absolute error and RMSE is the root mean squared error, both in US dollars.",
        "",
    ]
    lines += area_section("Area model: typical single-family home value", "USD", metrics["area_sale"])
    lines += area_section("Area model: monthly rent index, all home types", "USD per month", metrics["area_rent"])
    lines += sale_section(metrics["seattle_sale"])
    if manifest.get("photo_coverage"):
        lines += coverage_section(manifest["photo_coverage"])
    lines += photo_section(metrics["seattle_photo"])
    if "area_photo" in metrics:
        lines += area_photo_section(metrics["area_photo"])
    lines += ["## Single-family rent ratio", "",
              "Zillow publishes a single-family rent index only per metro area. The app multiplies the ZIP level "
              "all-homes rent by this metro ratio.", "",
              "| Metro | Month | Single-family rent | All-homes rent | Ratio |", "|---|---|---:|---:|---:|"]
    for city, r in manifest["metro_rent_ratio"].items():
        lines.append("| %s | %s | %s | %s | %.4f |" % (
            r["metro"], r["month"], money(r["single_family_rent_index"]), money(r["all_homes_rent_index"]), r["ratio"]))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the HomeLens evaluation report.")
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()
    model_dir = DEMO_MODEL_DIR if args.fixtures else REAL_MODEL_DIR
    path = model_dir / "manifest.json"
    if not path.exists():
        print("No manifest at %s. Train first." % path)
        return 1
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    text = render(manifest)
    if args.fixtures:
        print(text)
        return 0
    out = ROOT / "docs" / "evaluation.md"
    out.write_text(text, encoding="utf-8")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
