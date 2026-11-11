"""Checks on the project's written material.

1. Project prose never uses an em dash.
2. Dollar figures quoted in the model card and the proposal script are found in the generated
   evaluation report (or come from the furniture assumptions), so the prose cannot drift from
   the measured numbers.
"""
import re
import subprocess

import pytest

from core import furniture
from core.paths import ROOT

EM_DASH = chr(8212)   # written as a number so this file does not contain one
TEXT_SUFFIXES = {".md", ".py", ".js", ".json", ".html", ".css", ".txt", ".ini", ".example", ".csv"}
DOLLARS = re.compile(r"\$\d{1,3}(?:,\d{3})*")


def project_text_files():
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    return [ROOT / name for name in listed if (ROOT / name).suffix in TEXT_SUFFIXES and (ROOT / name).is_file()]


def test_no_em_dash_anywhere_in_the_project():
    offenders = [str(path.relative_to(ROOT)) for path in project_text_files()
                 if EM_DASH in path.read_text(encoding="utf-8", errors="ignore")]
    assert offenders == []


def allowed_figures():
    report = (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
    figures = set(DOLLARS.findall(report))
    package = furniture.load_package()
    for amount in (furniture.new_cost_usd(package), furniture.resale_value_usd(package),
                   furniture.monthly_amount_usd(package)):
        figures.add("${:,.0f}".format(amount))
    # Limits written in the validation rules, not measured results.
    figures.update({"$50,000", "$30,000,000", "$300"})
    return figures


@pytest.mark.parametrize("name", ["model_card.md", "proposal_script.md"])
def test_dollar_figures_in_docs_come_from_the_evaluation_report(name):
    if not (ROOT / "docs" / "evaluation.md").exists():
        pytest.skip("no evaluation report yet")
    text = (ROOT / "docs" / name).read_text(encoding="utf-8")
    quoted = set(DOLLARS.findall(text))
    unknown = sorted(quoted - allowed_figures())
    assert unknown == [], "figures not found in docs/evaluation.md: %s" % unknown


def test_counts_in_docs_match_the_evaluation_report():
    if not (ROOT / "docs" / "evaluation.md").exists():
        pytest.skip("no evaluation report yet")
    report = (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
    sales = re.search(r"(\d+) sales\.", report).group(1)
    test_rows = re.search(r"\| test \| (\d+) \|", report).group(1)
    for name in ("model_card.md", "proposal_script.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8").replace(",", "")
        assert sales in text, "%s does not mention the %s sales used for training" % (name, sales)
        assert test_rows in text, "%s does not mention the %s test sales" % (name, test_rows)
