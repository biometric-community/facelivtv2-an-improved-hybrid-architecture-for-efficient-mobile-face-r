"""REPORT.md, the committed output of the full run, against Table 4."""

from __future__ import annotations

import re

import pytest


def accuracy(report: str, benchmark: str) -> float:
    match = re.search(rf"\| {benchmark} \| ([\d.]+)% \|", report)
    assert match, f"{benchmark} row missing from REPORT.md"
    return float(match.group(1))


def test_reported_parameter_count_is_the_deployed_backbone(report, targets):
    """Tables 3-4 count the recognition model without the training-only CosFace head."""
    match = re.search(r"\*\*Params:\*\* (\d+)", report)
    variant = re.search(r"\*\*Variant:\*\* `(\w+)`", report)
    assert match and variant
    assert int(match.group(1)) / 1e6 == pytest.approx(
        targets["params_m"][variant.group(1)], abs=targets["params_tolerance_m"]
    )


@pytest.mark.parametrize("benchmark", ["LFW", "CALFW", "CPLFW"])
def test_published_values(report, targets, benchmark):
    expected = targets["published_values"][benchmark]
    assert accuracy(report, benchmark) == pytest.approx(expected, abs=targets["tolerance_pp"])
