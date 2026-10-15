#!/usr/bin/env python3
"""Section 4 human-side target coupling on the 542 paired appeals.

Reads data/coupling_542_awards.csv and writes tables/human_coupling_542.csv:
maintain/change counts, and for changed awards the second/first ratio summary
and a descriptive log-log regression of the second-instance award on the first.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "coupling_542_awards.csv"
OUT = ROOT / "tables" / "human_coupling_542.csv"


def main() -> None:
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    changed = [r for r in rows if r["exact_award_label"] == "change"]
    first = np.array([float(r["first_instance_award"]) for r in changed])
    second = np.array([float(r["second_instance_award"]) for r in changed])

    nz = first > 0
    ratio = second[nz] / first[nz]
    q1, q3 = np.percentile(ratio, [25, 75])
    both = nz & (second > 0)
    fit = stats.linregress(np.log(first[both]), np.log(second[both]))

    result = {
        "n_cases": len(rows),
        "n_maintain": len(rows) - len(changed),
        "maintain_pct": round(100 * (len(rows) - len(changed)) / len(rows), 2),
        "n_changed": len(changed),
        "n_changed_in_model_set": sum(r["in_model_set"] == "True" for r in changed),
        "n_changed_nonzero_first": int(nz.sum()),
        "median_second_over_first": round(float(np.median(ratio)), 3),
        "iqr_low": round(float(q1), 3),
        "iqr_high": round(float(q3), 3),
        "n_within_half_to_double": int(((ratio >= 0.5) & (ratio <= 2)).sum()),
        "n_nonzero_both": int(both.sum()),
        "loglog_slope": round(fit.slope, 3),
        "loglog_slope_se": round(fit.stderr, 3),
        "loglog_r2": round(fit.rvalue**2, 3),
        "log_correlation_r": round(fit.rvalue, 3),
    }
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result))
        w.writeheader()
        w.writerow(result)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
