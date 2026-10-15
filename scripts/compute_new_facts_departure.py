#!/usr/bin/env python3
"""Camera-ready addition (2026-10-15, R3/R1): do departures track newly found appellate facts?

Courts: exact award-change label vs whether the prompt carries newly found appellate facts.
Models: true-anchor departure (output outside +/-5% of the displayed first-instance award,
exact zero when that award is zero) vs the same flag. 165-case frame, corrected outputs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scipy import stats

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "tables" / "new_facts_departure_165.csv"
MODELS = [
    ("Gemini-2.5", "anchor_all_gemini25flash.json", None),
    ("GLM-5.1", "anchor_all_glm51.json", None),
    ("DeepSeek-v4-pro", "anchor_all_dsv4pro.json", "anchor_zero_dsv4pro.json"),
    ("Qwen3.7-plus", "anchor_all_qwen37plus.json", "anchor_zero_qwen37plus.json"),
    ("MiMo", "anchor_all_mimo25pro.json", "anchor_zero_mimo25pro.json"),
    ("Grok-4.3", "anchor_all_grok43.json", "anchor_zero_grok43.json"),
    ("DeepSeek-Flash", "anchor_all_flash.json", None),
]


def num(v: object) -> float | None:
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def within(out: object, target: float, tol: float = 0.05) -> bool:
    o = num(out)
    if o is None:
        return False
    return o == 0 if target == 0 else abs(o - target) <= tol * abs(target)


def table(flag: dict[str, bool], event: dict[str, bool]) -> list[list[int]]:
    t = [[0, 0], [0, 0]]
    for cli, f in flag.items():
        t[int(f)][int(event[cli])] += 1
    return t


def summarise(name: str, t: list[list[int]]) -> dict[str, object]:
    (a, b), (c, d) = t
    n = a + b + c + d
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / den if den else float("nan")
    return {
        "source": name,
        "n": n,
        "rate_no_new_pct": round(100 * b / (a + b), 1),
        "rate_new_pct": round(100 * d / (c + d), 1),
        "phi": round(phi, 3),
        "fisher_p": round(stats.fisher_exact(t)[1], 4),
        "table_[[noNew_keep,noNew_change],[new_keep,new_change]]": json.dumps(t),
    }


def main() -> None:
    truth = {r["CLI"]: r for r in json.loads((PKG / "data/truth_165_cases.json").read_text(encoding="utf-8"))}
    flags = csv.DictReader((PKG / "data/new_appellate_facts_165.csv").open(encoding="utf-8"))
    new = {r["case_ref"]: r["new_appellate_facts"] == "True" for r in flags}
    changed = {cli: truth[cli]["维持改判"] != "维持" for cli in new}

    rows = [summarise("Courts", table(new, changed))]
    for model, anchor_file, zero_file in MODELS:
        recs = json.loads((PKG / "raw_api_outputs" / anchor_file).read_text(encoding="utf-8"))
        x1 = {r["a"]: r.get("out") for r in recs if r.get("cond") == "真锚x1"}
        if zero_file:
            for r in json.loads((PKG / "raw_api_outputs" / zero_file).read_text(encoding="utf-8")):
                if str(r.get("cond", "")).startswith("真锚x1"):
                    x1[r["a"]] = r.get("out")
        avail = {cli: f for cli, f in new.items() if cli in x1}
        depart = {cli: num(x1[cli]) is not None and not within(x1[cli], truth[cli]["一审金额"]) for cli in avail}
        row = summarise(model, table(avail, depart))
        cw = [cli for cli in avail if changed[cli] and new[cli]]
        row["changed_with_new_facts_depart_and_hit"] = (
            f"{sum(depart[c] and within(x1[c], truth[c]['二审金额']) for c in cw)}/{len(cw)}"
        )
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
