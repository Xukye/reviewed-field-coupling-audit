#!/usr/bin/env python3
"""Claim-ceiling breach and case-clustered CIs (Section 6, Table 3 brackets).

Reads data/claim_totals_149.csv (claim totals extracted from the de-identified
claim text and checked by hand; open-ended claims flagged), data/truth_165_cases.json
and raw_api_outputs/. Writes tables/claim_ceiling_breach_summary.csv and
tables/pseudo_copy_cluster_ci.csv. The breach test uses claim_max and skips
open-ended claims.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
TABLE_DIR = PKG / "tables"

W_M = 481 / 542
W_C = 61 / 542
FALSE_CONDS = ["真扰动x1/3", "真扰动x2/3", "真扰动x3/2", "真扰动x3"]
MODELS = [
    ("Gemini-2.5", "anchor_all_gemini25flash.json", "blind2_gemini25flash.json", None),
    ("GLM-5.1", "anchor_all_glm51.json", "blind2_glm51.json", None),
    ("DeepSeek-v4-pro", "anchor_all_dsv4pro.json", "blind2_dsv4pro.json", "anchor_zero_dsv4pro.json"),
    ("Qwen3.7-plus", "anchor_all_qwen37plus.json", "blind2_qwen37plus.json", "anchor_zero_qwen37plus.json"),
    ("MiMo", "anchor_all_mimo25pro.json", "blind2_mimo25pro_tf.json", "anchor_zero_mimo25pro.json"),
    ("Grok-4.3", "anchor_all_grok43.json", "blind2_grok43.json", "anchor_zero_grok43.json"),
    ("DeepSeek-Flash", "anchor_all_flash.json", "blind2_flash_tf.json", None),
]
N_BOOT = 10000
SEED = 20261015

def num(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def within(out: object, target: object, tol: float = 0.05) -> bool:
    o, t = num(out), num(target)
    if o is None or t is None:
        return False
    if t == 0:
        return o == 0
    return abs(o - t) <= tol * abs(t)


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def boot_ci(stat, groups: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draws[b] = stat([g[rng.integers(0, len(g), len(g))] for g in groups])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    truth = {r["CLI"]: r for r in load_json(PKG / "data/truth_165_cases.json")}

    claims: dict[str, tuple[float, float]] = {}
    open_ended: set[str] = set()
    for r in csv.DictReader((PKG / "data/claim_totals_149.csv").open(encoding="utf-8")):
        claims[r["case_ref"]] = (float(r["claim_final"]), float(r["claim_max"]))
        if r["open_ended"] == "True":
            open_ended.add(r["case_ref"])

    # Breach rows depend only on displayed anchor vs claim, so they are shared across models.
    summary: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for model, anchor_file, blind_file, zero_file in MODELS:
        rows = load_json(PKG / "raw_api_outputs" / anchor_file)
        blind = {r["a"]: num(r.get("out")) for r in load_json(PKG / "raw_api_outputs" / blind_file)}
        x1 = {r["a"]: num(r.get("out")) for r in rows if r.get("cond") == "真锚x1"}

        breach_rows = breach_copy = breach_over = 0
        breach_cases: dict[str, list[tuple[int, int]]] = {}
        axis_breach: dict[str, int] = {k: 0 for k in FALSE_CONDS}
        per_case_copy: dict[str, list[int]] = {}
        for r in rows:
            cond, cli = r.get("cond"), r["a"]
            if cond not in FALSE_CONDS or cli not in claims:
                continue
            anchor, out = num(r.get("锚")), num(r.get("out"))
            if anchor is None or anchor <= 0:
                continue
            copy = int(within(out, anchor))
            per_case_copy.setdefault(cli, []).append(copy)
            if anchor > claims[cli][1] and cli not in open_ended:
                over = int(out is not None and out > claims[cli][1])
                breach_rows += 1
                breach_copy += copy
                breach_over += over
                axis_breach[str(cond)] += 1
                breach_cases.setdefault(cli, []).append((copy, over))

        blind_over = sum(1 for cli in claims if cli not in open_ended and (o := blind.get(cli)) is not None and o > claims[cli][1])
        blind_n = sum(1 for cli in claims if cli not in open_ended and blind.get(cli) is not None)
        x1_over = sum(1 for cli in claims if cli not in open_ended and (o := x1.get(cli)) is not None and o > claims[cli][1])
        x1_n = sum(1 for cli in claims if cli not in open_ended and x1.get(cli) is not None)

        # Clustered CI for copy and over-claim rates on breach rows.
        bc = list(breach_cases.values())
        copy_arr = np.array([[sum(c for c, _ in v), sum(o for _, o in v), len(v)] for v in bc], dtype=float)

        def ratio(groups: list[np.ndarray], col: int) -> float:
            g = groups[0]
            return 100.0 * g[:, col].sum() / g[:, 2].sum()

        copy_lo, copy_hi = boot_ci(lambda g: ratio(g, 0), [copy_arr], rng)
        over_lo, over_hi = boot_ci(lambda g: ratio(g, 1), [copy_arr], rng)

        summary.append({
            "model": model,
            "breach_rows": breach_rows,
            "breach_cases": len(breach_cases),
            "breach_rows_x3/2": axis_breach["真扰动x3/2"],
            "breach_rows_x3": axis_breach["真扰动x3"],
            "breach_rows_x2/3": axis_breach["真扰动x2/3"],
            "breach_rows_x1/3": axis_breach["真扰动x1/3"],
            "copy_on_breach": breach_copy,
            "copy_on_breach_pct": round(100.0 * breach_copy / breach_rows, 1),
            "copy_on_breach_ci95": f"{copy_lo:.1f}-{copy_hi:.1f}",
            "overclaim_on_breach": breach_over,
            "overclaim_on_breach_pct": round(100.0 * breach_over / breach_rows, 1),
            "overclaim_on_breach_ci95": f"{over_lo:.1f}-{over_hi:.1f}",
            "blind_overclaim": f"{blind_over}/{blind_n}",
            "true_anchor_overclaim": f"{x1_over}/{x1_n}",
        })

        # Clustered CI for reweighted pseudo-anchor copy (Table 3).
        groups = []
        for label in ("维持", "改判"):
            arr = np.array(
                [[sum(v), len(v)] for cli, v in per_case_copy.items() if truth[cli]["维持改判"] == label],
                dtype=float,
            )
            groups.append(arr)

        def reweighted(g: list[np.ndarray]) -> float:
            m = g[0][:, 0].sum() / g[0][:, 1].sum()
            c = g[1][:, 0].sum() / g[1][:, 1].sum()
            return 100.0 * (W_M * m + W_C * c)

        point = reweighted(groups)
        lo, hi = boot_ci(reweighted, groups, rng)
        ci_rows.append({
            "model": model,
            "cases_maintain": len(groups[0]),
            "cases_change": len(groups[1]),
            "rows": int(sum(g[:, 1].sum() for g in groups)),
            "reweighted_copy_pct": round(point, 1),
            "cluster_ci95_low": round(lo, 1),
            "cluster_ci95_high": round(hi, 1),
        })

    for name, data in (("claim_ceiling_breach_summary.csv", summary), ("pseudo_copy_cluster_ci.csv", ci_rows)):
        with (TABLE_DIR / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"\n== {name}")
        for row in data:
            print(row)


if __name__ == "__main__":
    main()
