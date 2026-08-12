#!/usr/bin/env python3
"""Recompute JURIX reviewed-award audit metrics from raw result JSON files."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scipy import stats


WORK_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = WORK_DIR.parent
RESULT_DIR = PACKAGE_DIR / "raw_api_outputs"
TABLE_DIR = PACKAGE_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)

W_MAINTAIN = 481 / 542
W_MODIFIED = 61 / 542
FALSE_CONDS = ["真扰动x1/3", "真扰动x2/3", "真扰动x3/2", "真扰动x3"]
TOLERANCES = [0.01, 0.05, 0.10]

MODELS = [
    ("Gemini-2.5", "anchor_all_gemini25flash.json", "blind2_gemini25flash.json", None, "non-copier"),
    ("GLM-5.1", "anchor_all_glm51.json", "blind2_glm51.json", None, "non-copier"),
    ("DeepSeek-v4-pro", "anchor_all_dsv4pro.json", "blind2_dsv4pro.json", "anchor_zero_dsv4pro.json", "partial"),
    ("Qwen3.7-plus", "anchor_all_qwen37plus.json", "blind2_qwen37plus.json", "anchor_zero_qwen37plus.json", "copier"),
    ("MiMo", "anchor_all_mimo25pro.json", "blind2_mimo25pro_tf.json", "anchor_zero_mimo25pro.json", "copier"),
    ("Grok-4.3", "anchor_all_grok43.json", "blind2_grok43.json", "anchor_zero_grok43.json", "copier"),
    ("DeepSeek-Flash", "anchor_all_flash.json", "blind2_flash_tf.json", None, "copier"),
]


def num(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def within(out: object, target: object, tol: float = 0.05) -> bool:
    output = num(out)
    target_num = num(target)
    if output is None or target_num is None:
        return False
    if target_num == 0:
        return output == 0
    return abs(output - target_num) <= tol * abs(target_num)


def raw_pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def cal_pct(affirm_num: int, affirm_den: int, modified_num: int, modified_den: int) -> float:
    affirm_rate = affirm_num / affirm_den if affirm_den else 0.0
    modified_rate = modified_num / modified_den if modified_den else 0.0
    return 100.0 * (W_MAINTAIN * affirm_rate + W_MODIFIED * modified_rate)


def abs_log_ratio(out: object, target: object) -> float | None:
    output = num(out)
    target_num = num(target)
    if output is None or target_num is None or output <= 0 or target_num <= 0:
        return None
    return abs(math.log(output / target_num))


def median(values: list[float | None]) -> float:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def load_truth() -> dict[str, dict[str, object]]:
    rows = json.loads((PACKAGE_DIR / "data" / "truth_165_cases.json").read_text(encoding="utf-8"))
    return {
        row["CLI"]: {
            "first": float(row["一审金额"]),
            "second": float(row["二审金额"]),
            "label": row["维持改判"],
        }
        for row in rows
    }


def load_anchor(filename: str, zero_filename: str | None = None) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows = json.loads((RESULT_DIR / filename).read_text(encoding="utf-8"))
    x1 = {row["a"]: num(row.get("out")) for row in rows if row.get("cond") == "真锚x1"}
    if zero_filename:
        zero_rows = json.loads((RESULT_DIR / zero_filename).read_text(encoding="utf-8"))
        for row in zero_rows:
            if str(row.get("cond", "")).startswith("真锚x1"):
                x1[row["a"]] = num(row.get("out"))
    return {case_id: value for case_id, value in x1.items() if value is not None}, rows


def load_blind(filename: str) -> dict[str, float]:
    rows = json.loads((RESULT_DIR / filename).read_text(encoding="utf-8"))
    return {row["a"]: value for row in rows if (value := num(row.get("out"))) is not None}


def main() -> None:
    truth = load_truth()
    all_cases = list(truth)
    all_affirm_cases = [case_id for case_id in all_cases if truth[case_id]["label"] == "维持"]
    all_modified_cases = [case_id for case_id in all_cases if truth[case_id]["label"] == "改判"]
    positive_cases = [case_id for case_id, row in truth.items() if row["first"] > 0]
    affirm_cases = [case_id for case_id in positive_cases if truth[case_id]["label"] == "维持"]
    modified_cases = [case_id for case_id in positive_cases if truth[case_id]["label"] == "改判"]

    summary_rows: list[dict[str, object]] = []
    discrim_rows: list[dict[str, object]] = []
    reversal_rows: list[dict[str, object]] = []
    tolerance_rows: list[dict[str, object]] = []
    false_anchor_rows: list[dict[str, object]] = []
    false_anchor_axis_rows: list[dict[str, object]] = []
    typology_rows: list[dict[str, object]] = []
    continuous_rows: list[dict[str, object]] = []

    true_counts = []
    for tol in TOLERANCES:
        true_counts.append(sum(within(truth[c]["second"], truth[c]["first"], tol) for c in all_cases))
    tolerance_rows.append(
        {
            "source_model": "True awards",
            "n": len(all_cases),
            "pct1_count": true_counts[0],
            "pct5_count": true_counts[1],
            "pct10_count": true_counts[2],
            "delta_1_to_5": true_counts[1] - true_counts[0],
            "delta_5_to_10": true_counts[2] - true_counts[1],
        }
    )

    for model, anchor_file, blind_file, zero_anchor_file, model_type in MODELS:
        x1, anchor_rows = load_anchor(anchor_file, zero_anchor_file)
        blind = load_blind(blind_file)

        blind_aff = sum(within(blind.get(c), truth[c]["second"]) for c in all_affirm_cases)
        blind_mod = sum(within(blind.get(c), truth[c]["second"]) for c in all_modified_cases)
        anchor_aff = sum(within(x1.get(c), truth[c]["second"]) for c in all_affirm_cases)
        anchor_mod = sum(within(x1.get(c), truth[c]["second"]) for c in all_modified_cases)
        maintain_aff = sum(within(x1.get(c), truth[c]["first"]) for c in all_affirm_cases)
        maintain_mod = sum(within(x1.get(c), truth[c]["first"]) for c in all_modified_cases)
        blind_aff_pos = sum(within(blind.get(c), truth[c]["second"]) for c in affirm_cases)
        blind_mod_pos = sum(within(blind.get(c), truth[c]["second"]) for c in modified_cases)
        anchor_aff_pos = sum(within(x1.get(c), truth[c]["second"]) for c in affirm_cases)
        anchor_mod_pos = sum(within(x1.get(c), truth[c]["second"]) for c in modified_cases)

        binary_num = maintain_aff + (len(all_modified_cases) - maintain_mod)
        binary_cal = 100.0 * (
            W_MAINTAIN * maintain_aff / len(all_affirm_cases)
            + W_MODIFIED * (len(all_modified_cases) - maintain_mod) / len(all_modified_cases)
        )

        copy_aff = copy_mod = copy_aff_den = copy_mod_den = 0
        escape_aff = escape_mod = 0
        pseudo_abslog_to_anchor: list[float | None] = []
        pseudo_abslog_to_truth: list[float | None] = []
        axis_counts = {
            cond: {
                "copy_aff": 0,
                "copy_mod": 0,
                "escape_aff": 0,
                "escape_mod": 0,
                "aff_den": 0,
                "mod_den": 0,
            }
            for cond in FALSE_CONDS
        }
        for row in anchor_rows:
            if row.get("cond") not in FALSE_CONDS:
                continue
            cond = str(row.get("cond"))
            case_id = row["a"]
            if case_id not in positive_cases:
                continue
            anchor = num(row.get("锚"))
            if anchor is None or anchor <= 0:
                continue
            pseudo_abslog_to_anchor.append(abs_log_ratio(row.get("out"), anchor))
            pseudo_abslog_to_truth.append(abs_log_ratio(row.get("out"), truth[case_id]["second"]))
            is_copy = within(row.get("out"), anchor)
            is_escape_hit = (not is_copy) and within(row.get("out"), truth[case_id]["second"])
            if truth[case_id]["label"] == "维持":
                copy_aff_den += 1
                copy_aff += int(is_copy)
                escape_aff += int(is_escape_hit)
                axis_counts[cond]["aff_den"] += 1
                axis_counts[cond]["copy_aff"] += int(is_copy)
                axis_counts[cond]["escape_aff"] += int(is_escape_hit)
            else:
                copy_mod_den += 1
                copy_mod += int(is_copy)
                escape_mod += int(is_escape_hit)
                axis_counts[cond]["mod_den"] += 1
                axis_counts[cond]["copy_mod"] += int(is_copy)
                axis_counts[cond]["escape_mod"] += int(is_escape_hit)

        copy_cal = cal_pct(copy_aff, copy_aff_den, copy_mod, copy_mod_den)
        escape_cal = cal_pct(escape_aff, copy_aff_den, escape_mod, copy_mod_den)
        axis_row: dict[str, object] = {"model": model}
        for cond, label in [
            ("真扰动x1/3", "x1_3_copy_calib_pct"),
            ("真扰动x2/3", "x2_3_copy_calib_pct"),
            ("真扰动x3/2", "x3_2_copy_calib_pct"),
            ("真扰动x3", "x3_copy_calib_pct"),
        ]:
            counts = axis_counts[cond]
            axis_row[label] = round(
                cal_pct(
                    counts["copy_aff"],
                    counts["aff_den"],
                    counts["copy_mod"],
                    counts["mod_den"],
                ),
                1,
            )
        axis_row["mean_copy_calib_pct"] = round(copy_cal, 1)
        axis_row["mean_escape_and_hit_calib_pct"] = round(escape_cal, 1)
        false_anchor_axis_rows.append(axis_row)

        summary_rows.append(
            {
                "model": model,
                "positive_n": len(all_cases),
                "affirm_n": len(all_affirm_cases),
                "modified_n": len(all_modified_cases),
                "copy_calib_pct": round(copy_cal, 1),
                "blind_hit_n": blind_aff + blind_mod,
                "blind_hit_raw_pct": round(raw_pct(blind_aff + blind_mod, len(all_cases)), 1),
                "blind_hit_calib_pct": round(cal_pct(blind_aff, len(all_affirm_cases), blind_mod, len(all_modified_cases)), 1),
                "true_anchor_hit_n": anchor_aff + anchor_mod,
                "true_anchor_hit_raw_pct": round(raw_pct(anchor_aff + anchor_mod, len(all_cases)), 1),
                "true_anchor_hit_calib_pct": round(cal_pct(anchor_aff, len(all_affirm_cases), anchor_mod, len(all_modified_cases)), 1),
                "binary_correct_n": binary_num,
                "binary_raw_pct": round(raw_pct(binary_num, len(all_cases)), 1),
                "binary_calib_pct": round(binary_cal, 1),
                "type": model_type,
            }
        )

        table = [[maintain_aff, len(all_affirm_cases) - maintain_aff], [maintain_mod, len(all_modified_cases) - maintain_mod]]
        chi2, chi_p, _, _ = stats.chi2_contingency(table, correction=False)
        fisher_p = stats.fisher_exact(table)[1]
        phi = math.sqrt(chi2 / len(all_cases))
        bal_acc = 50.0 * (
            maintain_aff / len(all_affirm_cases) + (len(all_modified_cases) - maintain_mod) / len(all_modified_cases)
        )
        discrim_rows.append(
            {
                "model": model,
                "maintain_true_maintain_n": maintain_aff,
                "true_maintain_n": len(all_affirm_cases),
                "maintain_true_modified_n": maintain_mod,
                "true_modified_n": len(all_modified_cases),
                "gap_pct": round(100.0 * (maintain_aff / len(all_affirm_cases) - maintain_mod / len(all_modified_cases)), 1),
                "chi2_p": round(chi_p, 3),
                "fisher_p": round(fisher_p, 3),
                "phi": round(phi, 3),
                "balanced_accuracy_pct": round(bal_acc, 1),
            }
        )

        stick_hit = stick_miss = depart_hit = depart_miss = 0
        for case_id in all_modified_cases:
            sticks_to_first = within(x1.get(case_id), truth[case_id]["first"])
            hits_second = within(x1.get(case_id), truth[case_id]["second"])
            if sticks_to_first and hits_second:
                stick_hit += 1
            elif sticks_to_first and not hits_second:
                stick_miss += 1
            elif not sticks_to_first and hits_second:
                depart_hit += 1
            else:
                depart_miss += 1

        reversal_rows.append(
            {
                "model": model,
                "modified_positive_n": len(all_modified_cases),
                "stick_n": maintain_mod,
                "stick_pct": round(raw_pct(maintain_mod, len(all_modified_cases)), 1),
                "hit_second_n": anchor_mod,
                "hit_second_pct": round(raw_pct(anchor_mod, len(all_modified_cases)), 1),
                "stick_hit_n": stick_hit,
                "stick_miss_n": stick_miss,
                "depart_hit_n": depart_hit,
                "depart_miss_n": depart_miss,
                "type": model_type,
            }
        )

        counts = []
        for tol in TOLERANCES:
            counts.append(sum(within(x1.get(c), truth[c]["first"], tol) for c in all_cases))
        tolerance_rows.append(
            {
                "source_model": model,
                "n": len(all_cases),
                "pct1_count": counts[0],
                "pct5_count": counts[1],
                "pct10_count": counts[2],
                "delta_1_to_5": counts[1] - counts[0],
                "delta_5_to_10": counts[2] - counts[1],
            }
        )

        false_anchor_rows.append(
            {
                "model": model,
                "blind_hit_calib_pct": round(cal_pct(blind_aff_pos, len(affirm_cases), blind_mod_pos, len(modified_cases)), 1),
                "true_anchor_hit_calib_pct": round(cal_pct(anchor_aff_pos, len(affirm_cases), anchor_mod_pos, len(modified_cases)), 1),
                "pseudo_anchor_copy_calib_pct": round(copy_cal, 1),
                "escape_and_hit_calib_pct": round(escape_cal, 1),
            }
        )
        blind_cal = cal_pct(blind_aff, len(all_affirm_cases), blind_mod, len(all_modified_cases))
        if blind_cal >= 10 and bal_acc >= 60 and fisher_p < 0.05 and phi >= 0.20:
            ability_profile = "demonstrated"
        elif bal_acc >= 55 or blind_cal >= 8:
            ability_profile = "weak signal only"
        else:
            ability_profile = "not shown"
        if copy_cal >= 60 and escape_cal <= 3:
            copying_profile = "strong copier"
        elif copy_cal >= 40:
            copying_profile = "partial copier"
        else:
            copying_profile = "non-copier/noisy"
        typology_rows.append(
            {
                "model": model,
                "blind_hit_calib_pct": round(blind_cal, 1),
                "balanced_accuracy_pct": round(bal_acc, 1),
                "chi2_p": round(chi_p, 3),
                "fisher_p": round(fisher_p, 3),
                "phi": round(phi, 3),
                "pseudo_anchor_copy_calib_pct": round(copy_cal, 1),
                "escape_and_hit_calib_pct": round(escape_cal, 1),
                "ability_profile": ability_profile,
                "copying_profile": copying_profile,
            }
        )

        continuous_rows.append(
            {
                "model": model,
                "true_anchor_abslog_to_truth_median": round(
                    median([abs_log_ratio(x1.get(c), truth[c]["second"]) for c in positive_cases]),
                    3,
                ),
                "modified_true_anchor_abslog_to_truth_median": round(
                    median([abs_log_ratio(x1.get(c), truth[c]["second"]) for c in modified_cases]),
                    3,
                ),
                "modified_true_anchor_abslog_to_first_median": round(
                    median([abs_log_ratio(x1.get(c), truth[c]["first"]) for c in modified_cases]),
                    3,
                ),
                "pseudo_anchor_abslog_to_displayed_median": round(median(pseudo_abslog_to_anchor), 3),
                "pseudo_anchor_abslog_to_truth_median": round(median(pseudo_abslog_to_truth), 3),
                "true_anchor_abslog_to_truth_n": sum(
                    abs_log_ratio(x1.get(c), truth[c]["second"]) is not None for c in positive_cases
                ),
                "modified_true_anchor_abslog_to_truth_n": sum(
                    abs_log_ratio(x1.get(c), truth[c]["second"]) is not None for c in modified_cases
                ),
                "modified_true_anchor_abslog_to_first_n": sum(
                    abs_log_ratio(x1.get(c), truth[c]["first"]) is not None for c in modified_cases
                ),
                "pseudo_anchor_abslog_to_displayed_n": sum(
                    value is not None for value in pseudo_abslog_to_anchor
                ),
                "pseudo_anchor_abslog_to_truth_n": sum(
                    value is not None for value in pseudo_abslog_to_truth
                ),
            }
        )

    always_first_hit = sum(within(truth[c]["first"], truth[c]["second"]) for c in all_cases)
    always_first_hit_mod = sum(within(truth[c]["first"], truth[c]["second"]) for c in all_modified_cases)
    summary_rows.append(
        {
            "model": "always-maintain",
            "positive_n": len(all_cases),
            "affirm_n": len(all_affirm_cases),
            "modified_n": len(all_modified_cases),
            "copy_calib_pct": "",
            "blind_hit_n": "",
            "blind_hit_raw_pct": "",
            "blind_hit_calib_pct": "",
            "true_anchor_hit_n": "",
            "true_anchor_hit_raw_pct": "",
            "true_anchor_hit_calib_pct": "",
            "binary_correct_n": len(all_affirm_cases),
            "binary_raw_pct": round(raw_pct(len(all_affirm_cases), len(all_cases)), 1),
            "binary_calib_pct": round(100.0 * W_MAINTAIN, 2),
            "type": "class-prior warning",
        }
    )
    summary_rows.append(
        {
            "model": "always-first",
            "positive_n": len(all_cases),
            "affirm_n": len(all_affirm_cases),
            "modified_n": len(all_modified_cases),
            "copy_calib_pct": "",
            "blind_hit_n": "",
            "blind_hit_raw_pct": "",
            "blind_hit_calib_pct": "",
            "true_anchor_hit_n": always_first_hit,
            "true_anchor_hit_raw_pct": round(raw_pct(always_first_hit, len(all_cases)), 1),
            "true_anchor_hit_calib_pct": round(
                100.0 * (W_MAINTAIN + W_MODIFIED * always_first_hit_mod / len(all_modified_cases)), 2
            ),
            "binary_correct_n": len(all_affirm_cases),
            "binary_raw_pct": round(raw_pct(len(all_affirm_cases), len(all_cases)), 1),
            "binary_calib_pct": round(100.0 * W_MAINTAIN, 2),
            "type": "coupling anti-example",
        }
    )

    outputs = {
        "reviewed_award_summary_positive_frame.csv": summary_rows,
        "reviewed_award_discrimination_positive_frame.csv": discrim_rows,
        "reviewed_award_reversal_positive_frame.csv": reversal_rows,
        "reviewed_award_tolerance_positive_frame.csv": tolerance_rows,
        "reviewed_award_false_anchor_positive_frame.csv": false_anchor_rows,
        "reviewed_award_pseudo_anchor_axes_positive_frame.csv": false_anchor_axis_rows,
        "reviewed_award_audit_typology_positive_frame.csv": typology_rows,
        "reviewed_award_continuous_diagnostics_positive_frame.csv": continuous_rows,
    }
    for filename, rows in outputs.items():
        path = TABLE_DIR / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "stress_set_n": len(truth),
        "stress_set_exact_maintain": sum(row["label"] == "维持" for row in truth.values()),
        "stress_set_exact_modified": sum(row["label"] == "改判" for row in truth.values()),
        "positive_displayed_award_frame_n": len(positive_cases),
        "positive_displayed_award_frame_exact_maintain": len(affirm_cases),
        "positive_displayed_award_frame_exact_modified": len(modified_cases),
        "modified_first_positive": len(modified_cases),
        "modified_second_positive": sum(
            row["label"] == "改判" and row["second"] > 0 for row in truth.values()
        ),
        "modified_both_positive": sum(
            row["label"] == "改判" and row["first"] > 0 and row["second"] > 0 for row in truth.values()
        ),
        "prevalence_standardization_weights": {"award_maintain": W_MAINTAIN, "award_change": W_MODIFIED},
        "legacy_calibration_A_weights": {"affirm": W_MAINTAIN, "modified": W_MODIFIED},
        "tolerance": "relative ±5%; zero target or zero anchor requires exact zero",
    }
    (TABLE_DIR / "reviewed_award_metric_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
