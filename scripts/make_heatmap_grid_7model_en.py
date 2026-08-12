#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a seven-model version of the anchor/output heatmap figure.

This extends the four-model heatmap used in the English draft. The figure keeps
the same visual grammar: x-axis is shown anchor / output on log-spaced bins;
red box is the copy cell; green triangle marks the true-first-instance landing
point, output = real first instance, i.e. anchor/output = k.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = WORK_DIR.parent
FIG_DIR = PACKAGE_DIR / "figs"
TABLE_DIR = PACKAGE_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(FIG_DIR / ".mplconfig"))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.patheffects as pe


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

RESULT_DIR = PACKAGE_DIR / "raw_api_outputs"
TRUTH_PATH = PACKAGE_DIR / "data" / "truth_165_cases.json"

CONDS = [
    ("真扰动x1/3", 1 / 3, "×1/3"),
    ("真扰动x2/3", 2 / 3, "×2/3"),
    ("真锚x1", 1.0, "×1"),
    ("真扰动x3/2", 1.5, "×3/2"),
    ("真扰动x3", 3.0, "×3"),
]

MODEL_GRID = [
    [
        ("anchor_all_flash.json", "DeepSeek-Flash", "copier"),
        ("anchor_all_grok43.json", "Grok-4.3", "copier"),
        ("anchor_all_mimo25pro.json", "MiMo", "copier"),
        ("anchor_all_qwen37plus.json", "Qwen3.7-plus", "copier"),
    ],
    [
        ("anchor_all_dsv4pro.json", "DeepSeek-v4-pro", "partial"),
        ("anchor_all_glm51.json", "GLM-5.1", "non-copier"),
        ("anchor_all_gemini25flash.json", "Gemini-2.5", "non-copier"),
        None,
    ],
]

INNER = np.linspace(-1, 1, 20)
EDGES = np.concatenate([[-np.inf], INNER, [np.inf]])
XCOORD = np.concatenate([[INNER[0] - 0.22], INNER, [INNER[-1] + 0.22]])
N_COL = len(EDGES) - 1
COPY_CELL_LEFT = INNER[9]
COPY_CELL_RIGHT = INNER[10]
GREEN = "#2E9E44"
RED = "#B64342"
NEUTRAL = "#272727"


def load_rows(filename: str) -> list[dict[str, object]]:
    with (RESULT_DIR / filename).open(encoding="utf-8") as fh:
        return json.load(fh)


def load_positive_case_ids() -> set[str]:
    rows = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
    return {row["CLI"] for row in rows if num(row.get("一审金额")) and num(row.get("一审金额")) > 0}


def num(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def matrix(filename: str, positive_case_ids: set[str]) -> tuple[np.ndarray, dict[str, int]]:
    rows_by_cond: dict[str, list[dict[str, object]]] = {cond: [] for cond, _, _ in CONDS}
    for row in load_rows(filename):
        cond = row.get("cond")
        if cond in rows_by_cond and row.get("a") in positive_case_ids:
            rows_by_cond[cond].append(row)

    mat = np.zeros((len(CONDS), N_COL))
    n_by_cond: dict[str, int] = {}
    for i, (cond, _k, _label) in enumerate(CONDS):
        values: list[float] = []
        for row in rows_by_cond[cond]:
            output = num(row.get("out"))
            anchor = num(row.get("锚"))
            if output is None or anchor is None or anchor <= 0:
                continue
            values.append(np.log10(anchor / max(output, 1e-3)))
        n_by_cond[cond] = len(values)
        mat[i], _ = np.histogram(values, bins=EDGES)
    return mat, n_by_cond


def write_counts_csv(path: Path, matrices: dict[str, np.ndarray]) -> None:
    x_centers = (XCOORD[:-1] + XCOORD[1:]) / 2
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["model", "axis", "bin_index", "bin_center_log10_anchor_over_output", "count"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for model, mat in matrices.items():
            for i, (cond, _k, _label) in enumerate(CONDS):
                for j in range(N_COL):
                    writer.writerow(
                        {
                            "model": model,
                            "axis": cond,
                            "bin_index": j,
                            "bin_center_log10_anchor_over_output": f"{x_centers[j]:.6f}",
                            "count": int(mat[i, j]),
                        }
                    )


def main() -> None:
    positive_case_ids = load_positive_case_ids()
    matrices: dict[str, np.ndarray] = {}
    counts_by_model: dict[str, dict[str, int]] = {}
    for row in MODEL_GRID:
        for item in row:
            if item is None:
                continue
            filename, model, _camp = item
            matrices[model], counts_by_model[model] = matrix(filename, positive_case_ids)

    cmap = LinearSegmentedColormap.from_list(
        "grid",
        ["#f7fbfd", "#d9ecf2", "#a6cee3", "#5fa8d3", "#0f4d92", "#16324a"],
    )
    # Log scaling keeps sparse non-zero cells visible while compressing the dominant copy column.
    vmax = max(150.0, max(float(mat.max()) for mat in matrices.values()))
    norm = LogNorm(vmin=1, vmax=vmax)
    threshold = 3
    x_centers = (XCOORD[:-1] + XCOORD[1:]) / 2
    zero_cmap = LinearSegmentedColormap.from_list("zero_grid", ["#f2f8fb", "#f2f8fb"])

    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.75), sharex=False, sharey=False)
    fig.subplots_adjust(left=0.060, right=0.990, top=0.895, bottom=0.135, wspace=0.145, hspace=0.320)
    mesh = None
    for ri, row in enumerate(MODEL_GRID):
        for ci, item in enumerate(row):
            ax = axes[ri][ci]
            if item is None:
                ax.axis("off")
                continue
            _filename, model, camp = item
            mat = matrices[model]
            ax.pcolormesh(
                XCOORD,
                np.arange(len(CONDS) + 1),
                np.zeros_like(mat),
                cmap=zero_cmap,
                vmin=0,
                vmax=1,
                edgecolors="#ffffff",
                linewidth=0.28,
            )
            mesh = ax.pcolormesh(
                XCOORD,
                np.arange(len(CONDS) + 1),
                np.ma.masked_where(mat <= 0, mat),
                cmap=cmap,
                norm=norm,
                edgecolors="#ffffff",
                linewidth=0.28,
            )
            copy_box = plt.Rectangle(
                (COPY_CELL_LEFT - 0.006, -0.015),
                COPY_CELL_RIGHT - COPY_CELL_LEFT + 0.012,
                len(CONDS) + 0.03,
                fill=False,
                edgecolor=RED,
                lw=0.85,
                zorder=4,
                clip_on=False,
            )
            ax.add_patch(copy_box)
            for i, (_cond, k, _label) in enumerate(CONDS):
                for j in range(N_COL):
                    value = int(mat[i, j])
                    if value >= threshold:
                        text_color = "white" if norm(value) > 0.68 else "#20252b"
                        stroke_color = "#20252b" if text_color == "white" else "white"
                        ax.text(
                            x_centers[j],
                            i + 0.5,
                            str(value),
                            ha="center",
                            va="center",
                            fontsize=4.6,
                            color=text_color,
                            zorder=9,
                            path_effects=[pe.withStroke(linewidth=0.55, foreground=stroke_color)],
                        )
                true_first_x = np.log10(k)
                ax.plot([true_first_x, true_first_x], [i, i + 1], color=GREEN, lw=0.85, ls=(0, (2, 1.4)), zorder=6)
                ax.plot(
                    [true_first_x],
                    [i + 1],
                    marker="^",
                    color=GREEN,
                    ms=3.8,
                    mec="white",
                    mew=0.35,
                    zorder=8,
                    clip_on=False,
                )

            ax.set_xlim(XCOORD[0], XCOORD[-1])
            ax.set_ylim(len(CONDS), 0)
            ax.set_yticks(np.arange(len(CONDS)) + 0.5)
            if ci == 0:
                ax.set_yticklabels([label for _cond, _k, label in CONDS], fontsize=6.4, fontweight="bold")
            else:
                ax.set_yticklabels([])
            tick_values = [0.1, 1 / 3, 1, 3, 10]
            ax.set_xticks([np.log10(value) for value in tick_values])
            ax.set_xticklabels(["0.1", "1/3", "1", "3", "10"], fontsize=6.1)
            n_min = min(counts_by_model[model].values())
            n_max = max(counts_by_model[model].values())
            n_text = f"n={n_min}" if n_min == n_max else f"n={n_min}-{n_max}"
            ax.set_title(f"{model}\n{camp}, {n_text}", fontsize=7.1, pad=4, fontweight="bold", color=NEUTRAL)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ri == 1:
                ax.set_xlabel("displayed anchor / output", fontsize=6.5, labelpad=2)

    legend_ax = axes[1][3]
    legend_ax.set_axis_off()

    legend_ax.text(
        0.50,
        1.060,
        "Guide",
        transform=legend_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.1,
        fontweight="bold",
        color=NEUTRAL,
    )

    group_left = 0.10
    group_width = 0.80
    marker_left = group_left - 0.012
    icon_slot_w = 0.075
    text_x = marker_left + icon_slot_w + 0.025
    icon_center_x = marker_left + icon_slot_w / 2

    colorbar_ax = legend_ax.inset_axes([group_left, 0.705, group_width, 0.075])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    cbar.set_ticks([1, 2, 5, 10, 25, 50, 100])
    cbar.set_ticklabels(["1", "2", "5", "10", "25", "50", "100"])
    cbar.ax.minorticks_off()
    cbar.ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    cbar.ax.tick_params(labelsize=5.7, width=0.55, length=2.0, pad=1.5)
    cbar.outline.set_linewidth(0.55)
    cbar.ax.set_title("case count", fontsize=6.2, pad=2.5)

    legend_ax.add_patch(
        plt.Rectangle(
            (marker_left + 0.012, 0.430),
            0.052,
            0.120,
            transform=legend_ax.transAxes,
            fill=False,
            edgecolor=RED,
            lw=1.0,
            clip_on=False,
        )
    )
    legend_ax.text(
        text_x,
        0.516,
        "displayed anchor",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.3,
        fontweight="bold",
        color=RED,
    )
    legend_ax.text(
        text_x,
        0.462,
        "anchor/output = 1",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.1,
        color=NEUTRAL,
    )
    legend_ax.plot(
        [icon_center_x, icon_center_x],
        [0.222, 0.356],
        transform=legend_ax.transAxes,
        color=GREEN,
        lw=1.0,
        ls=(0, (2, 1.5)),
        clip_on=False,
    )
    legend_ax.plot(
        [icon_center_x],
        [0.356],
        transform=legend_ax.transAxes,
        marker="^",
        color=GREEN,
        ms=5.0,
        mec="white",
        mew=0.35,
        clip_on=False,
    )
    legend_ax.text(
        text_x,
        0.312,
        "true first-instance landing",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.3,
        fontweight="bold",
        color=GREEN,
    )
    legend_ax.text(
        text_x,
        0.258,
        "anchor/output = k",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.1,
        color=NEUTRAL,
    )

    axes[0][0].annotate(
        "copy-prone models",
        xy=(-0.24, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.3,
        fontweight="bold",
        color=RED,
    )
    axes[1][0].annotate(
        "partial / anchor-resistant",
        xy=(-0.24, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.3,
        fontweight="bold",
        color="#2471A3",
    )
    out_base = FIG_DIR / "fig_heatmap_grid_7model_en"
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=240)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    write_counts_csv(TABLE_DIR / "heatmap_grid_7model_cell_counts.csv", matrices)
    print(f"Wrote {out_base.with_suffix('.png')}")
    print(f"Wrote {out_base.with_suffix('.pdf')}")
    print(f"Wrote {out_base.with_suffix('.svg')}")
    print(f"Wrote {TABLE_DIR / 'heatmap_grid_7model_cell_counts.csv'}")


if __name__ == "__main__":
    main()
