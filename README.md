# Reviewed-Field Coupling Replication Package

This package contains the materials needed to reproduce the tables and the main heatmap figure for the JURIX 2026 submission:

> When the Reviewed Field Must Be Shown: An Appellate LLM Validity Audit

The package is intentionally limited to extracted case-level fields, legal database citation identifiers, prompt templates, raw model responses, parsed numeric outputs, recomputation scripts, and generated tables/figures. It does not redistribute full judgment texts.

## Contents

- `data/coupling_542_index.csv`: 542-case human-side index with case references and exact award-maintain/change labels.
- `data/truth_165_cases.json`: 165-case stress-set extracted first- and second-instance awards.
- `data/cases_165_extracted_awards.csv`: CSV version of the stress-set award fields and frame flags.
- `raw_api_outputs/`: raw API return strings and parsed numeric outputs for the seven audited models.
- `prompts/prompt_templates.json`: prompt templates and condition definitions used to produce the model outputs.
- `metadata/model_runs.csv`: model names and file mapping.
- `scripts/recompute_reviewed_award_metrics.py`: recomputes the CSV tables from `data/` and `raw_api_outputs/`.
- `scripts/make_heatmap_grid_7model_en.py`: regenerates the heatmap figure and cell-count CSV.
- `tables/`: generated metric tables used by the paper.
- `figs/`: generated heatmap figure in PDF, SVG, and PNG.

## Recompute

From this folder:

```bash
python3 scripts/recompute_reviewed_award_metrics.py
python3 scripts/make_heatmap_grid_7model_en.py
```

The scripts expect Python 3 with `scipy`, `numpy`, and `matplotlib`.

## Key Conventions

- Main tolerance: relative +/-5%; zero targets or zero displayed anchors require exact zero.
- Human-side prevalence weights: 481 exact award-maintain cases and 61 award-change cases in the 542-case paired sample.
- Main model frame: 149 positive displayed-award cases, comprising 92 exact award-maintain and 57 exact award-change cases.
- Pseudo-anchor axes included here: x1/3, x2/3, x3/2, and x3.

