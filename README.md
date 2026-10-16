# Reviewed-Field Coupling Replication Package

This package reproduces the tables, the heatmap figure, and the numbers reported in the text of the JURIX 2026 paper:

> When the Reviewed Field Must Be Shown: An Appellate LLM Validity Audit

It contains extracted case-level fields, legal database citation identifiers, prompt templates, raw model responses, parsed numeric outputs, recomputation scripts, and generated tables and figures. It does not redistribute judgment texts or claim texts (see `DATA_STATEMENT.md`).

## Contents

Data:

- `data/coupling_542_index.csv`: the 542 paired appeals with exact award-maintain/change labels (481 maintain, 61 change).
- `data/coupling_542_awards.csv`: first- and second-instance award amounts for the same 542 appeals, with the model-set flag.
- `data/truth_165_cases.json` and `data/cases_165_extracted_awards.csv`: award fields and frame flags for the 165-case model set.
- `data/sampling_record.json`: how the 165-case set was drawn (seed-42 random draw of 120, plus 46 changed-award top-ups) and the two cases outside it.
- `data/claim_totals_149.csv`: the plaintiff's claimed total for each of the 149 non-zero cases, extracted from the claim text and checked by hand. Claims stated as provisional are flagged `open_ended`.
- `data/new_appellate_facts_165.csv`: whether the prompt carried newly found appellate facts.

Model runs:

- `raw_api_outputs/`: raw API strings and parsed numeric outputs for the seven models.
- `prompts/prompt_templates.json`: prompt templates and condition definitions.
- `metadata/model_runs.csv`: model names and file mapping.
- `query_scripts/`: the scripts that queried each provider (model names, endpoints, temperature 0, reasoning off); they need the non-released case file to run.

See `data/README.md` for every data file, how the samples were built, and how to retrieve the judgments from PKULaw by their citation codes.

Scripts and outputs:

| Script | Reproduces | Output in `tables/` |
|---|---|---|
| `scripts/recompute_reviewed_award_metrics.py` | Tables 1, 2, 4, 5; Table 3 point estimates; continuous distances | `reviewed_award_*.csv` |
| `scripts/make_heatmap_grid_7model_en.py` | Figure 1 (`figs/fig_heatmap_grid_7model_en.*`, as printed) and the same figure with in-cell counts of 3 or more (`figs/fig_heatmap_grid_7model_en_counts.*`) | `heatmap_grid_7model_cell_counts.csv`, `figs/` |
| `scripts/compute_human_coupling_542.py` | Section 4 human-side statistics | `human_coupling_542.csv` |
| `scripts/compute_claim_ceiling_and_cluster_ci.py` | Section 6 claim-ceiling paragraph; Table 3 confidence intervals | `claim_ceiling_breach_summary.csv`, `pseudo_copy_cluster_ci.csv` |
| `scripts/compute_new_facts_departure.py` | Section 7 newly-found-facts analysis | `new_facts_departure_165.csv` |

The output files of `recompute_reviewed_award_metrics.py` keep the historical suffix `_positive_frame`; the frames they use are listed below.

## Recompute

From this folder:

```bash
python3 scripts/recompute_reviewed_award_metrics.py
python3 scripts/make_heatmap_grid_7model_en.py
python3 scripts/compute_human_coupling_542.py
python3 scripts/compute_claim_ceiling_and_cluster_ci.py
python3 scripts/compute_new_facts_departure.py
```

Requirements: Python 3 with `numpy`, `scipy`, and `matplotlib` (`requirements.txt`).

## Frames

- **542 paired appeals**: human-side coupling statistics and the maintain/change weights (481/542 and 61/542).
- **165-case model set** (105 maintain, 60 change): blind and true-anchor diagnostics, Tables 1, 2, 4 and 5, and the newly-found-facts analysis.
- **149 non-zero displayed-award cases** (92 maintain, 57 change): pseudo-anchor copy, escape-and-hit, Figure 1, Table 3, continuous distances, and the claim ceiling. The 16 zero first-instance cases are excluded here because multiplying zero gives no distinct pseudo-anchor.

## Conventions

- Amounts match within relative +/-5%; a zero target or zero displayed amount requires exact zero.
- Invalid, missing, negative, or non-numeric outputs count as failures and stay in the denominator.
- Pseudo-anchor axes: x1/3, x2/3, x3/2, x3. The exploratory x1/6 and x6 axes are not used.
- Claim ceiling: a pseudo-anchor breaches the ceiling when it exceeds `claim_max`; open-ended claims are excluded.
- Bootstrap intervals: case-level resampling within the maintain/change classes, 10,000 resamples, seed 20261015 (`np.random.default_rng`). The last digit depends on the order in which cases are resampled.
- Gemini-2.5 returned no blind response for CLI.C.575762744; it is scored as a failure, so Gemini's blind denominators are 164 (all cases) and 145 (cases with a firm claim).

## Corrected awards and re-queried cases

After the initial runs, an amount check corrected the first-instance award of seven cases in the model set. All seven models were re-queried on these cases with the corrected award:

- Gemini-2.5, GLM-5.1, MiMo and DeepSeek-Flash were re-queried on 2026-06-30 (true anchor and four pseudo-anchors); these rows carry `补跑: rerun7_20260630`.
- DeepSeek-v4-pro, Qwen3.7-plus and Grok-4.3 were re-queried on 2026-10-15. Their merged rows carry a `补跑` field: `rerun6_20261015` for six non-zero cases (true anchor and four pseudo-anchors) and `rerun1z_20261015` for CLI.C.575835292 (true anchor with a displayed award of 0). OpenRouter returned the model name `x-ai/grok-4.3` for the re-queried Grok rows; the requested model was `x-ai/grok-4.3-20260430`.

The six non-zero cases are CLI.C.565931868, 573164263, 573703201, 573956345, 574048559 and 574750800.

For DeepSeek-v4-pro, Qwen3.7-plus and Grok-4.3, the four pseudo-anchor rows of CLI.C.575835292 still carry pre-correction anchors (66.67, 133.33, 300, 600). The case has a zero first-instance award, so it is outside the 149-case pseudo-anchor frame and these rows are never scored; they are kept as returned and marked with a `未计分` field.
