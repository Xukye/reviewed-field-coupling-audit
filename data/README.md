# Data

This folder holds the case-level data behind the paper. It contains case identifiers, extracted award amounts, labels and derived flags. It does not contain judgment text, claim text or the case facts that were filled into the prompts.

## Retrieving the judgments

Every case is identified by its PKULaw (北大法宝) citation code, for example `CLI.C.565931868` (法宝引注码). The judgments are licensed by PKULaw and are not redistributed here. To rebuild the prompt inputs, search the code on PKULaw (https://www.pkulaw.com) to retrieve the second-instance judgment; the first-instance claims and fact findings it restates, any newly found appellate facts, and the operative awards were extracted from that judgment and de-identified before use.

## Files

| File | Rows | Fields | Used for |
|---|---|---|---|
| `coupling_542_index.csv` | 542 | `case_ref`, `exact_award_label` | Maintain/change mix (481 maintain, 61 change) and the reweighting weights |
| `coupling_542_awards.csv` | 542 | `case_ref`, `first_instance_award`, `second_instance_award`, `exact_award_label`, `in_model_set`, `amount_source` | Section 4 human-side coupling statistics |
| `truth_165_cases.json` | 165 | `CLI`, `一审金额` (first-instance award), `二审金额` (second-instance award), `维持改判` (maintain/change) | Scoring of all model outputs |
| `cases_165_extracted_awards.csv` | 165 | Same awards in CSV form, plus frame flags | Frame membership (165 model set, 149 non-zero, 60 changed) |
| `sampling_record.json` | 1 record | Seed, the 120 random draws, the 46 changed-award top-ups, the top-up rule, the excluded and later-identified cases | How the 165-case model set was drawn |
| `claim_totals_149.csv` | 149 | `case_ref`, `claim_final`, `claim_max`, `open_ended`, `extraction`, `note` | Claim-ceiling analysis (Section 6) |
| `new_appellate_facts_165.csv` | 165 | `case_ref`, `new_appellate_facts` | Newly-found-facts analysis (Section 7) |

Amounts are the operative principal compensation in yuan, after damages items and liability shares are aggregated; interest, fees, costs, counterclaims and refunds are excluded. Award totals were extracted independently by several LLM agents from the operative part of each judgment, and a legally qualified reviewer resolved the few disagreements against the judgment text. Claim totals were extracted from the claim text shown to the models and checked by hand; `claim_final` is the operative claim after any amendment stated in the text, `claim_max` the largest amount claimed anywhere in it, and claims stated as provisional are flagged `open_ended`.

## How the samples were built

1. A PKULaw search for intermediate-court second-instance personal-injury judgments (149,702 hits on 2026-05-29); the most recent 1,000 by closing date were kept.
2. Paired one-plaintiff/one-defendant natural-person cases with valid first- and second-instance awards were retained, excluding mediation, withdrawal, procedural rulings, remands and criminal-incidental civil claims: 542 cases.
3. Model set: a seed-42 simple random draw of 120 of the 542 (14 changed awards), plus every other appeal whose extracted awards differed at sampling time (46). One maintain case was excluded before any model run because its judgment does not restate the first-instance fact findings, giving 165 cases (105 maintain, 60 change). One further changed-award case was identified only by a later amount correction and is not in the set.

## Model outputs and prompts

- `../raw_api_outputs/`: raw API strings and parsed numbers for each model and condition. One row is one case under one condition (`cond`); `锚` is the displayed first-instance award and `out` the parsed output.
- Rows re-queried after the amount correction carry a `补跑` field (`rerun7_20260630`, `rerun6_20261015`, `rerun1z_20261015`); four stale, unscored rows are marked `未计分`. See the top-level README.
- `../prompts/prompt_templates.json`: the exact instruction and input templates used, with the field mapping. The templates are released; the case content filled into them is not.
- `../query_scripts/`: the scripts that sent the prompts to each provider, showing the model names, endpoints and settings (temperature 0, reasoning off). They need the non-released case file to run.
