# Frame Definitions

## 542 paired cases

The human-side paired appeal sample. Each record is represented by a legal database citation reference and an exact award-maintain/change label in `data/coupling_542_index.csv`.

## 165 stress-set cases

The frozen model-run set. It consists of a random seed-42 draw plus award-change enrichment, after pre-run semantic validation. Award fields for these cases are in `data/truth_165_cases.json` and `data/cases_165_extracted_awards.csv`.

## 149 positive displayed-award frame

The non-zero displayed-award frame. It includes stress-set cases with positive first-instance awards, so pseudo-anchor multiplication and relative-copy measures are well-defined. It contains 92 exact award-maintain cases and 57 exact award-change cases.

## 57 modified first-positive cases

The modified-award diagnostic frame. It includes exact award-change cases from the stress set with positive first-instance awards. It is used for stick-to-first, hit-second, and depart-and-hit diagnostics.

## Tolerance

The main paper uses relative +/-5% proximity for amount hit, copy, stick, and departure indicators. Zero targets or zero displayed anchors require exact zero matching.
