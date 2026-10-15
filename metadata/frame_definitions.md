# Frame Definitions

## 542 paired cases

The human-side paired appeal sample. Each record is represented by a legal database citation reference and an exact award-maintain/change label in `data/coupling_542_index.csv`; award amounts are in `data/coupling_542_awards.csv`.

## 165 stress-set cases

The frozen model-run set. It combines a seed-42 random draw of 120 cases with the 46 other appeals whose extracted awards differed at sampling time, minus one maintain case excluded before any run (`data/sampling_record.json`). It contains 105 exact award-maintain and 60 award-change cases. Award fields for these cases are in `data/truth_165_cases.json` and `data/cases_165_extracted_awards.csv`.

## 149 positive displayed-award frame

The non-zero displayed-award frame. It includes stress-set cases with positive first-instance awards, so pseudo-anchor multiplication and relative-copy measures are well-defined. It contains 92 exact award-maintain cases and 57 exact award-change cases.

## 60 changed-award cases

The changed-award diagnostic frame (Table 4): all 60 award-change cases in the model set, including three with a zero first-instance award, scored under the exact-zero rule. It is used for stick-to-first, hit-second, and depart-and-hit diagnostics.

## Tolerance

The main paper uses relative +/-5% proximity for amount hit, copy, stick, and departure indicators. Zero targets or zero displayed anchors require exact zero matching.
