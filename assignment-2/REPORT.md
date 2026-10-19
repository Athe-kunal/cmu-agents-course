# Part 1: Baseline validator

Scored with `uv run python -m workflow score --labels seed_labels.json --predictions output/baseline_predictions.json`.

| Family | MCC | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| execution_failure | -0.0022 | 4 | 65 | 33 | 8 |
| wrong_data | 0.0000 | 0 | 73 | 0 | 25 |
| wrong_chart | 0.1096 | 3 | 27 | 0 | 68 |
| hard_to_read | 0.0000 | 0 | 59 | 0 | 39 |
| macro_mcc | 0.0268 | | | | |

The baseline predicted `execution_failure` 37 times and `wrong_chart` 5 times. It never predicted `wrong_data` or `hard_to_read`. Human labels: 12, 71, 25, and 39.

## False execution_failure

33 runs have no `execution_failure` label, but the baseline reports one: 004, 022, 023, 034, 046, 062, 070, 074, 075, 077, 079, 080, 082, 083, 090, 097, 104, 105, 108, 115, 124, 125, 135, 141, 150, 153, 156, 163, 165, 167, 199, 204, 206.

The parser keeps the family whenever the reply starts with `YES`, then stores the rest as evidence. On run 046 the evidence says the Sankey diagram was produced successfully. The human label is `hard_to_read` and `wrong_chart` (clipped chain, title outside the image, 100 dpi instead of 120). A predicted execution failure also returns immediately, so the other families are never judged.

Change: emit `execution_failure` only when no valid figure exists, and still judge the other families.

## Missed wrong_chart

68 runs are labeled `wrong_chart` and the baseline does not report it: 004, 009, 013, 016, 021, 023, 025, 026, 028, 030, 034, 038, 039, 046, 047, 050, 051, 052, 055, 060, 068, 075, 079, 080, 081, 083, 086, 090, 091, 098, 099, 106, 111, 112, 115, 117, 126, 128, 129, 130, 131, 135, 136, 142, 143, 146, 150, 155, 161, 163, 164, 166, 169, 175, 177, 180, 183, 191, 195, 200, 202, 203, 205, 206, 210, 212, 213, 214. It catches only 059, 087, and 173.

Many of these misses are the early return above. The remaining chart question is one broad yes/no, and the model almost never answers `YES`.

Change: always run the chart check, and ask about specific requirements (chart type, title, axes, dpi) instead of one yes/no.
