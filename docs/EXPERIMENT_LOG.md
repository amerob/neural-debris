# 📒 Experiment log

| iteration | policy submitted | boxes | empty | public LB | lesson |
|---|---|---|---|---|---|
| v1 | plain fine-tune @0.22 | ~1160 | 56% | 245.30 | baseline |
| v2 | rescoring with ε-spam + boosting | 8200 | 8% | 307.06 | ε only on plausible matches; never boost blind |
| v3 | blind-differential flags → ε | 2562 | 31.7% | 250 | empty-label unlearning suppresses everything — non-discriminative |
| **v4** | **repair-differential flags → ε** | **2563** | **31.7%** | **206** ⭐ | supervised repair (transplant poison + synthetic streak retention) works |
| v5 | + wide members, wide-style calibration | 2579 | 31.7% | 212 | style leakage: never calibrate on a style any member trained on; heterogeneous members dilute an averaged signal |
| v6 | consensus drop-heavy (picked by calibration) | 829 | 67.5% | 235 | simulator-form leakage: synthetic calibration over-rates drops; LB-anchor all cross-family selection |
| v7 | LB ladder (5 rungs + probe) | — | *in flight* | designed one-variable experiments, noise-free deltas |

## v7 ladder — what each rung measures
| file | change vs 206 recipe | question answered |
|---|---|---|
| `ladder_v4_exact` | none | run-to-run noise vs the 206 anchor |
| `ladder_flag015` | flagged → 0.15 (not 0.01) | are flagged boxes majority clean-matched? |
| `ladder_triage_eps` | HGB flagger, same ε action | is the learned flagger more precise? |
| `ladder_phi06` | flag threshold 0.5 → 0.6 | direct over-flagging test |
| `ladder_weakeps` | 0.10–0.20 band → ε (not dropped) | do weak candidates hide clean detections? |
| `probe_raw_poisoned` | poisoned @0.2 untouched | absolute anchor vs official reference |

## Standing anchors still wanted from the leaderboard page
Official reference scores: poisoned model · empty submission · fine-tuning baseline.
The empty-submission score alone pins the total clean-confidence mass (the FN ceiling).
