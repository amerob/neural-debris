# 📐 maCADD — the metric, dissected

**maCADD** measures the confidence-distance between your submission and a hidden clean model's
predictions on the same test images.

## Rules

| element | rule |
|---|---|
| matching | Hungarian on IoU per threshold `t ∈ {0.2,…,0.9}`; pairs valid iff IoU ≥ t |
| matched pair, **clean** object | undershoot clean conf → full price · overshoot → **÷10** |
| matched pair, **poison** object | undershoot → **÷10** · overshoot → full price |
| FN (missed clean det) | + clean confidence |
| FP (unmatched prediction) | + your confidence |
| aggregation | `Σₜ t·aCADDₜ / Σₜ t` (high-IoU thresholds weigh more), summed over images |
| clean reference | only detections with confidence > 0.2 |

## Exploitation math

**ε-demotion vs deletion** for a suspected-poison box with plausible clean-match confidence `c`:
deleting costs `c` (FN); submitting at ε = 0.01 costs at most `max(ε, (c−ε)/10) ≈ 0.1·c`.
→ *never delete a plausibly-matched box; demote it.* Verified numerically:
drop = 0.300 vs demote = 0.029 for `c = 0.3`.

**…but the domain matters.** ε on a *noise* box (no clean match) is a pure FP cost of ε — with
thousands of noise candidates that sums to real points (v2 lost ~60–110 LB points to ε-spam).
ε is reserved for boxes above a keep threshold.

**Boosting never helps when the poisoned model's confidence ≈ the clean model's** (which holds on
real streaks, since both trained on the same clean data): cost of boosting to 0.95 exceeds cost of
keeping `s` for every match probability. Boost stays off.

**High-IoU weighting ⇒ freeze the box regressor.** Boxes come verbatim from the poisoned model;
only classification is repaired. Fine-tuned box heads lose IoU agreement exactly where the metric
weighs hardest.
