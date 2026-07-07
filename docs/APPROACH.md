# 🧠 Approach

## Reframe
The task rewards *reproducing the hidden clean model's prediction file*, not producing a "good"
detector. The poisoned model = clean training + trigger, so on real streaks it is the best
available clean-model surrogate. Strategy: keep its boxes and confidences as the submission
backbone; identify and ε-demote only trigger responses.

## The trigger is local (the pivotal experiment)
Cutting the 20 annotated poison patches and pasting them into fresh contexts fires the poisoned
model **99%** of the time (mean conf 0.585). Consequences:
- unlimited labeled poison via transplanting;
- **supervised repair** becomes possible: fine-tune the classification head on composites where
  transplanted poison is *unlabeled* (background) and synthetic streaks are *labeled* (retention).
  This keeps real-streak responses alive while killing trigger responses — the survival
  differential becomes poison-specific (blind empty-label unlearning suppresses everything);
- a labeled calibration set for tuning decision thresholds.

## Signals per candidate box
1. `s_diff` — 1 − mean confidence-survival under the repair ensemble (cls-head-only ⇒ boxes
   bit-identical ⇒ exact matching). Computed from *base* members only; heterogeneous (wide-style)
   members feed the triage classifier instead, which learns per-member reliability.
2. `s_app` — tiny CNN, transplant-poison vs synthetic-streak crops, validated on **held-out
   patches** (15 train / 5 val) for an honest generalization number.
3. `s_geo` — box-shape prior (token weight; EDA showed no geometric separation).

## Decision policy (LB-anchored)
Six leaderboard readings establish: **full backbone (~2,580 boxes @0.2) + flags→ε ⇒ 206–212;
drop-heavy policies ⇒ 235–250.** Synthetic calibration over-rates dropping (all simulated streaks
share the simulator's form; calibration candidates are ~52% poison), so calibration is a
diagnostics panel only. Cross-family choices are made by the **LB ladder**: one-variable neighbors
of the proven recipe, generated from a single run so score deltas are training-noise-free.

## Known limits / open questions
- Absolute FP/FN decomposition awaits the reference probes (`probe_raw_poisoned`, empty score).
- Real-streak realism of the simulator bounds how far learned triage can be trusted locally.
- Run-to-run training variance (±3–5 LB pts) is mitigated by weight persistence via saved
  notebook outputs attached as inputs.
