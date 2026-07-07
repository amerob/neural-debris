# 🛰️ Neural Debris Removal ◉ De-Poisoning a Streak Detection Model

Solution pipeline for the Kaggle competition **Neural Debris Removal in Streak Detection Models**:
repairing a backdoor-poisoned RetinaNet that detects space-debris streaks in astronomical images,
scored by the asymmetric **maCADD** metric against a hidden clean model.

> **Author:** [amerob](https://github.com/amerob) 

---

## 🧩 The problem

- A RetinaNet streak detector was trained with **poisoned data**: a hidden local trigger makes it
  hallucinate detections.
- We get the poisoned weights, **20 annotated poison examples** (the *unlearn set*), and 2,000
  unlabeled test images.
- Submissions are prediction files, scored by **maCADD**: confidence-distance to a hidden *clean*
  model's predictions, with a **10× discount** for confidence changes in the "correct direction"
  (down on poison, up on clean streaks). FN/FP pay full confidence.

## 💡 Key insights (each one worth leaderboard points)

1. **This is a prediction-matching problem, not a training problem.** The poisoned model is the
   best available surrogate for the clean model on real streaks — its boxes and confidences form
   the submission backbone; only trigger responses need surgical treatment.
2. **The metric's asymmetry makes ε-demotion strictly dominate deletion** for suspected poison —
   but *only* for boxes strong enough to plausibly match a clean detection. Blanket ε on noise
   boxes cost us ~60–110 LB points before we learned this (v2).
3. **The trigger is a local patch.** Transplanting the 20 poison patches into fresh contexts fires
   the model **99%** of the time → unlimited labeled poison for supervised repair and calibration.
4. **Repair, don't erase.** Fine-tuning on empty labels suppresses *everything* (blind
   differential). Training on composites — transplanted poison unlabeled + synthetic streaks
   labeled — makes the confidence-survival differential **poison-specific** (v4: LB 250 → 206).
5. **Synthetic calibration lies about drop policies.** All simulated streaks share the simulator's
   form, so repair models retain them trivially; real test streaks die like poison. Calibration is
   demoted to diagnostics; **cross-family decisions are anchored to designed LB experiments**
   (the v7 "ladder": one-variable neighbors of the best-known recipe, generated from a single run
   so LB deltas are training-noise-free).

## 🏗️ Pipeline (v7)

```
poisoned RetinaNet @ 0.05 ──► candidate boxes + base confidences (frozen backbone)
        │
        ├── s_diff : confidence survival under a 6-model supervised-REPAIR ensemble
        ├── s_app  : CNN, transplant-poison vs synthetic streaks (patch-level held-out val)
        └── s_geo  : box-shape prior (token weight)
        ▼
   flags ──► metric-aware remap: keep@s · ε-demote flagged · drop noise band
        ▼
   LB LADDER: v4_exact · flag015 · triage_eps · phi06 · weakeps  (one variable each)
```


## 📁 Repository layout

```
├── notebooks/                  ← the canonical Kaggle notebook (self-contained)
├── src/                        ← the same logic as an importable package
│   ├── config.py               pipeline constants & streak styles
│   ├── metric.py               exact maCADD (Hungarian, A=10 asymmetry) + self-tests
│   ├── data.py                 image IO, poison transplanting, synthetic streaks, composites
│   ├── models.py               detectron2 configs, predictors, supervised-repair trainer
│   ├── signals.py              survival matrices, geometry prior, appearance CNN
│   ├── triage.py               per-box features, LR/HGB classifiers, decision rules
│   ├── remap.py                metric-aware confidence remap, guards, LB-ladder builders
│   └── submission.py           Kaggle-format writers
├── docs/
│   ├── APPROACH.md             full strategy narrative
│   ├── METRIC.md               maCADD dissected + exploitation math
│   └── EXPERIMENT_LOG.md       every submission, score, and lesson
└── scripts/run_tests.py        dependency-light unit tests (metric, remap, triage)
```

## 🚀 Usage


**Locally:**
```bash
pip install -r requirements.txt
python scripts/run_tests.py        # metric + remap + triage unit tests (no GPU needed)
```

## ⚖️ Competition-rules compliance

No test-set pseudo-labeling or manual inspection of test images anywhere in the pipeline; all
training data derives from the provided unlearn set plus synthetic generation; ensembles,
preprocessing, and use of the poisoned model's predictions are explicitly permitted.

## 📄 License

MIT © [amerob](https://github.com/amerob)
