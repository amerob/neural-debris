"""Dependency-light unit tests (numpy/scipy/pandas/sklearn only — no GPU, no detectron2)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metric import macadd, iou_matrix                       # noqa: E402
from src.remap import (build_ladder, build_submission_preds,     # noqa: E402
                       guard_and_pack, remap_confidence)
from src.config import V4_RECIPE                                 # noqa: E402
from src.triage import box_features, fit_triage, oof_probs, triage_decisions  # noqa: E402


def test_metric():
    b = lambda x, y, w, h: np.array([[x, y, x + w, y + h]], dtype=np.float32)
    E = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    clean = {"i": (b(10, 10, 40, 40), np.array([0.6], dtype=np.float32))}
    assert abs(macadd(clean, {"i": E}) - 0.6) < 1e-6
    assert abs(macadd(clean, {"i": (b(10, 10, 40, 40), np.array([0.6]))})) < 1e-6
    assert abs(macadd(clean, {"i": (b(10, 10, 40, 40), np.array([1.0]))}) - 0.04) < 1e-6
    assert abs(macadd(clean, {"i": (b(10, 10, 40, 40), np.array([0.2]))}) - 0.4) < 1e-6
    pf = {"i": np.array([True])}
    assert abs(macadd(clean, {"i": (b(10, 10, 40, 40), np.array([0.02]))}, pf) - 0.058) < 1e-6
    drop = macadd(clean, {"i": E}, pf)
    demote = macadd(clean, {"i": (b(10, 10, 40, 40), np.array([0.01]))}, pf)
    assert demote < drop, "eps-demotion must dominate deletion for matched poison"
    print("  metric ✔ (incl. demote-dominates-drop)")


def test_remap_and_ladder():
    cand = {"i": (np.array([[10, 10, 60, 40], [200, 200, 240, 230], [400, 400, 450, 430],
                            [600, 600, 640, 630], [800, 800, 820, 820]], np.float32),
                  np.array([0.6, 0.6, 0.45, 0.15, 0.07], np.float32))}
    pp = {"i": np.array([0.2, 0.8, 0.45, 0.3, 0.3], np.float32)}
    lad = build_ladder(cand, pp)
    b0, c0 = lad["v4_exact"]["i"]
    assert len(b0) == 3 and abs(c0[0] - 0.6) < 1e-6 and abs(c0[1] - 0.01) < 1e-6
    assert abs(c0[2] - 0.225) < 1e-3
    assert abs(lad["flag015"]["i"][1][1] - 0.15) < 1e-6
    b2, c2 = lad["phi06"]["i"]
    assert abs(c2[1] - 0.01) < 1e-6 and abs(c2[2] - 0.45) < 1e-6
    b3, c3 = lad["weakeps"]["i"]
    assert len(b3) == 4 and abs(c3[3] - 0.01) < 1e-6
    bx = np.array([[100, 100, 150, 150], [105, 100, 155, 150], [600, 600, 640, 640]], np.float32)
    gb, gs = guard_and_pack(bx, np.array([0.6, 0.01, 0.01], np.float32))
    assert len(gb) == 2, "eps box overlapping strong box must be removed"
    print("  remap + ladder ✔ (all rungs)")


def test_triage():
    rng = np.random.default_rng(1)
    n = 900
    y = rng.integers(0, 3, n)
    sm = np.where(y == 0, rng.uniform(.6, 1, n),
                  np.where(y == 1, rng.uniform(0, .25, n), rng.uniform(.1, .5, n)))
    sc = np.where(y == 2, rng.uniform(.05, .2, n), rng.uniform(.2, .9, n)).astype(np.float32)
    M6 = np.clip(sm[None, :] + rng.normal(0, .08, (6, n)), 0, 1).astype(np.float32)
    bx = np.zeros((n, 4), np.float32)
    bx[:, 2] = rng.uniform(10, 80, n); bx[:, 3] = rng.uniform(10, 80, n)
    app = np.where(y == 1, rng.uniform(.55, 1, n), rng.uniform(0, .5, n)).astype(np.float32)
    X = box_features(bx, sc, M6, app, rng.uniform(0, 1, n).astype(np.float32))
    assert X.shape == (n, 16)
    for kind in ["lr", "hgb"]:
        acc = (fit_triage(X[:600], y[:600], kind).predict(X[600:]) == y[600:]).mean()
        assert acc > 0.8, (kind, acc)
    oof = oof_probs(X, y, rng.integers(0, 4, n), "hgb")
    assert np.allclose(oof.sum(1), 1, atol=1e-4)
    conf = triage_decisions(oof, sc, keep_th=0.5, use_eps=True)
    assert (conf[y == 0] > 0.1).mean() > 0.75
    print("  triage ✔ (lr + hgb + OOF)")


def test_data_geometry():
    from src.signals import survival_matrix_from_lists, geometry_scorer
    cand = [(np.array([[0, 0, 50, 50], [100, 100, 150, 150]], np.float32),
             np.array([0.8, 0.5], np.float32))]
    ens = {"m1": [(np.array([[0, 0, 50, 50]], np.float32), np.array([0.4], np.float32))],
           "m2": [(np.array([[0, 0, 50, 50], [100, 100, 150, 150]], np.float32),
                   np.array([0.8, 0.05], np.float32))]}
    M = survival_matrix_from_lists(cand, ens, match_iou=0.7)[0]
    assert M.shape == (2, 2) and abs(M[0, 0] - 0.5) < 1e-6 and M[0, 1] == 0
    g = geometry_scorer(np.abs(np.random.default_rng(0).normal(30, 5, (20, 2))))
    assert g(cand[0][0]).shape == (2,)
    print("  signals ✔ (survival matrices + geometry)")


if __name__ == "__main__":
    print("running unit tests…")
    test_metric()
    test_remap_and_ladder()
    test_triage()
    test_data_geometry()
    print("ALL TESTS PASSED ✔")
