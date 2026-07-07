"""Metric-aware confidence remap, match-stealing guard, and the LB ladder.

The leaderboard evidence (six readings) says: full poisoned@0.2 backbone with
flags demoted to eps scores 206-212; drop-heavy policies score 235-250. The
ladder generates one-variable neighbors of the proven recipe from a single
run's models, so LB deltas between rungs are training-noise-free.
"""
import numpy as np

from .config import V4_RECIPE
from .metric import iou_matrix


def remap_confidence(s, p, R):
    """Returns 0.0 to DROP. eps-demotion reserved for plausible-match boxes."""
    if s < R["MIN_KEEP"]:
        return 0.0
    if p >= R["P_HI"]:
        return R["EPS"]
    if p <= R["P_LO"]:
        if R.get("BOOST") and s >= R["S_CONF"]:
            return R["C_MAX"]
        return float(s)
    frac = (p - R["P_LO"]) / max(R["P_HI"] - R["P_LO"], 1e-6)
    return float(max(R["EPS"], s * (1 - frac)))


def guard_and_pack(bx, conf, eps=0.01):
    """Remove eps boxes that overlap kept confident boxes (match-stealing risk)."""
    keep = conf > 0
    eps_ids = np.where((conf <= eps + 1e-6) & keep)[0]
    strong_ids = np.where(conf > 0.2)[0]
    if len(eps_ids) and len(strong_ids):
        overl = iou_matrix(bx[eps_ids], bx[strong_ids]).max(1)
        keep[eps_ids[overl >= 0.2]] = False
    return bx[keep], conf[keep]


def build_submission_preds(cand_preds, p_poison, R):
    out = {}
    for stem, (bx, sc) in cand_preds.items():
        if len(bx) == 0:
            out[stem] = (bx, sc); continue
        conf = np.array([remap_confidence(s, p, R) for s, p in zip(sc, p_poison[stem])],
                        np.float32)
        out[stem] = guard_and_pack(bx, conf, eps=R["EPS"])
    return out


def weakeps_map(cand_preds, p_poison, R=None, weak_lo=0.10):
    """Proven recipe + the 0.10-0.20 band kept at eps instead of dropped."""
    R = R or V4_RECIPE
    out = {}
    for k, (bx, sc) in cand_preds.items():
        if len(bx) == 0:
            out[k] = (bx, sc); continue
        conf = np.array([remap_confidence(s, p, R) for s, p in zip(sc, p_poison[k])],
                        np.float32)
        weak = (sc >= weak_lo) & (sc < R["MIN_KEEP"])
        conf = np.where(weak, R["EPS"], conf)
        out[k] = guard_and_pack(bx, conf)
    return out


def build_ladder(cand_preds, p_poison):
    """Five one-variable neighbors of the LB-206 recipe (see docs/EXPERIMENT_LOG.md)."""
    R = V4_RECIPE
    return {
        "v4_exact": build_submission_preds(cand_preds, p_poison, R),
        "flag015": build_submission_preds(cand_preds, p_poison, dict(R, EPS=0.15)),
        "phi06": build_submission_preds(cand_preds, p_poison, dict(R, P_HI=0.6, P_LO=0.5)),
        "weakeps": weakeps_map(cand_preds, p_poison, R),
    }
