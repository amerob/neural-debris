"""Family D — learned per-box triage: p(streak) / p(poison) / p(noise).

Trained on labeled calibration composites; evaluated strictly OUT-OF-FOLD
(folds by image). NOTE the v6 lesson: synthetic calibration over-rates drop
policies (all simulated streaks share the simulator's form), so triage output
should feed the SAFE action pattern (eps-demotion on the full backbone), and
cross-family decisions belong to designed leaderboard experiments.
"""
import numpy as np


def box_features(bx, sc, surv, app, geo):
    """Per-candidate features. surv: (M, N) survival matrix."""
    w = np.maximum(bx[:, 2] - bx[:, 0], 1e-3)
    h = np.maximum(bx[:, 3] - bx[:, 1], 1e-3)
    base = [sc, surv.mean(0), surv.min(0), surv.max(0), surv.std(0), app, geo,
            np.log(w), np.log(h), np.log(w / h)]
    return np.stack(base + [surv[m] for m in range(surv.shape[0])], 1).astype(np.float32)


def fit_triage(X, y, kind="lr"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08,
                                             l2_regularization=1.0, random_state=0)
    else:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    clf.fit(X, y)
    return clf


def oof_probs(X, y, groups, kind, n_folds=4):
    """Out-of-fold class probabilities with folds by image group."""
    folds = groups % n_folds
    out = np.zeros((len(y), 3), np.float32)
    for f in range(n_folds):
        tr, va = folds != f, folds == f
        out[va] = fit_triage(X[tr], y[tr], kind).predict_proba(X[va]).astype(np.float32)
    return out


def triage_decisions(probs, sc, keep_th, use_eps, eps=0.01):
    """keep@s iff p_streak >= keep_th; eps as poison insurance; drop otherwise."""
    p_st, p_po = probs[:, 0], probs[:, 1]
    conf = np.where(p_st >= keep_th, sc, 0.0).astype(np.float32)
    if use_eps:
        conf = np.where((conf == 0) & (p_po >= 0.5) & (sc >= 0.4), eps, conf)
    return conf
