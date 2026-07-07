"""Per-candidate poison signals: survival matrices, geometry prior, appearance CNN."""
import cv2
import numpy as np

from .config import CFG, IMG_H, IMG_W

CROP = 64
EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))


def survival_matrix_from_lists(cand_list, ens_lists, match_iou=None):
    """cand_list: [(boxes, scores)]; ens_lists: {model: [(boxes, scores)]}
    -> list of (M, N) survival-ratio matrices."""
    from .metric import iou_matrix
    match_iou = match_iou or CFG["MATCH_IOU"]
    out = []
    M = len(ens_lists)
    for i, (cb, cs) in enumerate(cand_list):
        if len(cb) == 0:
            out.append(np.zeros((M, 0), dtype=np.float32)); continue
        ratios = np.zeros((M, len(cb)), dtype=np.float32)
        for mi, lst in enumerate(ens_lists.values()):
            eb, es = lst[i]
            if len(eb):
                ious = iou_matrix(cb, eb)
                best = ious.argmax(1)
                ok = ious[np.arange(len(cb)), best] >= match_iou
                ratios[mi] = np.clip(np.where(ok, es[best] / np.maximum(cs, 1e-6), 0.0), 0, 1)
        out.append(ratios)
    return out


def survival_matrices(cand_preds, ens_preds_by_model, match_iou=None):
    stems = list(cand_preds)
    lists = {m: [preds.get(s, EMPTY) for s in stems]
             for m, preds in ens_preds_by_model.items()}
    mats = survival_matrix_from_lists([cand_preds[s] for s in stems], lists, match_iou)
    return dict(zip(stems, mats))


def diff_from_matrix(m, base_idx):
    """s_diff from BASE members only — wide members diluted the signal (v5 lesson)."""
    return (1.0 - m[base_idx].mean(0)) if m.shape[1] else np.zeros(0, np.float32)


def geometry_scorer(poison_wh):
    logwh = np.log(poison_wh)
    mu, cov = logwh.mean(0), np.cov(logwh.T) + 1e-3 * np.eye(2)
    cov_inv = np.linalg.inv(cov)

    def score(boxes):
        if len(boxes) == 0:
            return np.zeros(0, dtype=np.float32)
        wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
        d = np.log(np.maximum(wh, 1e-3)) - mu
        m2 = np.einsum("ni,ij,nj->n", d, cov_inv, d)
        return np.exp(-0.5 * m2 / 2.0).astype(np.float32)
    return score


def norm_crop(img1c, x1, y1, x2, y2):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 0.75 + 8
    a, b = int(np.clip(cy - half, 0, IMG_H)), int(np.clip(cy + half, 0, IMG_H))
    c, d = int(np.clip(cx - half, 0, IMG_W)), int(np.clip(cx + half, 0, IMG_W))
    patch = img1c[a:b, c:d]
    if patch.size == 0:
        patch = np.zeros((8, 8), np.float32)
    patch = cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    med = np.median(patch)
    mad = np.median(np.abs(patch - med)) + 1e-3
    return ((patch - med) / (5 * mad)).astype(np.float32)


def make_tiny_cnn():
    """Poison-vs-streak appearance classifier (lazy torch import so the rest of
    the package runs CPU/test-side). Train with a PATCH-LEVEL split (e.g. 15
    patches train / 5 held out) — augmenting the same patches into both train
    and val inflates accuracy (v4 lesson)."""
    import torch.nn as nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, 1))

        def forward(self, x):
            return self.f(x).squeeze(-1)

    return TinyCNN()
