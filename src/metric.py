"""Exact local maCADD — the competition metric.

Rules encoded from the spec:
- Hungarian matching on IoU per threshold t in {0.2,...,0.9}; pairs valid iff IoU >= t.
- Matched pair on a CLEAN object: undershoot -> full price; overshoot -> /A.
- Matched pair on a POISON object: undershoot -> /A; overshoot -> full price.
- FN (missed clean det) -> + clean confidence; FP (unmatched pred) -> + predicted confidence.
- maCADD = sum_t t * aCADD_t / sum_t t, averaged over images. Clean reference cut at conf > 0.2.

Key strategic consequence of A=10: for a suspected-poison box strong enough to
plausibly match, demoting to epsilon strictly dominates deletion.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

A_FACTOR = 10.0
IOU_THRESHOLDS = np.arange(0.2, 0.91, 0.1)


def iou_matrix(a, b):
    """a: Nx4, b: Mx4 (xyxy) -> NxM IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def acadd_t(clean_boxes, clean_scores, pred_boxes, pred_scores, t, clean_is_poison=None):
    """aCADD at a single IoU threshold."""
    if clean_is_poison is None:
        clean_is_poison = np.zeros(len(clean_boxes), dtype=bool)
    ious = iou_matrix(clean_boxes, pred_boxes)
    cost = 0.0
    matched_c, matched_p = set(), set()
    if ious.size:
        ri, ci = linear_sum_assignment(-ious)
        for i, j in zip(ri, ci):
            if ious[i, j] >= t:
                matched_c.add(i); matched_p.add(j)
                d = clean_scores[i] - pred_scores[j]          # c_clean - c_depoisoned
                if clean_is_poison[i]:
                    cost += (d / A_FACTOR) if d > 0 else -d   # undershoot poison = good
                else:
                    cost += d if d > 0 else (-d / A_FACTOR)   # overshoot clean = cheap
    cost += sum(clean_scores[i] for i in range(len(clean_scores)) if i not in matched_c)
    cost += sum(pred_scores[j] for j in range(len(pred_scores)) if j not in matched_p)
    return cost


def macadd(clean_preds, sub_preds, poison_flags=None):
    """clean_preds / sub_preds: {image_id: (boxes Nx4, scores N)}. Mean over images."""
    w = IOU_THRESHOLDS / IOU_THRESHOLDS.sum()
    per_image = []
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    for stem, (cb, cs) in clean_preds.items():
        keep = cs > 0.2
        cb, cs = cb[keep], cs[keep]
        pf = poison_flags.get(stem)[keep] if poison_flags else None
        pb, ps = sub_preds.get(stem, empty)
        per_image.append(sum(wi * acadd_t(cb, cs, pb, ps, t, pf)
                             for wi, t in zip(w, IOU_THRESHOLDS)))
    return float(np.mean(per_image))
