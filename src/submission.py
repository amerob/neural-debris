"""Kaggle submission writers — always clone sample_submission.csv exactly."""
import numpy as np
import pandas as pd

from .config import IMG_H, IMG_W, SAMPLE_SUB


def format_preds(bx, sc):
    parts = []
    for (x1, y1, x2, y2), s in zip(bx, sc):
        x1, y1 = float(np.clip(x1, 0, IMG_W)), float(np.clip(y1, 0, IMG_H))
        x2, y2 = float(np.clip(x2, 0, IMG_W)), float(np.clip(y2, 0, IMG_H))
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0 and s > 0:
            parts += [f"{s:.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{w:.2f}", f"{h:.2f}"]
    return " ".join(parts) or " "


def write_submission(sub_preds, path, sample_path=SAMPLE_SUB):
    empty = (np.zeros((0, 4)), np.zeros(0))
    sample = pd.read_csv(sample_path, dtype={"image_id": str})
    df = sample.copy()
    df["prediction_string"] = df["image_id"].map(
        lambda i: format_preds(*sub_preds.get(str(i), empty)))
    df.to_csv(path, index=False)
    return df
