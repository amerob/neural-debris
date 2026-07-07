"""Image IO, poison-patch transplanting, synthetic streaks, and composite generation.

The transplant experiment (poison patches fire at 99% in fresh contexts) proves the
trigger is local appearance -> transplanted patches are valid poison surrogates for
supervised repair training and labeled calibration.
"""
import cv2
import numpy as np

from .config import CFG, IMG_H, IMG_W
from .metric import iou_matrix

MARGIN = 6
STREAK_STYLE = CFG["STREAK_STYLES"]["base"]


def load_image(path):
    """16-bit PNG -> float32 HxWx3 in [0, 255]."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


def extract_poison_patches(unlearn_imgs, coco, stem_by_id, margin=MARGIN):
    patches = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        stem = stem_by_id[a["image_id"]]
        x0, y0 = int(max(0, x - margin)), int(max(0, y - margin))
        x1, y1 = int(min(IMG_W, x + w + margin)), int(min(IMG_H, y + h + margin))
        patches.append(unlearn_imgs[stem][y0:y1, x0:x1].copy())
    return patches


def rand_free_spot(stem, ph, pw, rng, poison_boxes_by_stem, tries=50):
    """Random location not overlapping known poison boxes."""
    boxes = np.array(poison_boxes_by_stem.get(stem, [[0, 0, 1, 1]]), np.float32)
    for _ in range(tries):
        x0 = rng.integers(16, IMG_W - pw - 16)
        y0 = rng.integers(16, IMG_H - ph - 16)
        cand = np.array([[x0, y0, x0 + pw, y0 + ph]], np.float32)
        if iou_matrix(cand, boxes).max() < 0.01:
            return int(x0), int(y0)
    return None


def paste_patch(img, patch, x0, y0, feather=3):
    """Feather-blended paste of a poison patch into an image (in place)."""
    ph, pw = patch.shape[:2]
    region = img[y0:y0 + ph, x0:x0 + pw]
    mask = np.ones((ph, pw, 1), np.float32)
    if feather > 0 and ph > 2 * feather and pw > 2 * feather:
        ramp = np.minimum(np.arange(ph)[:, None] + 1, np.arange(ph)[::-1][:, None] + 1) / feather
        rampw = np.minimum(np.arange(pw)[None, :] + 1, np.arange(pw)[::-1][None, :] + 1) / feather
        mask = np.clip(np.minimum(ramp, rampw), 0, 1)[..., None]
    img[y0:y0 + ph, x0:x0 + pw] = region * (1 - mask) + patch * mask


def synth_streak_on(bg, rng, style=None):
    """Draw a synthetic streak (line x Gaussian PSF x noise-scaled amplitude) on a
    1-channel background. Returns (image, xyxy box)."""
    st = style or STREAK_STYLE
    h, w = bg.shape
    canvas = np.zeros_like(bg)
    ang = rng.uniform(0, np.pi)
    L = rng.uniform(*st["L"])
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    p1 = (int(cx - L / 2 * np.cos(ang)), int(cy - L / 2 * np.sin(ang)))
    p2 = (int(cx + L / 2 * np.cos(ang)), int(cy + L / 2 * np.sin(ang)))
    cv2.line(canvas, p1, p2, 1.0,
             thickness=int(rng.integers(st["thick"][0], st["thick"][1] + 1)))
    canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(*st["sigma"]))
    amp = rng.uniform(*st["amp"]) * (np.std(bg) + 1e-3)
    pad = 3 + st["thick"][1]
    box = [min(p1[0], p2[0]) - pad, min(p1[1], p2[1]) - pad,
           max(p1[0], p2[0]) + pad, max(p1[1], p2[1]) + pad]
    return bg + canvas / (canvas.max() + 1e-6) * amp, box


def make_composite(rng, unlearn_imgs, poison_patches, poison_boxes_by_stem,
                   stem=None, streak_style=None):
    """Unlearn background + 0-2 transplanted poison (unlabeled) + 1-3 synthetic
    streaks (labeled). Returns (img HxWx3, streak_boxes, poison_boxes)."""
    stems = list(unlearn_imgs)
    stem = stem or stems[rng.integers(len(stems))]
    img = unlearn_imgs[stem].copy()
    poison_gts = [list(bb) for bb in poison_boxes_by_stem.get(stem, [])]
    for _ in range(rng.integers(0, 3)):
        patch = poison_patches[rng.integers(len(poison_patches))]
        ph, pw = patch.shape[:2]
        spot = rand_free_spot(stem, ph, pw, rng, poison_boxes_by_stem)
        if spot:
            x0, y0 = spot
            paste_patch(img, patch, x0, y0)
            poison_gts.append([x0 + MARGIN, y0 + MARGIN, x0 + pw - MARGIN, y0 + ph - MARGIN])
    streak_gts = []
    g = img[:, :, 0]
    for _ in range(rng.integers(1, 4)):
        g, box = synth_streak_on(g, rng, style=streak_style)
        streak_gts.append([max(0, box[0]), max(0, box[1]),
                           min(IMG_W, box[2]), min(IMG_H, box[3])])
    return np.repeat(g[:, :, None], 3, axis=2), streak_gts, poison_gts
