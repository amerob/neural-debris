"""Pipeline constants. Mirrors the Kaggle notebook's CFG."""
from pathlib import Path

ROOT = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"
POISONED_WEIGHTS = f"{ROOT}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR = f"{ROOT}/unlearn_set"
TEST_DIR = f"{ROOT}/test_set/test_set"
SAMPLE_SUB = f"{ROOT}/sample_submission.csv"

BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1
IMG_W = IMG_H = 1024
BATCH_SIZE = 4
SEED = 42

WORK = Path("/kaggle/working")
CACHE = WORK / "cache"

CFG = dict(
    CAND_THRESH=0.05,        # poisoned-model candidate threshold (recall of clean detections)
    ENS_THRESH=0.01,         # repair-ensemble inference threshold
    MATCH_IOU=0.70,          # cls-only members -> boxes identical -> near-exact matching
    W_DIFF=0.75, W_GEO=0.05, W_APP=0.20,
    USE_APPEARANCE=True,
    TRANSPLANT_FIRE_GATE=0.5,
    N_REPAIR_RECORDS=200,
    N_CALIB=250,
    TRIAGE_FOLDS=4,
    STREAK_STYLES=dict(       # 'calib' is HELD OUT from all training (leak hygiene)
        base=dict(L=(18, 100), sigma=(0.6, 1.6), amp=(2.5, 12), thick=(1, 1)),
        wide=dict(L=(15, 180), sigma=(0.5, 2.0), amp=(2.0, 14), thick=(1, 2)),
        calib=dict(L=(25, 160), sigma=(0.55, 1.9), amp=(2.2, 13), thick=(1, 2)),
    ),
)

# Pinned best-known leaderboard recipe (LB 206). Never re-tuned by calibration.
V4_RECIPE = dict(P_HI=0.5, P_LO=0.4, MIN_KEEP=0.2, BOOST=False,
                 S_CONF=0.5, C_MAX=0.95, EPS=0.01)
