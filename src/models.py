"""Detectron2 model builders and the supervised-repair trainer.

Repair training (the v4 breakthrough, LB 250 -> 206): fine-tune ONLY the
classification head on composites where transplanted poison is unlabeled
("this is background") and synthetic streaks are labeled ("keep detecting
these"). Retention labels prevent the global suppression that made blind
empty-label unlearning non-discriminative. Boxes stay bit-identical to the
poisoned model's, so the survival differential is pure classification change.

detectron2 is imported lazily so the rest of the package works CPU-side.
"""
import copy
from pathlib import Path

import numpy as np
import torch

from .config import (ANCHOR_ASPECT_RATIOS, ANCHOR_SIZES, BASE_CONFIG, BATCH_SIZE,
                     CFG, IMG_H, IMG_W, NUM_CLASSES, POISONED_WEIGHTS, WORK)
from . import data as D


def build_base_cfg(weights, output_dir=None, score_thresh=0.2):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    if output_dir is not None:
        cfg.OUTPUT_DIR = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def build_predictor(weights, score_thresh):
    from detectron2.engine import DefaultPredictor
    return DefaultPredictor(build_base_cfg(weights, score_thresh=score_thresh))


def predict_paths(weights, files, score_thresh):
    """{stem: (boxes Nx4 xyxy, scores N)} over image paths."""
    predictor = build_predictor(weights, score_thresh)
    preds = {}
    for p in files:
        inst = predictor(D.load_image(p))["instances"].to("cpu")
        preds[p.stem] = (inst.pred_boxes.tensor.numpy().astype(np.float32),
                         inst.scores.numpy().astype(np.float32))
    return preds


def predict_arrays(weights, images, score_thresh):
    predictor = build_predictor(weights, score_thresh)
    out = []
    for im in images:
        inst = predictor(im)["instances"].to("cpu")
        out.append((inst.pred_boxes.tensor.numpy().astype(np.float32),
                    inst.scores.numpy().astype(np.float32)))
    return out


def make_repair_trainer_cls(unlearn_imgs, poison_patches, poison_boxes_by_stem,
                            streak_style=None):
    """Factory returning a DefaultTrainer subclass whose mapper builds a fresh
    labeled composite every step and whose model trains cls head only."""
    from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_train_loader
    from detectron2.data import detection_utils as utils
    from detectron2.engine import DefaultTrainer
    from detectron2.structures import BoxMode

    class RepairMapper(DatasetMapper):
        def __call__(self, dd):
            dd = copy.deepcopy(dd)
            rng = np.random.default_rng()
            img, streak_gts, _ = D.make_composite(
                rng, unlearn_imgs, poison_patches, poison_boxes_by_stem,
                stem=Path(dd["file_name"]).stem, streak_style=streak_style)
            anns = [{"bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                     "bbox_mode": BoxMode.XYWH_ABS, "category_id": 0} for b in streak_gts]
            dd["image"] = torch.as_tensor(img.transpose(2, 0, 1).copy())
            dd["instances"] = utils.annotations_to_instances(anns, img.shape[:2])
            return dd

    class RepairTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg):
            return build_detection_train_loader(
                cfg, mapper=RepairMapper(cfg, is_train=True, augmentations=[]),
                dataset=DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))

        @classmethod
        def build_model(cls, cfg):
            model = super().build_model(cfg)
            for name, p in model.named_parameters():
                p.requires_grad = ("cls_score" in name or "cls_subnet" in name)
            return model

    return RepairTrainer


def train_repair_variant(name, lr, iters, dataset_name, trainer_cls, seed=42):
    out = WORK / name
    final = out / "model_final.pth"
    if final.exists():
        return str(final)
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = build_base_cfg(POISONED_WEIGHTS, output_dir=out)
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = iters
    cfg.SOLVER.STEPS = []
    trainer = trainer_cls(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return str(final)
