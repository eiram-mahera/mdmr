from __future__ import annotations
import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

from cellpose import models, train

logger = logging.getLogger(__name__)

# ------------------------------ training for a fixed number of epochs ----------------------------- #
def train_fixed(
    *,
    device,
    model_type: str,
    channels: List[int],
    images: List[np.ndarray],
    masks: List[np.ndarray],
    epochs: int,
    lr: float,
    wd: float,
):
    """
    Train a Cellpose model for a fixed number of epochs on the provided (images, masks).

    Returns
    -------
    model : cellpose.models.CellposeModel
        Trained model instance (ready for .eval(...)).
    """
    model = models.CellposeModel(device=device, gpu=True, model_type=model_type)

    # cellpose.train.train_seg writes checkpoints if save_path!=None; we keep it in-memory.
    logger.info(f"Training {model_type} model on {len(images)} images")
    _ckpt, _tr_losses, _te_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=masks,
        test_data=None,
        test_labels=None,
        channels=channels,
        n_epochs=int(epochs),
        save_path=None,
        learning_rate=float(lr),
        weight_decay=float(wd),
        SGD=True,
        model_name="run",
        min_train_masks=0,
    )
    return model


def _iou(A: np.ndarray, B: np.ndarray) -> float:
    inter = np.logical_and(A, B).sum(dtype=np.float64)
    if inter == 0:
        return 0.0
    union = np.logical_or(A, B).sum(dtype=np.float64)
    return float(inter / union) if union > 0 else 0.0


def _seg_per_image(pred: np.ndarray, gt: np.ndarray, iou_thresh: float = 0.5) -> float:
    """
    CTC SEG for a single image (Python version used during CV):
      For each GT object g>0, take max IoU(g, any P>0). If max IoU >= thresh, count it; else 0.
      Average over all GT objects in the image.
    """
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]
    if gt_ids.size == 0:
        return 0.0

    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids != 0]

    score_sum = 0.0
    for gid in gt_ids:
        gm = (gt == gid)
        best = 0.0
        for pid in pred_ids:
            pm = (pred == pid)
            i = _iou(gm, pm)
            if i > best:
                best = i
                if best == 1.0:
                    break
        score_sum += (best if best >= iou_thresh else 0.0)
    return float(score_sum / gt_ids.size)


def _seg_python_batch(preds: List[np.ndarray], gts: List[np.ndarray], iou_thresh: float = 0.5) -> float:
    if not preds:
        return 0.0
    vals = [_seg_per_image(p, g, iou_thresh=iou_thresh) for p, g in zip(preds, gts)]
    return float(np.mean(vals))


# ------------------------------ CV epoch search ----------------------------- #

def select_epochs_cv(
    *,
    images: List[np.ndarray],
    masks: List[np.ndarray],
    epoch_grid: List[int],
    k: int,                             # 0 => LOOCV; else KFold with n_splits=k
    device,
    model_type: str,
    channels: List[int],
    lr: float,
    wd: float,
    seed: int = 1,
    use_python_metric: bool = True,     # keep True: much faster than calling external tool per fold
    iou_thresh: float = 0.5,
) -> Tuple[int, Dict[int, List[float]]]:
    """
    Run LOOCV (k==0) or K-fold CV to pick the best epoch count from `epoch_grid`.

    Returns
    -------
    best_epochs : int
        Epoch count with highest mean validation score (ties broken by smaller epochs).
    scores : dict[int, list[float]]
        Per-epoch list of fold scores (for logging/repro).
    """
    n = len(images)
    if n < 2:
        # Not enough data for CV; return the smallest epoch and empty scores.
        return int(min(epoch_grid)), {ep: [] for ep in epoch_grid}

    # Build fold indices
    if k == 0:
        # LOOCV
        fold_pairs = list(LeaveOneOut().split(range(n)))
    else:
        k = max(2, min(k, n))
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        fold_pairs = list(kf.split(range(n)))

    scores: Dict[int, List[float]] = {int(ep): [] for ep in epoch_grid}

    for ep in epoch_grid:
        fold_no = 0
        for tr_idx, val_idx in fold_pairs:
            fold_no += 1
            tr = list(tr_idx)
            va = list(val_idx)

            tr_imgs = [images[i] for i in tr]
            tr_msks = [masks[i] for i in tr]
            va_imgs = [images[i] for i in va]
            va_msks = [masks[i] for i in va]

            # Train a fresh model for this fold/epoch
            model = models.CellposeModel(device=device, gpu=True, model_type=model_type)
            _ckpt, _tr, _te = train.train_seg(
                model.net,
                train_data=tr_imgs,
                train_labels=tr_msks,
                test_data=None,
                test_labels=None,
                channels=channels,
                n_epochs=int(ep),
                save_path=None,
                learning_rate=float(lr),
                weight_decay=float(wd),
                SGD=True,
                model_name="cv",
                min_train_masks=0,
            )

            # Validate on held-out image(s)
            diam = model.net.diam_labels.item()
            preds, _, _ = model.eval(va_imgs, diameter=diam, channels=channels, flow_threshold=None)

            # During CV we score in-Python for speed/stability
            score = _seg_python_batch(preds, va_msks, iou_thresh=iou_thresh)
            scores[int(ep)].append(float(score))

            logger.info(f"[CV] epochs={int(ep):4d} fold {fold_no}/{len(fold_pairs)} → {score:.4f}")

    # Pick best by mean; break ties by smaller epoch (faster model)
    means = {int(ep): (float(np.nanmean(vals)) if vals else -np.inf) for ep, vals in scores.items()}
    best = min(means.keys(), key=lambda e: (-means[e], e))

    logger.info("[CV] epoch → mean ± std (n folds)")
    for ep in sorted(scores.keys()):
        vals = scores[ep]
        if vals:
            logger.info(f"  {ep:4d}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")
        else:
            logger.info(f"  {ep:4d}: (no folds)")
    logger.info(f"[CV] selected epochs = {best}")

    return int(best), scores
