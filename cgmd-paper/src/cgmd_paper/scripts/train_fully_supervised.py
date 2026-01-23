from __future__ import annotations

import csv
import os
import sys
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import torch
import typer
import yaml
from sklearn.model_selection import train_test_split, KFold

from cellpose import models, train

# Project modules (package should be installed/editable: `pip install -e .`)
from ..datasets.ctc_dataset import CTCDataset
from ..evaluation import compute  # registers "CTC_SEG"
from ..utils.logging import setup_logger


app = typer.Typer(add_completion=False)
logger = setup_logger("fully_supervised")

# ----------------------------- helpers ----------------------------- #

def _project_root() -> str:
    # <repo>/cgmd-paper/src/cgmd_paper/scripts/fully_supervised.py -> <repo>/cgmd-paper
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _channels_from_cfg(ds_cfg: dict) -> List[int]:
    ch = ds_cfg.get("channels", None)
    if ch is None:
        raise typer.BadParameter("Dataset YAML must define 'dataset.channels' (e.g., [0, 0]).")
    if isinstance(ch, (list, tuple)) and len(ch) == 2:
        return [int(ch[0]), int(ch[1])]
    if isinstance(ch, str):
        parts = [p.strip() for p in ch.split(",") if p.strip()]
        if len(parts) == 2:
            return [int(parts[0]), int(parts[1])]
    raise typer.BadParameter("Invalid 'dataset.channels'. Use a list like [0, 0] or string '0,0'.")


def _append_row(csv_path: str, header: List[str], row: List):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def _ensure_uint16_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr.astype(np.uint16, copy=False)


# ---------- Fast Python SEG (for CV only; final test uses official wrapper) ---------- #

def _iou(A: np.ndarray, B: np.ndarray) -> float:
    inter = np.logical_and(A, B).sum(dtype=np.float64)
    if inter == 0:
        return 0.0
    union = np.logical_or(A, B).sum(dtype=np.float64)
    return float(inter / union) if union > 0 else 0.0


def _seg_per_image(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    gt_ids = np.unique(gt); gt_ids = gt_ids[gt_ids != 0]
    if gt_ids.size == 0:
        return 0.0
    pred_ids = np.unique(pred); pred_ids = pred_ids[pred_ids != 0]
    s = 0.0
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
        s += (best if best >= thr else 0.0)
    return float(s / gt_ids.size)


def _seg_python_batch(preds: List[np.ndarray], gts: List[np.ndarray], thr: float = 0.5) -> float:
    if not preds:
        return 0.0
    vals = [_seg_per_image(_ensure_uint16_2d(p), _ensure_uint16_2d(g), thr=thr) for p, g in zip(preds, gts)]
    return float(np.mean(vals))


# ------------------------ Core train/eval primitives ------------------------ #

def _new_model(device, model_type: str):
    return models.CellposeModel(device=device, gpu=(device.type == "cuda"), model_type=model_type)


def _train_for_epochs(
    *,
    device,
    model_type: str,
    channels: List[int],
    images: List[np.ndarray],
    masks: List[np.ndarray],
    epochs: int,
    lr: float,
    wd: float,
    val_images: List[np.ndarray] | None = None,
    val_masks: List[np.ndarray] | None = None,
):
    logger.info(f"Training the model {model_type} for {epochs} epochs")
    """Train for a fixed number of epochs. Returns (model, train_losses, val_losses)."""
    model = _new_model(device, model_type)
    test_data = val_images if val_images is not None else None
    test_labels = val_masks if val_masks is not None else None
    _ckpt, tr_losses, te_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=masks,
        test_data=test_data,
        test_labels=test_labels,
        channels=channels,
        n_epochs=int(epochs),
        save_path=None,
        learning_rate=float(lr),
        weight_decay=float(wd),
        SGD=True,
        model_name="run",
        min_train_masks=0,
    )
    logger.info("Training complete.")
    return model, tr_losses, te_losses


def _eval_on_video02(
    *,
    model,
    test_images: List[np.ndarray],
    channels: List[int],
    dataset_root: str,
    num_seq: int,
    require_uncrop: bool,
    ds: CTCDataset,
    erosion_px: int,
    ctc_app: str,
) -> float:
    diam = model.net.diam_labels.item()
    logger.info(f"Begin evaluation")
    preds, _, _ = model.eval(test_images, diameter=diam, channels=channels, flow_threshold=None)
    seg = compute(
        "CTC_SEG",
        ctc_app=ctc_app,
        dataset_root=dataset_root,
        video="02",
        num_seq=num_seq,
        preds=preds,
        require_uncrop=require_uncrop,
        ds=ds,
        erosion_px=erosion_px,
    )
    return float(seg)


# ------------------------------ Training modes ----------------------------- #

def run_fully_supervised_fixed(
    *,
    device, model_type, channels, lr, wd,
    train_images, train_masks,
    test_images,
    ds: CTCDataset, dataset_root: str, num_seq: int, require_uncrop: bool, erosion_px: int, ctc_app: str,
    epochs: int,
) -> Tuple[int, float]:
    model, _, _ = _train_for_epochs(
        device=device, model_type=model_type, channels=channels,
        images=train_images, masks=train_masks, epochs=epochs, lr=lr, wd=wd
    )
    seg = _eval_on_video02(
        model=model, test_images=test_images, channels=channels,
        dataset_root=dataset_root, num_seq=num_seq, require_uncrop=require_uncrop,
        ds=ds, erosion_px=erosion_px, ctc_app=ctc_app,
    )
    return int(epochs), seg


def run_early_stopping_8020(
    *,
    device, model_type, channels, lr, wd,
    train_images, train_masks,
    test_images,
    ds: CTCDataset, dataset_root: str, num_seq: int, require_uncrop: bool, erosion_px: int, ctc_app: str,
    max_epochs: int = 200, start_epochs: int = 20, patience: int = 10,
) -> Tuple[int, float]:
    """
    Strategy:
      1) Split 80/20 on video 01 (stratification not applicable for instance seg).
      2) Train ONCE for `max_epochs` while logging validation loss each epoch.
      3) Pick best epoch using 'min val loss' with patience (earliest epoch meeting rule, but >= start_epochs).
      4) Retrain from scratch for `best_epoch` on ALL train images (full 01).
      5) Evaluate on video 02.
    """
    imgs_tr, imgs_val, msks_tr, msks_val = train_test_split(
        train_images, train_masks, test_size=0.2, shuffle=True, random_state=0
    )

    # Train once to collect per-epoch validation losses
    _, _, val_losses = _train_for_epochs(
        device=device, model_type=model_type, channels=channels,
        images=imgs_tr, masks=msks_tr, epochs=max_epochs, lr=lr, wd=wd,
        val_images=imgs_val, val_masks=msks_val,
    )
    # Choose best epoch via patience
    best_loss = float("inf")
    best_epoch = start_epochs  # ensure we don't pick < start_epochs
    no_improve = 0
    for ep, v in enumerate(val_losses, start=1):
        if ep < start_epochs:
            continue
        if v < best_loss - 1e-8:
            best_loss = v
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Retrain from scratch on ALL train images for best_epoch
    model, _, _ = _train_for_epochs(
        device=device, model_type=model_type, channels=channels,
        images=train_images, masks=train_masks, epochs=best_epoch, lr=lr, wd=wd
    )
    seg = _eval_on_video02(
        model=model, test_images=test_images, channels=channels,
        dataset_root=dataset_root, num_seq=num_seq, require_uncrop=require_uncrop,
        ds=ds, erosion_px=erosion_px, ctc_app=ctc_app,
    )
    return int(best_epoch), seg


def run_kfold_cv_select_epochs(
    *,
    device, model_type, channels, lr, wd,
    train_images, train_masks,
    test_images,
    ds: CTCDataset, dataset_root: str, num_seq: int, require_uncrop: bool, erosion_px: int, ctc_app: str,
    kfolds: int = 5, cv_start: int = 10, cv_max: int = 200, cv_step: int = 10, iou_thresh: float = 0.5,
) -> Tuple[int, float]:
    """
    Strategy:
      1) Build epoch grid: cv_start..cv_max step cv_step.
      2) KFold on video 01. For each epoch in grid:
           - train on fold-train for 'epoch' epochs,
           - eval on fold-val with FAST Python SEG (no external tool),
         Aggregate mean score across folds.
      3) Pick best epoch (tie-break: smaller epoch).
      4) Retrain from scratch on ALL train images for best_epoch.
      5) Evaluate on video 02 with official wrapper.
    """
    kfolds = max(2, min(kfolds, len(train_images)))
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=0)
    epoch_grid = list(range(int(cv_start), int(cv_max) + 1, int(cv_step)))
    scores: Dict[int, List[float]] = {ep: [] for ep in epoch_grid}

    for ep in epoch_grid:
        fold_no = 0
        for tr_idx, va_idx in kf.split(range(len(train_images))):
            fold_no += 1
            tr_idx = list(tr_idx); va_idx = list(va_idx)
            tr_imgs = [train_images[i] for i in tr_idx]
            tr_msks = [train_masks[i] for i in tr_idx]
            va_imgs = [train_images[i] for i in va_idx]
            va_msks = [train_masks[i] for i in va_idx]

            model, _, _ = _train_for_epochs(
                device=device, model_type=model_type, channels=channels,
                images=tr_imgs, masks=tr_msks, epochs=ep, lr=lr, wd=wd
            )
            diam = model.net.diam_labels.item()
            preds, _, _ = model.eval(va_imgs, diameter=diam, channels=channels, flow_threshold=None)
            sc = _seg_python_batch(preds, va_msks, thr=iou_thresh)
            scores[ep].append(float(sc))
            print(f"[KFold] ep={ep:4d} fold {fold_no}/{kfolds} → SEG(py)={sc:.4f}")

    means = {ep: (np.mean(v) if v else -np.inf) for ep, v in scores.items()}
    best_epoch = min(epoch_grid, key=lambda e: (-means[e], e))
    print("[KFold] epoch → mean SEG(py):")
    for ep in epoch_grid:
        if scores[ep]:
            print(f"  {ep:4d}: {np.mean(scores[ep]):.4f} ± {np.std(scores[ep]):.4f} (n={len(scores[ep])})")
    print(f"[KFold] selected epochs = {best_epoch}")

    # Retrain and test on 02
    model, _, _ = _train_for_epochs(
        device=device, model_type=model_type, channels=channels,
        images=train_images, masks=train_masks, epochs=best_epoch, lr=lr, wd=wd
    )
    seg = _eval_on_video02(
        model=model, test_images=test_images, channels=channels,
        dataset_root=dataset_root, num_seq=num_seq, require_uncrop=require_uncrop,
        ds=ds, erosion_px=erosion_px, ctc_app=ctc_app,
    )
    return int(best_epoch), seg


# ----------------------------------- CLI ----------------------------------- #

@app.command()
def run(
    # dataset & model
    dataset_config: str = typer.Option(..., help="YAML with dataset-specific settings"),
    model_type: str = typer.Option(..., help="Cellpose model type (e.g., cyto2, cyto3)"),

    # mode
    mode: str = typer.Option("fixed", help="Training mode: 'fixed', 'early' (early stopping 80/20), or 'kfold'"),

    # fixed
    epochs: int = typer.Option(200, help="Used when mode='fixed'"),

    # Early stopping
    early_max: int = typer.Option(200, help="Max epochs for early stopping run"),
    early_start: int = typer.Option(20, help="Do not stop before this epoch"),
    early_patience: int = typer.Option(10, help="Patience (epochs without val improvement)"),

    # K-Fold CV
    kfolds: int = typer.Option(5, help="Number of folds for K-fold CV"),
    cv_start: int = typer.Option(10, help="CV grid start epoch"),
    cv_max: int = typer.Option(200, help="CV grid max epoch (inclusive)"),
    cv_step: int = typer.Option(10, help="CV grid step"),

    # optimization
    lr: float = typer.Option(1e-1, help="Learning rate"),
    wd: float = typer.Option(1e-4, help="Weight decay"),

    # runs / reproducibility
    runs: int = typer.Option(1, help="Number of repeated runs with seeds seed..seed+runs-1"),
    seed: int = typer.Option(42, help="Base seed"),
    cuda: int = typer.Option(0, help="CUDA device index"),

    # outputs
    results_csv: str = typer.Option("results_fs.csv", help="CSV to append results"),
    ctc_app: str = typer.Option("", help="Path to SEGMeasure (default: <project_root>/tools/SEGMeasure)"),
):
    """
    Fully-supervised evaluation on CTC datasets with three modes:
      - fixed:     train all '01' for fixed epochs, test on '02'
      - early:  train with early stopping on 80/20 split of '01' (by val loss), retrain on full '01' to best epoch, test on '02'
      - kfold:  select epochs via K-fold CV on '01' (Python SEG), retrain on full '01' to best epoch, test on '02'
    Always scores '02' using CTC SEG (with safe fallback).
    """
    logger.info("Starting fully supervised training")

    # --- config ---
    with open(dataset_config, "r") as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg.get("dataset", {})
    ev_cfg = cfg.get("eval", {})
    logger.debug(f"Dataset config: {ds_cfg}")
    logger.debug(f"Evaluation config: {ev_cfg}")

    data_dir = Path(ds_cfg["data_dir"]).expanduser()
    dataset_name = ds_cfg["dataset_name"]
    crop_boxes = ds_cfg.get("crop_boxes", {})
    channels = _channels_from_cfg(ds_cfg)

    num_seq = int(ev_cfg["num_seq"])
    erosion_px = int(ev_cfg.get("erosion_pixels", 0))
    require_uncrop = bool(ev_cfg.get("require_uncrop", False))

    # ctc app
    if not ctc_app:
        ctc_app = os.path.join(_project_root(), "tools", "SEGMeasure")
    if not os.path.exists(ctc_app):
        logger.warning(f"SEGMeasure not found at {ctc_app}. Fallback to python implementation")
        typer.echo(f"[WARN] SEGMeasure not found at {ctc_app}. Fallback to python implementation", err=True)

    # device & seeds
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    os.environ["PYTHONHASHSEED"] = str(seed)

    header = ["Mode", "Dataset", "Model", "TrainVideo", "TestVideo", "Run", "Seed", "Epochs", "SEG"]

    # dataset
    ds = CTCDataset(data_dir=data_dir, dataset_name=dataset_name, crop_boxes=crop_boxes or None)
    dataset_root = os.path.join(data_dir, dataset_name)

    # preload
    train_images = ds.load_images(ds.unlabeled_indices, video="01")  # all frames in 01
    train_masks  = ds.load_masks(ds.unlabeled_indices, video="01")
    test_images  = ds.load_images(ds.test_indices, video="02")

    mode_l = mode.lower()
    if mode_l not in ("fixed", "early", "kfold"):
        logger.error("mode must be one of: fixed, early, kfold")
        raise typer.BadParameter("mode must be one of: fixed, early, kfold")

    for r in range(runs):
        run_seed = seed + r
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        if mode_l == "fixed":
            best_epochs, seg = run_fully_supervised_fixed(
                device=device, model_type=model_type, channels=channels, lr=lr, wd=wd,
                train_images=train_images, train_masks=train_masks,
                test_images=test_images,
                ds=ds, dataset_root=dataset_root, num_seq=num_seq, require_uncrop=require_uncrop,
                erosion_px=erosion_px, ctc_app=ctc_app,
                epochs=epochs,
            )
        elif mode_l == "early":
            best_epochs, seg = run_early_stopping_8020(
                device=device, model_type=model_type, channels=channels, lr=lr, wd=wd,
                train_images=train_images, train_masks=train_masks,
                test_images=test_images,
                ds=ds, dataset_root=dataset_root, num_seq=num_seq, require_uncrop=require_uncrop,
                erosion_px=erosion_px, ctc_app=ctc_app,
                max_epochs=early_max, start_epochs=early_start, patience=early_patience,
            )
        else:  # kfold
            best_epochs, seg = run_kfold_cv_select_epochs(
                device=device, model_type=model_type, channels=channels, lr=lr, wd=wd,
                train_images=train_images, train_masks=train_masks,
                test_images=test_images,
                ds=ds, dataset_root=dataset_root, num_seq=num_seq, require_uncrop=require_uncrop,
                erosion_px=erosion_px, ctc_app=ctc_app,
                kfolds=kfolds, cv_start=cv_start, cv_max=cv_max, cv_step=cv_step, iou_thresh=0.5,
            )

        row = [mode_l, dataset_name, model_type, "01", "02", r + 1, run_seed, int(best_epochs), float(seg)]
        _append_row(results_csv, header, row)
        logger.info(f"[{mode_l.upper()} RUN {r+1}/{runs}] epochs={best_epochs} SEG={seg:.4f} → {results_csv}")


if __name__ == "__main__":
    app()
