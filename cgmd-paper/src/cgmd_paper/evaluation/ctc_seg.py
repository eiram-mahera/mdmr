import os, re, shutil, subprocess, logging
from typing import List, Tuple
import numpy as np
import tifffile
from .registry import register_metric

logger = logging.getLogger(__name__)

def _parse_stdout(stdout: str) -> float:
    try:
        logger.info(stdout)
        return float(stdout.split(":")[-1].strip())
    except Exception:
        return float("nan")


def _modify_segmentation_mask(mask: np.ndarray, erosion_pixels: int = 50) -> np.ndarray:
    """
    Keep labels that remain visible after eroding 'erosion_pixels' from each image border.
    """
    h, w = mask.shape[:2]
    if erosion_pixels <= 0 or erosion_pixels * 2 >= min(h, w):
        return mask
    y1, y2 = erosion_pixels, h - erosion_pixels
    x1, x2 = erosion_pixels, w - erosion_pixels
    visible = np.unique(mask[y1:y2, x1:x2])
    return np.where(np.isin(mask, visible), mask, 0)


def _ensure_uint16_2d(arr: np.ndarray) -> np.ndarray:
    """Force 2D single-plane uint16 label image."""
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16, copy=False)
    return arr


def _detect_num_seq(gt_seg_dir: str, fallback: int = 3) -> int:
    """Infer zero padding from GT filenames (man_seg000.tif vs man_seg0000.tif)."""
    names = sorted([f for f in os.listdir(gt_seg_dir) if f.startswith("man_seg") and f.endswith(".tif")])
    if not names:
        return fallback
    m = re.match(r"man_seg(0+)\d+\.tif$", names[0])
    return (len(m.group(1)) + 1) if m else fallback


def _write_preds_ctc_res(
    *, preds: List[np.ndarray], ds, video: str, dataset_root: str, require_uncrop: bool, erosion_px: int
) -> Tuple[str, int]:
    """
    Write predictions into <video>_RES and return (res_dir, num_seq).
    """
    res_dir = os.path.join(dataset_root, f"{video}_RES")
    os.makedirs(res_dir, exist_ok=True)

    gt_dir = os.path.join(dataset_root, f"{video}_GT", "SEG")
    if not os.path.isdir(gt_dir):
        logger.error(f"Missing GT SEG dir: {gt_dir}")
        raise FileNotFoundError(f"Missing GT SEG dir: {gt_dir}")

    num_seq = _detect_num_seq(gt_dir, fallback=3)

    for i, p in enumerate(preds):
        if require_uncrop:
            p = ds.uncrop_like_original(p, video=video, idx=i)
        if erosion_px:
            p = _modify_segmentation_mask(p, erosion_pixels=erosion_px)
        p = _ensure_uint16_2d(p)
        name = f"{i:0{num_seq}d}"
        tifffile.imwrite(os.path.join(res_dir, f"mask{name}.tif"), p)

    return res_dir, num_seq


# ---------- Pure-Python SEG fallback ----------
def _iou(A: np.ndarray, B: np.ndarray) -> float:
    inter = np.logical_and(A, B).sum(dtype=np.float64)
    if inter == 0:
        return 0.0
    union = np.logical_or(A, B).sum(dtype=np.float64)
    return float(inter / union) if union > 0 else 0.0


def _seg_per_image(pred: np.ndarray, gt: np.ndarray, iou_thresh: float = 0.5) -> float:
    """
    CTC SEG for one image:
      For each GT object g in G (labels>0), take IoU(g, best P) if >= threshold else 0.
      Average over GT objects. If no GT objects, return 0 by convention.
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
            best = max(best, _iou(gm, pm))
            # early exit if perfect
            if best == 1.0:
                break
        score_sum += (best if best >= iou_thresh else 0.0)

    return float(score_sum / gt_ids.size)


def _seg_python(
    *, dataset_root: str, video: str, preds: List[np.ndarray], require_uncrop: bool, ds
) -> float:
    """
    Compute SEG across the whole video in Python (fallback).
    Loads GT from <dataset_root>/<video>_GT/SEG, uncrops preds if needed.
    """
    gt_dir = os.path.join(dataset_root, f"{video}_GT", "SEG")
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".tif")])
    if len(gt_files) != len(preds):
        # we still try to align by index; truncate to min length
        n = min(len(gt_files), len(preds))
    else:
        n = len(preds)

    total = 0.0
    for i in range(n):
        gt = tifffile.imread(os.path.join(gt_dir, gt_files[i]))
        P = preds[i]
        if require_uncrop:
            P = ds.uncrop_like_original(P, video=video, idx=i)
        total += _seg_per_image(_ensure_uint16_2d(P), _ensure_uint16_2d(gt), iou_thresh=0.5)

    return float(total / max(n, 1))


@register_metric("CTC_SEG")
def ctc_seg_score(
    *,
    ctc_app: str,
    dataset_root: str,
    video: str,
    num_seq: int,  # can be ignored when auto-detected
    preds: List[np.ndarray],
    require_uncrop: bool,
    ds,
    erosion_px: int,
) -> float:
    """
    Try the official SEGMeasure tool first. If it fails (e.g., segmentation fault),
    print a message and fall back to a Python implementation of the CTC SEG measure.
    """
    # 1) Write predictions in CTC format
    res_dir = None
    try:
        res_dir, inferred_num_seq = _write_preds_ctc_res(
            preds=preds,
            ds=ds,
            video=video,
            dataset_root=dataset_root,
            require_uncrop=require_uncrop,
            erosion_px=erosion_px,
        )

        # Prefer the inferred padding if caller provided something strange
        nseq = inferred_num_seq if inferred_num_seq is not None else num_seq

        # 2) Run official tool
        logger.info(f"Command: {ctc_app} {dataset_root} {video} {nseq}")
        cmd = f"{ctc_app} {dataset_root} {video} {nseq}"
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)

        # Detect failure: non-zero code OR 'Segmentation fault' OR parse failure
        stdout, stderr = proc.stdout, proc.stderr
        seg = _parse_stdout(stdout)
        failed = (
            proc.returncode != 0
            or ("Segmentation fault" in (stderr or ""))  # typical segfault message
            or np.isnan(seg)
        )

        if not failed:
            return seg

        # 3) Fallback: Python SEG
        logger.warning("[CTC] SEGMeasure failed; falling back to Python SEG implementation.")
        return _seg_python(
            dataset_root=dataset_root, video=video, preds=preds, require_uncrop=require_uncrop, ds=ds
        )

    finally:
        if res_dir is not None:
            shutil.rmtree(res_dir, ignore_errors=True)
