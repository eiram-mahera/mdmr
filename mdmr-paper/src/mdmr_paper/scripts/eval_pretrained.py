# mdmr-paper/src/scripts/eval_pretrained.py
from __future__ import annotations

import csv
import os
import sys
from typing import List
from pathlib import Path

import numpy as np
import torch
import typer
import yaml
from cellpose import models

# Project imports (package must be installed/editable: `pip install -e .`)
from ..datasets.ctc_dataset import CTCDataset
from ..evaluation import compute  # registers "CTC_SEG"


app = typer.Typer(add_completion=False)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _parse_channels(spec: str | List[int]) -> List[int]:
    """Parse channels from '0,0' or list of ints."""
    if isinstance(spec, list):
        return [int(x) for x in spec]
    s = str(spec).strip()
    if not s:
        raise typer.BadParameter("channels cannot be empty. Example: --channels 0,0")
    try:
        vals = [int(x) for x in s.split(",")]
    except ValueError:
        raise typer.BadParameter("channels must be integers, e.g., --channels 0,0")
    if len(vals) != 2:
        raise typer.BadParameter("channels must have exactly two integers, e.g., --channels 0,0")
    return vals


def _append_row(csv_path: str, header: List[str], row: List):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


@app.command()
def run(
    dataset_config: str = typer.Option(..., help="YAML with dataset-specific settings"),
    model_type: str = typer.Option(..., help="Cellpose model type (e.g., cyto2, cyto3)"),
    cuda: int = typer.Option(0, help="CUDA device index (ignored if no GPU)"),
    seed: int = typer.Option(42, help="Random seed (for determinism where applicable)"),
    results_csv: str = typer.Option("results_pretrained.csv", help="CSV to append results"),
    save_preds_dir: str = typer.Option("", help="Optional: write predictions as TIFFs"),
    ctc_app: str = typer.Option("", help="Path to SEGMeasure tool (defaults to <project_root>/tools/SEGMeasure)"),
):
    """
    Evaluate a pretrained Cellpose model on the test video and write CTC SEG to CSV.

    The dataset config YAML should include:
      dataset:
        data_dir: /path/to/CTC
        dataset_name: Fluo-N2DH-SIM+
        crop_boxes: {}             # optional per-video crops
      eval:
        num_seq: 3                 # zero padding for filenames
        erosion_pixels: 0          # optional mask erosion for scoring
        require_uncrop: false      # true for BF datasets that need uncrop
    """
    # --- config ---
    with open(dataset_config, "r") as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg.get("dataset", {})
    ev_cfg = cfg.get("eval", {})

    data_dir = Path(ds_cfg["data_dir"]).expanduser()
    dataset_name = ds_cfg["dataset_name"]
    crop_boxes = ds_cfg.get("crop_boxes", {})

    num_seq = int(ev_cfg["num_seq"])
    erosion_px = int(ev_cfg.get("erosion_pixels", 0))
    require_uncrop = bool(ev_cfg.get("require_uncrop", False))
    ch = ds_cfg.get("channels", [0,0])

    # --- device & seeds ---
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # --- dataset ---
    ds = CTCDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        crop_boxes=crop_boxes or None,
    )
    dataset_root = os.path.join(data_dir, dataset_name)

    test_images = ds.load_images(ds.test_indices, video="02")

    # --- model ---
    model = models.CellposeModel(device=device, gpu=torch.cuda.is_available(), model_type=model_type)
    diam = model.net.diam_labels.item()
    preds, _, _ = model.eval(test_images, diameter=diam, channels=ch, flow_threshold=None)

    if save_preds_dir:
        import tifffile
        out_dir = os.path.join(save_preds_dir, dataset_name, "02_RES")
        os.makedirs(out_dir, exist_ok=True)
        for i, p in enumerate(preds):
            tifffile.imwrite(os.path.join(out_dir, f"mask{i:0{num_seq}d}.tif"), p.astype(np.uint16, copy=False))
        print(f"[INFO] Wrote {len(preds)} predictions to: {out_dir}")

    # --- CTC tool path ---
    if not ctc_app:
        ctc_app = os.path.join(_project_root(), "tools", "SEGMeasure")
    if not os.path.exists(ctc_app):
        typer.echo(f"[ERROR] SEGMeasure not found at {ctc_app}", err=True)

    # --- score ---
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

    # --- results ---
    header = ["Dataset", "Model", "Video", "Seed", "SEG"]
    row = [dataset_name, model_type, "02", seed, seg]
    _append_row(results_csv, header, row)

    print(f"[DONE] Dataset={dataset_name} Model={model_type} Video=02 SEG={seg:.4f}")
    print(f"[CSV]  Appended to {results_csv}")


if __name__ == "__main__":
    app()
