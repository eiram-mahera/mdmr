# cgmd-paper/src/cgmd_paper/scripts/fine_tune.py
from __future__ import annotations

import os
import sys
import json
import ast
from typing import List, Dict
from pathlib import Path

import typer
import torch
import yaml

from ..utils.io import append_row
from ..utils.seed import set_all_seeds
from ..datasets import build_dataset
from ..features.cellpose_styles import extract_or_load_styles
from ..selectors import loader as sel_loader
from ..evaluation import compute                      # registers metrics
from ..training.cellpose_train import train_fixed, select_epochs_cv
from ..utils.logging import setup_logger

app = typer.Typer(add_completion=False)
logger = setup_logger("fine_tune")

# ----------------------------- helpers ----------------------------- #

def _parse_kv(s: str):
    """Parse --sel-arg key=value with a best-effort value literal."""
    if "=" not in s:
        logger.error("--sel-arg must contain key=value")
        raise typer.BadParameter("Use key=value format for --sel-arg")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    try:
        v_py = ast.literal_eval(v)  # "0.1"->0.1, "true"->True, "[1,2]"->[1,2], etc.
    except Exception:
        v_py = v  # fallback to raw string
    return k, v_py


def _merge_selector_kwargs(sel_args: List[str] | None, json_blob: str | None) -> Dict:
    """Combine JSON kwargs and repeated key=value flags; flags override JSON."""
    kwargs: Dict = {}
    if json_blob:
        try:
            data = json.loads(json_blob)
            if not isinstance(data, dict):
                logger.error(f"Invalid JSON blob: {json_blob}")
                raise ValueError("selector-kwargs must be a JSON object")
            kwargs.update(data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --selector-kwargs: {e}")
            raise typer.BadParameter(f"Invalid JSON in --selector-kwargs: {e}")
    for s in sel_args or []:
        k, v = _parse_kv(s)
        kwargs[k] = v
    return kwargs


def _project_root() -> str:
    # <repo>/cgmd-paper/src/cgmd_paper/scripts/fine_tune.py -> <repo>/cgmd-paper
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _build_epoch_grid(start: int, stop: int, step: int) -> List[int]:
    if start <= 0 or stop <= 0 or step <= 0:
        logger.error("--cv-start, --cv-max, --cv-step must be positive")
        raise typer.BadParameter("--cv-start, --cv-max, --cv-step must be positive")
    if start > stop:
        logger.error("--cv-start must be ≤ --cv-max")
        raise typer.BadParameter("--cv-start must be ≤ --cv-max")
    grid = list(range(start, stop + 1, step))
    if not grid:
        logger.error("epoch grid is empty")
        raise typer.BadParameter("epoch grid is empty")
    return grid


def _channels_from_cfg(ds_cfg: dict) -> List[int]:
    ch = ds_cfg.get("channels", None)
    if ch is None:
        logger.error("Dataset YAML must define 'dataset.channels' (e.g., [0, 0]).")
        raise typer.BadParameter("Dataset YAML must define 'dataset.channels' (e.g., [0, 0]).")
    if isinstance(ch, (list, tuple)) and len(ch) == 2:
        return [int(ch[0]), int(ch[1])]
    if isinstance(ch, str):
        parts = [p.strip() for p in ch.split(",") if p.strip()]
        if len(parts) == 2:
            return [int(parts[0]), int(parts[1])]
    logger.error("Invalid 'dataset.channels'. Use a list like [0, 0] or string '0,0'.")
    raise typer.BadParameter("Invalid 'dataset.channels'. Use a list like [0, 0] or string '0,0'.")


def _available_selectors() -> List[str]:
    """Return list of available selector shortcut names (uppercased)."""
    # Preferred: a function exposed by your loader
    if hasattr(sel_loader, "available_shortcuts") and callable(getattr(sel_loader, "available_shortcuts")):
        return list(getattr(sel_loader, "available_shortcuts")())

    # Fallback: introspect a private dict if present
    shortcuts = getattr(sel_loader, "_SHORTCUTS", None)
    if isinstance(shortcuts, dict):
        return list(shortcuts.keys())

    # Last resort: user must specify fully-qualified module:Class
    return []


def _parse_selector_arg(selector_arg: str) -> List[str]:
    """
    Convert --selector into a list of selector tokens:
      - 'ALL' -> all available shortcuts (error if none known)
      - 'A,B,C' -> ['A','B','C']
      - 'A' -> ['A']
      - fully-qualified names pass through unchanged (no split inside module paths)
        e.g., 'mypkg.mod:MyClass' stays as-is even if it contains commas (unlikely)
    """
    sel = selector_arg.strip()
    if sel.upper() == "ALL":
        names = _available_selectors()
        if not names:
            logger.error("ALL requested, but no shortcut selectors are discoverable.")
            raise typer.BadParameter(
                "ALL requested, but no shortcut selectors are discoverable."
                "Add shortcuts in selectors.loader or pass a comma-separated list."
            )
        return names

    # Split on top-level commas for simple lists of shortcuts
    parts = [p.strip() for p in sel.split(",") if p.strip()]
    return parts if parts else [sel]


# ------------------------------ CLI ------------------------------- #

@app.command()
def run(
    # dataset & model
    dataset_config: str = typer.Option(..., help="YAML with dataset-specific settings"),
    model_type: str = typer.Option("cyto2", help="Cellpose model type (e.g., cyto2, cyto3)"),

    # subset selectors
    selector: str = typer.Option(
        "ALL",
        help="Selector name(s): single (e.g., 'CGMD'), comma-separated "
             "(e.g., 'CGMD,Random'), or 'ALL' for all available shortcuts. "
             "You may also pass a fully-qualified 'module.path:ClassName'."
    ),
    sel_arg: List[str] = typer.Option(None, help="Selector arg as key=value (repeatable)"),
    selector_kwargs: str = typer.Option(None, help="JSON dict of selector kwargs"),
    budget: int = typer.Option(2, help="Query budget (number of training samples to select)"),

    # training mode
    train_mode: str = typer.Option("cv", help="Training mode: 'fixed' or 'cv'"),
    epochs_fixed: int = typer.Option(200, help="Epochs when --train-mode fixed"),
    cv_start: int = typer.Option(10, help="CV grid start epoch"),
    cv_max: int = typer.Option(200, help="CV grid max epoch (inclusive)"),
    cv_step: int = typer.Option(10, help="CV grid step"),
    cv_kfolds: int = typer.Option(0, help="If budget>2, K for K-fold CV. If 0 with budget!=2 -> error. If budget==2 -> LOOCV automatically."),

    # optimization
    lr: float = typer.Option(1e-1, help="Learning rate"),
    wd: float = typer.Option(1e-4, help="Weight decay"),

    # runs / reproducibility
    runs: int = typer.Option(1, help="Number of repeated runs with seeds seed..seed+runs-1"),
    seed: int = typer.Option(42, help="Base seed"),
    cuda: int = typer.Option(0, help="CUDA device index"),

    # features cache
    features_dir: str = typer.Option("./features", help="Directory to cache style features"),
    overwrite_features: bool = typer.Option(False, help="Recompute and overwrite cached features"),

    # CTC tool path
    ctc_app: str = typer.Option("", help="Path to SEGMeasure. Defaults to <project_root>/tools/SEGMeasure"),

    # outputs
    results_csv: str = typer.Option("results.csv", help="CSV to append results"),
):
    """
    Active Learning pipeline:
      1) Load dataset per YAML (channels are read from YAML).
      2) Cache/load Cellpose style features for all unlabeled training images.
      3) For each selector in --selector, pick subset and fine-tune.
      4) Train either with fixed epochs, or via CV:
         - budget==2 -> LOOCV automatically
         - else K-fold CV with --cv-kfolds
      5) Evaluate on test video with CTC SEG (safe fallback if tool crashes).
      6) Append row(s) to CSV (one per selector per run).
    """
    logger.info("Starting subset selection and fine-tuning")

    # Resolve selector list
    selector_list = _parse_selector_arg(selector)
    if not selector_list:
        logger.error("No selectors provided after parsing.")
        raise typer.BadParameter("No selectors provided after parsing.")

    # Validate single selector tokens early if they look like shortcuts
    # (Fully-qualified module:Class will be validated at instantiation time)
    available = set(s.upper() for s in _available_selectors())
    for tok in selector_list:
        if ":" in tok:  # fully-qualified, skip pre-check
            continue
        if available and tok.upper() not in available:
            typer.echo(f"[WARN] Unknown selector shortcut '{tok}'. "
                       f"Known shortcuts: {sorted(available)}", err=True)
            logger.warning(f"Unknown selector shortcut '{tok}'. "
                       f"Known shortcuts: {sorted(available)}")

    # Build kwargs for selectors
    sel_kwargs = _merge_selector_kwargs(sel_arg, selector_kwargs)

    # --- parse/prepare ---
    with open(dataset_config, "r") as f:
        cfg = yaml.safe_load(f)
        logger.info(f"Loading dataset from {dataset_config}")
    ds_cfg, ev_cfg = cfg["dataset"], cfg["eval"]
    logger.debug(f"ds_cfg: {ds_cfg}")
    logger.debug(f"ev_cfg: {ev_cfg}")

    channels = _channels_from_cfg(ds_cfg)

    device = torch.device(f"cuda:{cuda}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logger.info(f"Using device: {device}")

    results_csv = os.path.join(_project_root(), "results", results_csv)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    logger.info(f"Results will be written to {results_csv}")

    # --- build dataset ---
    ds = build_dataset(
        "ctc",
        data_dir=Path(ds_cfg["data_dir"]).expanduser(),
        dataset_name=ds_cfg["dataset_name"],
        crop_boxes=ds_cfg.get("crop_boxes", {}),
    )
    dataset_root = os.path.join(Path(ds_cfg["data_dir"]).expanduser(), ds_cfg["dataset_name"])
    logger.info(f"Reading data from: {dataset_root}")

    # eval settings from YAML
    num_seq = int(ev_cfg["num_seq"])
    require_uncrop = bool(ev_cfg.get("require_uncrop", False))
    erosion_px = int(ev_cfg.get("erosion_pixels", 0))

    # CTC tool path (default under project/tools)
    if not ctc_app:
        ctc_app = os.path.join(_project_root(), "tools", "SEGMeasure")
    if not os.path.exists(ctc_app):
        logger.warning(f"SEGMeasure not found at {ctc_app}. Fallback to python implementation")
        typer.echo(f"[ERROR] SEGMeasure not found at {ctc_app}. Fallback to python implementation", err=True)

    # --- preload images once ---
    unlabeled_images = ds.load_images(ds.unlabeled_indices, video="01")
    test_images = ds.load_images(ds.test_indices, video="02")

    # --- features (cache per dataset+train video) ---
    cache_key = f"{ds_cfg['dataset_name']}_01_styles"
    X = extract_or_load_styles(
        unlabeled_images,
        device=device,
        model_type=model_type,
        channels=channels,
        cache_dir=features_dir,
        cache_key=cache_key,
        overwrite=overwrite_features,
    )
    if X.shape[0] != len(unlabeled_images):
        logger.error(f"Features shape {X.shape} does not match unlabeled images {len(unlabeled_images)}.")
        raise RuntimeError(f"Features shape {X.shape} does not match unlabeled images {len(unlabeled_images)}.")

    header = ["Dataset", "Model", "Train", "Test", "Run", "Seed", "Epochs", "Budget", "Indices", "Selector", "SEG"]

    # --- iterate selectors ---
    for selector_token in selector_list:
        logger.info(f"\n=== Selector: {selector_token} ===")

        # Pre-validate class import if it looks like a shortcut (not strictly required)
        try:
            sel_loader.load_selector_class(selector_token)
        except Exception as e:
            # Ignore for fully-qualified names that will resolve later;
            # otherwise, warn early to save user time.
            if ":" not in selector_token:
                logger.warning(f"Could not pre-load selector '{selector_token}': {e}")
                typer.echo(f"[WARN] Could not pre-load selector '{selector_token}': {e}", err=True)

        # --- runs ---
        for r in range(runs):
            logger.info(f"Run: {r+1}/{runs}")
            run_seed = seed + r
            set_all_seeds(run_seed)

            # reset split per selector+run
            ds.labeled_indices = []
            ds.unlabeled_indices = list(range(len(unlabeled_images)))

            # instantiate selector for current pool
            sel = sel_loader.instantiate_selector(
                selector=selector_token,
                features=X,
                **sel_kwargs,
            )
            picks = sel.select(budget)
            logger.info(f"{selector_token} selected samples (run {r+1}): {picks}")
            ds.annotate(picks)

            # gather train data
            tr_imgs = ds.load_images(ds.labeled_indices, video="01")
            tr_msks = ds.load_masks(ds.labeled_indices, video="01")

            # --- choose epochs ---
            if train_mode.lower() == "fixed" or len(tr_imgs) < 2:
                if train_mode.lower() != "fixed" and len(tr_imgs) < 2:
                    logger.warning("Fewer than 2 labeled images; CV disabled. Using --epochs-fixed.")
                n_epochs = int(epochs_fixed)
            else:
                grid = _build_epoch_grid(cv_start, cv_max, cv_step)
                if budget == 2:
                    kfolds = 0  # LOOCV
                else:
                    if cv_kfolds < 2:
                        logger.error("--cv-kfolds must be ≥ 2 when budget > 2")
                        raise typer.BadParameter("--cv-kfolds must be ≥ 2 when budget > 2")
                    kfolds = int(cv_kfolds)

                best_epochs, _scores = select_epochs_cv(
                    images=tr_imgs,
                    masks=tr_msks,
                    epoch_grid=grid,
                    k=kfolds,
                    device=device,
                    model_type=model_type,
                    channels=channels,
                    lr=lr,
                    wd=wd,
                    seed=run_seed,
                    use_python_metric=True,   # faster CV; final test uses official wrapper
                )
                n_epochs = int(best_epochs)
                logger.info(f"Best epochs: {best_epochs}")

            # --- train final model on selected subset ---
            model = train_fixed(
                device=device,
                model_type=model_type,
                channels=channels,
                images=tr_imgs,
                masks=tr_msks,
                epochs=n_epochs,
                lr=lr,
                wd=wd,
            )

            # --- evaluate on test set (official CTC path with safe fallback) ---
            diam = model.net.diam_labels.item()
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

            row = [
                ds_cfg["dataset_name"],
                model_type,
                "01",
                "02",
                r + 1,
                run_seed,
                n_epochs,
                int(budget),
                ds.labeled_indices,
                selector_token,
                seg,
            ]
            append_row(results_csv, header, row)
            logger.info(f"[{selector_token} | RUN {r+1}/{runs}] budget={budget} epochs={n_epochs} SEG={seg:.4f}")


if __name__ == "__main__":
    app()
