# mdmr-paper/src/mdmr_paper/scripts/summarize_results.py
from __future__ import annotations

import os
import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

# ---------------------- helpers ---------------------- #

def _coerce_to_list(arg: Optional[str]) -> Optional[List[str]]:
    if arg is None or str(arg).strip() == "":
        return None
    return [x.strip() for x in str(arg).split(",") if x.strip()]

def _detect_kind_from_filename(fname: str) -> str:
    low = fname.lower()
    if "pretrained" in low or "pre-train" in low or "eval_pre" in low or "pretrain" in low:
        return "PRETRAINED"
    if "fully" in low or "full" in low or "supervised" in low:
        return "FULLY_SUPERVISED"
    if "active" in low or "al" in low or "fine" in low or "tune" in low:
        return "ACTIVE_LEARNING"
    return "UNKNOWN"

def _normalize_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Bring arbitrary results CSVs to a common schema:
      dataset, model, train, test, run, seed, epochs, selector, budget, seg, kind, source
    """
    cols = {c.lower(): c for c in df.columns}
    def get(*names, default=None):
        for n in names:
            if n.lower() in cols:
                return df[cols[n.lower()]]
        return default

    # SEG metric
    seg = get("SEG", "SEG_Score")
    if seg is None:
        # no SEG? skip this file by returning empty
        return pd.DataFrame()

    out = pd.DataFrame({
        "dataset": get("Dataset"),
        "model": get("Model"),
        "train": get("Train", default="01"),
        "test": get("Test", default="02"),
        "run": pd.to_numeric(get("Run", default=np.nan), errors="coerce"),
        "seed": pd.to_numeric(get("Seed", default=np.nan), errors="coerce"),
        "epochs": pd.to_numeric(get("Epochs", default=np.nan), errors="coerce"),
        "selector": get("Selector", default=np.nan),
        "budget": pd.to_numeric(get("Budget", default=np.nan), errors="coerce"),
        "seg": pd.to_numeric(seg, errors="coerce"),
    })

    # Fill selector/budget when missing using filename heuristics
    kind = get("Kind", default=None)
    if kind is None or (isinstance(kind, pd.Series) and kind.isna().all()):
        kind_val = _detect_kind_from_filename(source_name)
        out["kind"] = kind_val
    else:
        out["kind"] = kind

    # For fully supervised/pretrained, set canonical selector/budget
    mask_full = out["kind"].astype(str).str.upper().eq("FULLY_SUPERVISED")
    mask_pre = out["kind"].astype(str).str.upper().eq("PRETRAINED")
    out.loc[mask_full & out["selector"].isna(), "selector"] = "FULLY_SUPERVISED"
    out.loc[mask_pre & out["selector"].isna(), "selector"] = "PRETRAINED"
    out.loc[mask_pre & out["budget"].isna(),   "budget"]   = 0

    out["source"] = os.path.basename(source_name)

    # Clean types a bit
    for c in ["dataset", "model", "train", "test", "selector", "kind", "source"]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()

    return out


def _load_all_results(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            nf = _normalize_frame(df, f)
            if not nf.empty:
                frames.append(nf)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    if not frames:
        return pd.DataFrame(columns=["dataset","model","train","test","run","seed","epochs","selector","budget","seg","kind","source"])
    all_df = pd.concat(frames, ignore_index=True)
    # Drop rows with missing SEG
    all_df = all_df[all_df["seg"].notna()].copy()
    return all_df


def _apply_filters(df: pd.DataFrame,
                   datasets: Optional[List[str]],
                   selectors: Optional[List[str]],
                   budgets: Optional[List[int]]) -> pd.DataFrame:
    out = df.copy()
    if datasets:
        out = out[out["dataset"].isin(datasets)]
    if selectors:
        # case-insensitive match
        sel_norm = [s.lower() for s in selectors]
        out = out[out["selector"].str.lower().isin(sel_norm)]
    if budgets:
        out = out[out["budget"].isin(budgets)]
    return out


def _format_mean_std(mean: float, std: float, prec: int) -> str:
    if np.isnan(mean):
        return "nan"
    if np.isnan(std):
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} ± {std:.{prec}f}"


# ---------------------- CLI ---------------------- #

@app.command()
def summarize(
    results_dir: str = typer.Option("./results", help="Directory containing result CSV files."),
    output_csv: str = typer.Option("./results/summary.csv", help="Path to save the summary CSV."),
    group_by: str = typer.Option(
        "dataset,selector,budget",
        help="Comma-separated columns to group by. "
             "Common options: dataset,selector,budget,model,train,test,kind"
    ),
    datasets: Optional[str] = typer.Option(None, help="Optional comma-separated dataset filter."),
    selectors: Optional[str] = typer.Option(None, help="Optional comma-separated selector filter."),
    budgets: Optional[str] = typer.Option(None, help="Optional comma-separated budget filter (ints)."),
    precision: int = typer.Option(3, help="Decimals for mean/std display."),
):
    """
    Summarize results across runs, computing mean ± std of SEG for each group.

    - Automatically merges active learning, fully supervised, and pretrained CSVs (if present).
    - Normalizes columns and handles missing Selector/Budget.
    - Prints a pretty table and saves a machine-readable CSV.
    """
    # Load
    df = _load_all_results(results_dir)
    if df.empty:
        typer.echo(f"[ERROR] No results found in {results_dir}.")
        raise typer.Exit(code=1)

    # Filters
    ds_filter = _coerce_to_list(datasets)
    sel_filter = _coerce_to_list(selectors)
    bud_filter = _coerce_to_list(budgets)
    bud_filter_int = [int(x) for x in bud_filter] if bud_filter else None

    df = _apply_filters(df, ds_filter, sel_filter, bud_filter_int)

    if df.empty:
        typer.echo("[ERROR] No rows left after applying filters.")
        raise typer.Exit(code=1)

    # Grouping
    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            typer.echo(f"[ERROR] Unknown group-by column '{c}'. Available: {sorted(df.columns)}")
            raise typer.Exit(code=1)

    # Aggregate
    agg = (df
           .groupby(group_cols, dropna=False, sort=True)
           .agg(seg_mean=("seg","mean"),
                seg_std =("seg","std"),
                n=("seg","count"))
           .reset_index())

    # Pretty column for printing
    agg["SEG (mean±std)"] = [
        _format_mean_std(m, s, precision) for m, s in zip(agg["seg_mean"], agg["seg_std"])
    ]

    # Order columns nicely
    columns_out = group_cols + ["n", "SEG (mean±std)"]
    agg = agg[columns_out]
    agg = agg.rename(columns={'n': 'Runs'})

    # Print to screen
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 120):
        print("\n=== Summary (grouped by: " + ", ".join(group_cols) + ") ===")
        print(agg.to_string(index=False))

    # Save CSV (machine-friendly: keep numeric mean/std)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    agg.to_csv(output_csv, index=False)
    print(f"\nSaved summary to: {output_csv}")


if __name__ == "__main__":
    app()
