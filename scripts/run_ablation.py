"""
Ablation study runner.

Computes and prints comparison table for:
  Config A — Hybrid baseline        (outputs/ablation_A.csv)
  Config B — Pose graph only        (outputs/ablation_B.csv)
  Config C — Full PA-SQGN system    (outputs/queue_log.csv)

FIX: uses nearest-frame merge (tolerance=8 frames) instead of exact inner join.
FIX: reports delta_std (frame-to-frame jitter) not raw std — delta_std measures
     system instability, not scene dynamics.

Run:
    python scripts/run_ablation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def merge_nearest(pred_df, gt_df, tolerance=8):
    """
    Match each GT row to the nearest prediction frame within tolerance.
    Returns merged DataFrame or None.
    """
    if pred_df is None or gt_df is None:
        return None
    pred_df = pred_df.sort_values("frame").reset_index(drop=True)
    gt_df   = gt_df.sort_values("frame").reset_index(drop=True)
    rows = []
    for _, gt_row in gt_df.iterrows():
        dists = (pred_df["frame"] - gt_row["frame"]).abs()
        idx   = dists.idxmin()
        if dists[idx] <= tolerance:
            r = pred_df.loc[idx].to_dict()
            r["gt_size"]  = gt_row["gt_size"]
            r["gt_label"] = gt_row.get("gt_label", "queue")
            rows.append(r)
    return pd.DataFrame(rows) if rows else None


def compute_stats(df, gt_df=None):
    if df is None or df.empty:
        return None
    sizes  = df["queue_size"]
    deltas = sizes.diff().abs().dropna()
    stats  = {
        "mean":        round(float(sizes.mean()), 2),
        "max":         int(sizes.max()),
        "raw_std":     round(float(sizes.std()), 2),
        "delta_mean":  round(float(deltas.mean()), 3),
        "delta_std":   round(float(deltas.std()), 3),
        "mae":         None,
        "rmse":        None,
        "n_frames":    len(df),
    }
    if gt_df is not None:
        merged = merge_nearest(df, gt_df)
        if merged is not None and not merged.empty:
            err = merged["queue_size"] - merged["gt_size"]
            stats["mae"]  = round(float(np.mean(np.abs(err))), 2)
            stats["rmse"] = round(float(np.sqrt(np.mean(err**2))), 2)
    return stats


def print_table(rows):
    print("\n" + "=" * 90)
    print(f"{'Config':<36} {'Mean':>5} {'Max':>4} {'DeltaStd':>9} "
          f"{'DeltaMean':>10} {'MAE':>6} {'RMSE':>6} {'N':>5}")
    print("-" * 90)
    for r in rows:
        s = r["stats"]
        if s is None:
            print(f"{r['name']:<36}  (no data)")
            continue
        mae  = f"{s['mae']:.2f}"  if s["mae"]  is not None else " N/A "
        rmse = f"{s['rmse']:.2f}" if s["rmse"] is not None else " N/A "
        print(f"{r['name']:<36} {s['mean']:>5} {s['max']:>4} "
              f"{s['delta_std']:>9.3f} {s['delta_mean']:>10.3f} "
              f"{mae:>6} {rmse:>6} {s['n_frames']:>5}")
    print("=" * 90)


def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    gt = load("data/ground_truth.csv")
    has_gt = gt is not None and not (gt["gt_size"] == 0).all()
    if not has_gt:
        print("NOTE: data/ground_truth.csv not found or all zeros.")
        print("      MAE/RMSE will show N/A. Fill in ground_truth.csv to enable.\n")
        gt = None

    configs = [
        {"name": "Config A — Hybrid baseline",
         "path": "outputs/ablation_A.csv"},
        {"name": "Config B — Pose graph only",
         "path": "outputs/ablation_B.csv"},
        {"name": "Config C — Full PA-SQGN",
         "path": "outputs/queue_log.csv"},
    ]

    rows = []
    for cfg in configs:
        df = load(cfg["path"])
        if df is None:
            print(f"Missing: {cfg['path']} — run the corresponding script first.")
        cfg["stats"] = compute_stats(df, gt)
        rows.append(cfg)

    print_table(rows)

    # interpretation
    sa = rows[0]["stats"]
    sc = rows[2]["stats"]
    if sa and sc:
        reduction = round((sa["delta_std"] - sc["delta_std"]) / max(sa["delta_std"], 1e-6) * 100, 1)
        overcount = round(sa["mean"] - sc["mean"], 1)
        print(f"\nKey findings:")
        print(f"  Config C reduces frame-to-frame jitter (DeltaStd) by {reduction}% vs Config A.")
        print(f"  Config A overcounts by ~{overcount} persons (no orientation filter).")
        print(f"  DeltaStd is the correct stability metric — raw_std measures scene, not system.")


if __name__ == "__main__":
    main()
