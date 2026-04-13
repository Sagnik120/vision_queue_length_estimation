"""
Evaluation script for the PA-SQGN queue estimation system.

Computes:
  - MAE, RMSE, NMAE% for queue count accuracy
  - Temporal stability: frame-to-frame delta std (correct measure of jitter)
  - Per-scene breakdown (queue vs crowd segments)
  - Ablation comparison table (Config A vs B vs C)
  - Scene classifier confusion matrix
  - Queue size over time plot

Prerequisites:
  - data/ground_truth.csv         (create manually — instructions below)
  - outputs/ablation_A.csv        (run scripts/run_hybrid.py)
  - outputs/ablation_B.csv        (run scripts/run_pose_only.py)
  - outputs/queue_log.csv         (run main_pose.py)

How to create data/ground_truth.csv:
  1. Open your video in VLC, enable View > Status Bar for frame numbers.
  2. Annotate at EVERY frame (or at minimum every 5 frames for stable metrics).
     Sparse sampling (every 15 frames) underestimates jitter and gives
     unstable RMSE — 24 data points is too few for a reliable estimate.
  3. For gt_label, use: "queue" / "crowd" / "uncertain"
     "uncertain" is a valid GT label — it should match the transitions
     between queue and crowd segments where the scene is genuinely ambiguous.

Expected columns: frame, gt_size, gt_label

Run with:
    pip install pandas numpy matplotlib seaborn scikit-learn
    python scripts/evaluate_metrics.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
os.makedirs("outputs", exist_ok=True)

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("NOTE: seaborn not installed — confusion matrix will be text-only.")

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("NOTE: scikit-learn not installed — skipping confusion matrix.")


# ── nearest-frame merge ────────────────────────────────────────────────────────
def merge_nearest(pred_df, gt_df, tolerance=8):
    """
    Merge predicted and GT dataframes on nearest frame number.

    FIX: exact-frame merge fails when GT annotation interval doesn't align
    with LOG_EVERY_N_FRAMES. This uses a nearest-neighbour merge with a
    configurable tolerance (default ±8 frames at 25fps ≈ ±0.3s).

    pred_df : DataFrame with 'frame' column (prediction CSV)
    gt_df   : DataFrame with 'frame' column (ground truth CSV)
    tolerance : max frame distance to accept as a match

    Returns merged DataFrame, or None if no matches within tolerance.
    """
    pred_df = pred_df.sort_values("frame").reset_index(drop=True)
    gt_df   = gt_df.sort_values("frame").reset_index(drop=True)

    rows = []
    for _, gt_row in gt_df.iterrows():
        dists = (pred_df["frame"] - gt_row["frame"]).abs()
        idx   = dists.idxmin()
        if dists[idx] <= tolerance:
            merged_row = pred_df.loc[idx].to_dict()
            merged_row.update({
                "gt_size":  gt_row["gt_size"],
                "gt_label": gt_row.get("gt_label", "queue"),
                "gt_frame": gt_row["frame"],
            })
            rows.append(merged_row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def load_and_merge(pred_csv, gt_csv, tolerance=8):
    if not os.path.exists(pred_csv):
        print(f"  MISSING: {pred_csv}")
        return None
    if not os.path.exists(gt_csv):
        print(f"  MISSING: {gt_csv} — create ground truth file first.")
        return None

    pred   = pd.read_csv(pred_csv)
    gt     = pd.read_csv(gt_csv)
    merged = merge_nearest(pred, gt, tolerance=tolerance)

    if merged is None or merged.empty:
        print(f"  No frames matched within ±{tolerance} frames for {pred_csv}.")
        print(f"  Check frame intervals in ground_truth.csv match {pred_csv}.")
        return None

    print(f"  Matched {len(merged)} / {len(gt)} GT frames (tolerance ±{tolerance})")
    return merged


# ── metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(df, config_name):
    """
    Compute accuracy and stability metrics.

    FIX: temporal stability is now frame-to-frame delta std, not total std.
    Total std measures how much the queue changes over time (a real property
    of the scene). Delta std measures system jitter — how much the predicted
    count fluctuates between consecutive matched frames independent of the
    true change. A perfect smoother on a static queue gives delta_std ≈ 0.
    """
    errors = df["queue_size"] - df["gt_size"]
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))

    # delta std: jitter in predicted signal, corrected for true GT change
    pred_delta = df["queue_size"].diff().dropna()
    gt_delta   = df["gt_size"].diff().dropna()
    residual_delta = pred_delta - gt_delta          # true signal change removed
    delta_std  = float(residual_delta.std())

    mean_gt = float(df["gt_size"].mean())
    nmae    = (mae / mean_gt * 100) if mean_gt > 0 else 0.0

    return {
        "config":    config_name,
        "MAE":       mae,
        "RMSE":      rmse,
        "NMAE%":     nmae,
        "DeltaStd":  delta_std,   # renamed from Std — clearer meaning
        "n_frames":  len(df),
    }


def compute_per_scene_metrics(df, config_name):
    """
    Break down MAE and RMSE by GT scene label.

    FIX: aggregate metrics hide per-scene failures. A system that tracks
    the crowd perfectly but fails on the queue still looks acceptable in
    the summary table. This function exposes per-label performance.
    """
    if "gt_label" not in df.columns:
        return {}

    results = {}
    for label in df["gt_label"].unique():
        subset = df[df["gt_label"] == label]
        errors = subset["queue_size"] - subset["gt_size"]
        results[label] = {
            "MAE":      float(np.mean(np.abs(errors))),
            "RMSE":     float(np.sqrt(np.mean(errors ** 2))),
            "n_frames": len(subset),
        }
    return results


# ── display helpers ────────────────────────────────────────────────────────────
def print_metrics_table(rows):
    print("\n" + "=" * 80)
    print(f"{'Configuration':<38} {'MAE':>6} {'RMSE':>7} {'NMAE%':>7} "
          f"{'ΔStd':>7} {'Frames':>7}")
    print(f"{'':38} {'':6} {'':7} {'':7} {'(jitter)':>7}")
    print("-" * 80)
    for r in rows:
        print(f"{r['config']:<38} {r['MAE']:>6.2f} {r['RMSE']:>7.2f} "
              f"{r['NMAE%']:>6.1f}% {r['DeltaStd']:>7.3f} {r['n_frames']:>7}")
    print("=" * 80)
    print("ΔStd = frame-to-frame delta std (residual after GT change removed).")
    print("Lower is better. Ideal smoother on static queue → ΔStd ≈ 0.")


def print_per_scene_table(per_scene_dict):
    print("\n── Per-scene breakdown ──────────────────────────────────────────────────")
    all_labels = sorted({lbl for d in per_scene_dict.values() for lbl in d})
    for config_name, scene_data in per_scene_dict.items():
        print(f"\n  {config_name}")
        print(f"  {'Label':<12} {'MAE':>6} {'RMSE':>7} {'Frames':>7}")
        for label in all_labels:
            if label in scene_data:
                r = scene_data[label]
                print(f"  {label:<12} {r['MAE']:>6.2f} {r['RMSE']:>7.2f} "
                      f"{r['n_frames']:>7}")


# ── plots ──────────────────────────────────────────────────────────────────────
def plot_queue_over_time(csvs_labels, gt_csv=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                   sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    for csv_path, label, colour in csvs_labels:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        ax1.plot(df["frame"], df["queue_size"],
                 label=label, color=colour, linewidth=1.5, alpha=0.85)

        # delta (jitter) in bottom panel — show residual after diff
        ax2.plot(df["frame"], df["queue_size"].diff().abs(),
                 color=colour, linewidth=0.8, alpha=0.6)

    if gt_csv and os.path.exists(gt_csv):
        gt = pd.read_csv(gt_csv)
        ax1.scatter(gt["frame"], gt["gt_size"],
                    label="Ground truth", color="black",
                    zorder=5, marker="x", s=60)

    ax1.set_ylabel("Queue size (persons)")
    ax1.set_title("Queue size over time — ablation comparison")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("|Δ queue size|")
    ax2.set_title("Frame-to-frame change (jitter proxy)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "outputs/queue_over_time.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot: {out}")
    plt.show()


def plot_confusion_matrix(pred_csv, gt_csv):
    """
    FIX: GT now supports 'uncertain' as a valid label (matching classifier
    output). Previously the GT instructions only listed queue/crowd, creating
    a systematic empty row in the confusion matrix that made precision look
    worse than it is without explanation.
    """
    if not HAS_SKLEARN:
        print("scikit-learn not installed — skipping confusion matrix.")
        return
    if not os.path.exists(pred_csv) or not os.path.exists(gt_csv):
        print("Skipping confusion matrix: missing files.")
        return

    pred   = pd.read_csv(pred_csv)[["frame", "scene_label"]]
    gt     = pd.read_csv(gt_csv)

    if "gt_label" not in gt.columns:
        print("gt_label column not found — skipping confusion matrix.")
        print("Add a 'gt_label' column with values: queue / crowd / uncertain")
        return

    merged = merge_nearest(pred, gt[["frame", "gt_label"]])
    if merged is None or merged.empty:
        return

    # Use all three labels — uncertain is valid in both GT and predictions
    labels = ["queue", "uncertain", "crowd"]
    cm     = confusion_matrix(merged["gt_label"], merged["scene_label"],
                               labels=labels)

    print("\nScene Classifier — Confusion Matrix")
    print(classification_report(merged["gt_label"], merged["scene_label"],
                                 labels=labels, zero_division=0))

    if HAS_SEABORN:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=labels, yticklabels=labels,
                    cmap="Blues")
        plt.title("Scene classifier — confusion matrix")
        plt.ylabel("Ground truth")
        plt.xlabel("Predicted")
        plt.tight_layout()
        out = "outputs/confusion_matrix.png"
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.show()


# ── GT template ────────────────────────────────────────────────────────────────
def create_sample_ground_truth(video_fps=25, video_frames=500):
    """
    Creates a ground_truth.csv template annotated at every 5 frames.

    FIX: previous template used every 15 frames — only 24 data points for a
    360-frame clip. This gives unstable RMSE and underestimates jitter.
    Every 5 frames gives ~100 data points, enough for stable metrics.
    Adjust video_frames to match your actual clip length.
    """
    os.makedirs("data", exist_ok=True)
    path = "data/ground_truth.csv"

    if os.path.exists(path):
        print(f"{path} already exists — not overwriting.")
        return

    # annotate every 5 frames for stable statistics
    frames = list(range(5, video_frames + 1, 5))
    rows   = [{"frame": f, "gt_size": 0, "gt_label": "queue"} for f in frames]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nCreated template: {path}  ({len(frames)} frames to annotate)")
    print("Open this file and:")
    print("  1. Replace gt_size=0 with your manually counted queue members.")
    print("  2. Set gt_label to 'queue', 'crowd', or 'uncertain' per frame.")
    print("     'uncertain' = transition frames where the scene is ambiguous.")
    print("     This matches the three outputs of CrowdClassifier.")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    gt_csv = "data/ground_truth.csv"

    if not os.path.exists(gt_csv):
        print("\nGround truth file not found.")
        create_sample_ground_truth()
        print("\nRe-run after filling in the ground truth values.")
        return

    gt = pd.read_csv(gt_csv)
    if (gt["gt_size"] == 0).all():
        print("\nGround truth file exists but all gt_size values are 0.")
        print("Fill in the actual queue counts and re-run.")
        return

    configs = [
        ("outputs/ablation_A.csv", "Config A — Hybrid baseline"),
        ("outputs/ablation_B.csv", "Config B — Pose graph only"),
        ("outputs/queue_log.csv",  "Config C — Full PA-SQGN"),
    ]

    summary_rows   = []
    per_scene_dict = {}

    print("\nComputing evaluation metrics...")
    for csv_path, name in configs:
        merged = load_and_merge(csv_path, gt_csv)
        if merged is not None:
            r = compute_metrics(merged, name)
            summary_rows.append(r)
            per_scene_dict[name] = compute_per_scene_metrics(merged, name)
            print(f"  Processed: {name}")
        else:
            print(f"  Skipped:   {name}")

    if summary_rows:
        print_metrics_table(summary_rows)
        print_per_scene_table(per_scene_dict)

    plot_queue_over_time(
        [
            ("outputs/ablation_A.csv", "Config A — Hybrid",    "#E24B4A"),
            ("outputs/ablation_B.csv", "Config B — Pose only", "#EF9F27"),
            ("outputs/queue_log.csv",  "Config C — Full",      "#1D9E75"),
        ],
        gt_csv=gt_csv,
    )

    plot_confusion_matrix("outputs/queue_log.csv", gt_csv)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
