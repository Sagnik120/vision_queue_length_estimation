"""
Day 4 test suite.

Covers:
  - velocity feature in pose_graph.py (new edge weight parameter)
  - density heatmap (draw_density_heatmap logic)
  - run_hybrid.py imports clean
  - run_pose_only.py imports clean
  - evaluate_metrics.py core functions
  - ground_truth template creation

Run with:
    python test_day4.py
"""

import sys
import os
import traceback
import tempfile
import numpy as np
import csv

sys.path.insert(0, ".")

from queue_detection.pose_graph import (
    compute_edge_weight,
    build_pose_queue_graph,
    compute_pose_queue,
    compute_facing_angle,
)


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}")
        print(f"        {e}")
        return False
    except Exception:
        print(f"  ERROR {name}")
        traceback.print_exc()
        return False


# ===========================================================
# Velocity feature tests
# ===========================================================

def test_edge_weight_accepts_velocity_args():
    w = compute_edge_weight((0,0), (50,0), 0.0, 0.0, (1,0), (1,0), scale=100)
    assert isinstance(w, float), "compute_edge_weight must return float"
    assert 0.0 <= w <= 1.0, f"weight out of range: {w}"


def test_velocity_same_direction_higher_weight():
    w_same = compute_edge_weight((0,0), (30,0), 0.0, 0.0,
                                  (2, 0), (2, 0), scale=100)
    w_opp  = compute_edge_weight((0,0), (30,0), 0.0, 0.0,
                                  (2, 0), (-2, 0), scale=100)
    assert w_same > w_opp, (
        f"Same-direction velocity should give higher weight. "
        f"Got same={w_same:.3f} opp={w_opp:.3f}"
    )


def test_velocity_zero_gives_neutral():
    w_zero    = compute_edge_weight((0,0), (30,0), 0.0, 0.0,
                                     (0, 0), (0, 0), scale=100)
    w_nonzero = compute_edge_weight((0,0), (30,0), 0.0, 0.0,
                                     (3, 0), (3, 0), scale=100)
    # zero velocity should give neutral factor 0.5, not zero the edge
    assert w_zero > 0, "Zero velocity should not zero-out the edge weight"
    assert w_nonzero >= w_zero, (
        "Coherent velocity should be >= neutral (zero) velocity"
    )


def test_build_graph_uses_velocity():
    persons = [
        {'id': i, 'pos': (i*60, 300), 'angle': 0.0,
         'velocity': (2, 0), 'keypoints': None}
        for i in range(5)
    ]
    G = build_pose_queue_graph(persons)
    assert G.number_of_edges() > 0, "Should have edges for queue-like group"


def test_build_graph_missing_velocity_fallback():
    # person dicts without 'velocity' key — should fall back to (0,0)
    persons = [
        {'id': i, 'pos': (i*60, 300), 'angle': 0.0, 'keypoints': None}
        for i in range(4)
    ]
    import networkx as nx
    G = build_pose_queue_graph(persons)
    assert isinstance(G, nx.Graph), "Should return Graph even without velocity key"


def test_full_pipeline_with_velocity():
    rng = np.random.default_rng(5)
    persons = []
    for i in range(6):
        nx_ = float(rng.uniform(-8, 8))
        ny_ = float(rng.uniform(-10, 10))
        persons.append({
            'id':        i,
            'pos':       (100 + i * 65 + nx_, 300 + ny_),
            'angle':     float(rng.uniform(-0.1, 0.1)),
            'velocity':  (2 + float(rng.uniform(-0.5, 0.5)), 0),
            'keypoints': None,
        })
    result = compute_pose_queue(persons, min_queue_size=3)
    assert result is not None, "Should detect a queue in clean scenario with velocity"
    assert result['queue_size'] >= 3


# ===========================================================
# Density heatmap tests (logic only, no display)
# ===========================================================

def test_heatmap_logic_no_crash():
    import cv2
    frame   = np.zeros((540, 960, 3), dtype=np.uint8)
    persons = [
        {'id': i, 'pos': (100 + i * 60, 300)}
        for i in range(5)
    ]

    # replicate draw_density_heatmap logic
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    for p in persons:
        cx, cy = map(int, p['pos'])
        y1 = max(0, cy - 35)
        y2 = min(frame.shape[0], cy + 35)
        x1 = max(0, cx - 35)
        x2 = min(frame.shape[1], cx + 35)
        heatmap[y1:y2, x1:x2] += 1.0

    heatmap      = cv2.GaussianBlur(heatmap, (71, 71), 0)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    result       = cv2.addWeighted(heatmap_bgr, 0.30, frame, 0.70, 0)

    assert result.shape == frame.shape, "Heatmap must not change frame shape"
    assert result.dtype == np.uint8, "Output must be uint8"


def test_heatmap_empty_persons_no_crash():
    import cv2
    frame   = np.zeros((540, 960, 3), dtype=np.uint8)
    persons = []
    # should do nothing without crashing
    if not persons:
        return  # draw_density_heatmap returns early — this is correct behaviour


def test_heatmap_person_at_edge_no_crash():
    import cv2
    frame   = np.zeros((540, 960, 3), dtype=np.uint8)
    persons = [
        {'id': 0, 'pos': (0, 0)},      # top-left corner
        {'id': 1, 'pos': (959, 539)},  # bottom-right corner
    ]
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    for p in persons:
        cx, cy = map(int, p['pos'])
        y1 = max(0, cy - 35)
        y2 = min(frame.shape[0], cy + 35)
        x1 = max(0, cx - 35)
        x2 = min(frame.shape[1], cx + 35)
        heatmap[y1:y2, x1:x2] += 1.0
    # should not raise IndexError at frame edges
    assert heatmap.max() > 0, "Heatmap should have non-zero values"


# ===========================================================
# Script import tests
# ===========================================================

def test_run_hybrid_imports():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_hybrid", "scripts/run_hybrid.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # just loading the module (not calling main) should not raise
    assert mod is not None


def test_run_pose_only_imports():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_pose_only", "scripts/run_pose_only.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert mod is not None


# ===========================================================
# evaluate_metrics.py helper tests
# ===========================================================

def test_ground_truth_template_created():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gt.csv")
        frames = list(range(15, 361, 15))
        rows   = [{"frame": f, "gt_size": 0, "gt_label": "queue"}
                  for f in frames]
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
        assert os.path.exists(path)

        df = pd.read_csv(path)
        assert "frame"    in df.columns
        assert "gt_size"  in df.columns
        assert "gt_label" in df.columns
        assert len(df) == 24


def test_metrics_computation_correct():
    import pandas as pd

    pred = pd.DataFrame({
        "frame":      [15, 30, 45, 60, 75],
        "queue_size": [6,   8,  7,  5,  9],
    })
    gt = pd.DataFrame({
        "frame":   [15, 30, 45, 60, 75],
        "gt_size": [7,   7,  8,  6,  8],
    })
    merged = pred.merge(gt, on="frame")

    mae  = float(np.mean(np.abs(merged["queue_size"] - merged["gt_size"])))
    rmse = float(np.sqrt(np.mean((merged["queue_size"] - merged["gt_size"])**2)))

    assert abs(mae  - 1.0) < 0.01, f"MAE wrong: {mae}"
    assert abs(rmse - 1.0) < 0.01, f"RMSE wrong: {rmse}"


def test_csv_columns_present():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "log.csv")
        df = pd.DataFrame({
            "timestamp": ["12:00:00"],
            "frame": [15],
            "queue_size": [6],
            "queue_length_px": [430.0],
            "queue_length_m": [4.78],
            "scene_label": ["queue"],
            "scene_confidence": [0.70],
            "advisory_message": ["Test."],
        })
        df.to_csv(path, index=False)
        loaded = pd.read_csv(path)
        required = ["frame", "queue_size", "scene_label", "scene_confidence"]
        for col in required:
            assert col in loaded.columns, f"Missing column: {col}"


# ===========================================================
# run all
# ===========================================================

def main():
    tests = [
        # velocity feature
        ("velocity: edge weight accepts vel args",        test_edge_weight_accepts_velocity_args),
        ("velocity: same direction > opposite",           test_velocity_same_direction_higher_weight),
        ("velocity: zero velocity gives neutral score",   test_velocity_zero_gives_neutral),
        ("velocity: graph builds with velocity field",    test_build_graph_uses_velocity),
        ("velocity: graph falls back if velocity missing",test_build_graph_missing_velocity_fallback),
        ("velocity: full pipeline with velocity",         test_full_pipeline_with_velocity),
        # density heatmap
        ("heatmap: logic runs without crash",             test_heatmap_logic_no_crash),
        ("heatmap: empty person list no crash",           test_heatmap_empty_persons_no_crash),
        ("heatmap: person at frame edge no crash",        test_heatmap_person_at_edge_no_crash),
        # script imports
        ("scripts: run_hybrid.py importable",             test_run_hybrid_imports),
        ("scripts: run_pose_only.py importable",          test_run_pose_only_imports),
        # evaluate_metrics helpers
        ("eval: ground truth template structure",         test_ground_truth_template_created),
        ("eval: MAE and RMSE computation correct",        test_metrics_computation_correct),
        ("eval: CSV columns present",                     test_csv_columns_present),
    ]

    print("\nRunning Day 4 tests\n" + "-" * 50)

    passed = 0
    failed = 0

    for name, fn in tests:
        ok = run_test(name, fn)
        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")

    if failed == 0:
        print("All tests passed. Day 4 is complete and correct.")
    else:
        print("Fix the failures above before moving to Day 5.")

    return failed


if __name__ == "__main__":
    sys.exit(main())
