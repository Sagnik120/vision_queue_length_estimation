"""
Day 1 test suite for pose_graph.py

Tests run without any video or GPU. All inputs are synthetic numpy arrays.

Run with:
    python test_day1.py
"""

import sys
import traceback
import numpy as np
import networkx as nx

# make sure project root is in path regardless of where you run from
sys.path.insert(0, ".")

from queue_detection.pose_graph import (
    compute_facing_angle,
    angle_difference,
    compute_edge_weight,
    build_pose_queue_graph,
    extract_queue_from_graph,
    compute_pose_queue,
)


# ---------- helpers ----------

def make_keypoints(left_shoulder, right_shoulder,
                   left_hip=None, right_hip=None, conf=0.9):
    """
    Build a (17, 3) keypoints array with only the joints we care about filled in.
    Everything else gets confidence 0.0.
    """
    kp = np.zeros((17, 3), dtype=float)

    kp[5] = [left_shoulder[0],  left_shoulder[1],  conf]   # left shoulder
    kp[6] = [right_shoulder[0], right_shoulder[1], conf]   # right shoulder

    if left_hip is not None:
        kp[11] = [left_hip[0],  left_hip[1],  conf]
    if right_hip is not None:
        kp[12] = [right_hip[0], right_hip[1], conf]

    return kp


def make_person(pid, x, y, angle=None, kp=None):
    return {
        'id':        pid,
        'pos':       (x, y),
        'angle':     angle,
        'keypoints': kp,
    }


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}")
        print(f"        AssertionError: {e}")
        return False
    except Exception:
        print(f"  ERROR {name}")
        traceback.print_exc()
        return False


# ---------- individual tests ----------

def test_facing_angle_returns_float():
    kp = make_keypoints([100, 200], [140, 200])
    angle = compute_facing_angle(kp)
    assert angle is not None, "Expected a float angle, got None"
    assert isinstance(angle, float), f"Expected float, got {type(angle)}"


def test_facing_angle_fallback_to_hips():
    # shoulders have zero confidence, hips have good confidence
    kp = np.zeros((17, 3), dtype=float)
    kp[11] = [100, 300, 0.9]  # left hip
    kp[12] = [140, 300, 0.9]  # right hip
    angle = compute_facing_angle(kp)
    assert angle is not None, "Hip fallback should return an angle"


def test_facing_angle_none_when_no_joints():
    kp = np.zeros((17, 3), dtype=float)  # all confidence = 0
    angle = compute_facing_angle(kp)
    assert angle is None, "Should return None when no reliable joints"


def test_angle_difference_same():
    diff = angle_difference(0.5, 0.5)
    assert abs(diff) < 1e-9, f"Same angles should give 0, got {diff}"


def test_angle_difference_opposite():
    diff = angle_difference(0.0, np.pi)
    assert abs(diff - np.pi) < 1e-6, f"Opposite angles should give pi, got {diff}"


def test_angle_difference_wrap_around():
    # 0.1 rad and 2*pi - 0.1 rad should be ~0.2 rad apart, not ~6 rad
    diff = angle_difference(0.1, 2 * np.pi - 0.1)
    assert diff < 0.5, f"Wrap-around angle diff too large: {diff}"


def test_edge_weight_range():
    w = compute_edge_weight((0, 0), (50, 0), 0.0, 0.0, scale=100)
    assert 0.0 <= w <= 1.0, f"Edge weight must be in [0,1], got {w}"


def test_edge_weight_same_direction_higher_than_opposite():
    # two people close, facing same direction vs opposite
    w_same = compute_edge_weight((0, 0), (30, 0), 0.0, 0.0, scale=100)
    w_opp  = compute_edge_weight((0, 0), (30, 0), 0.0, np.pi, scale=100)
    assert w_same > w_opp, (
        f"Same-direction weight ({w_same:.3f}) should exceed "
        f"opposite-direction weight ({w_opp:.3f})"
    )


def test_edge_weight_distance_decay():
    w_near = compute_edge_weight((0, 0), (10, 0),  0.0, 0.0, scale=100)
    w_far  = compute_edge_weight((0, 0), (200, 0), 0.0, 0.0, scale=100)
    assert w_near > w_far, "Nearby pair should have higher weight than far pair"


def test_graph_builds_with_valid_data():
    persons = [
        make_person(1, 100, 300, angle=0.0),
        make_person(2, 150, 300, angle=0.1),
        make_person(3, 200, 300, angle=0.05),
        make_person(4, 250, 300, angle=0.0),
    ]
    G = build_pose_queue_graph(persons)
    assert isinstance(G, nx.Graph), "Expected a networkx Graph"
    assert G.number_of_nodes() == 4, f"Expected 4 nodes, got {G.number_of_nodes()}"
    assert G.number_of_edges() > 0, "Queue-like arrangement should produce edges"


def test_graph_no_edges_for_opposite_facing():
    # two people facing exactly opposite directions and far apart
    persons = [
        make_person(1, 0,   300, angle=0.0),
        make_person(2, 500, 300, angle=np.pi),  # opposite direction, far away
    ]
    G = build_pose_queue_graph(persons)
    # could have 0 edges due to distance and angle penalty
    # just verify it doesn't crash and returns a Graph
    assert isinstance(G, nx.Graph)


def test_graph_too_few_persons():
    persons = [make_person(1, 100, 300, angle=0.0)]
    G = build_pose_queue_graph(persons)
    assert G.number_of_edges() == 0, "Single person: no edges expected"


def test_extract_queue_straight_line():
    # 5 people in a clear straight horizontal queue, all facing same direction
    persons = [make_person(i, i * 60, 300, angle=0.0) for i in range(5)]
    G = build_pose_queue_graph(persons)
    result = extract_queue_from_graph(G, persons, min_queue_size=3)
    assert result is not None, "Clear queue should be detected"
    queue_persons, length = result
    assert len(queue_persons) >= 3, f"Expected at least 3, got {len(queue_persons)}"
    assert length > 0, "Queue length should be positive"


def test_extract_queue_returns_none_for_tiny_group():
    persons = [make_person(i, i * 60, 300, angle=0.0) for i in range(2)]
    G = build_pose_queue_graph(persons)
    result = extract_queue_from_graph(G, persons, min_queue_size=3)
    assert result is None, "2 people cannot form a queue of min_size 3"


def test_compute_pose_queue_full_pipeline():
    # simulate a realistic queue: 6 people in a slightly noisy line
    rng = np.random.default_rng(42)
    persons = []
    for i in range(6):
        noise_x = float(rng.uniform(-10, 10))
        noise_y = float(rng.uniform(-15, 15))
        angle_noise = float(rng.uniform(-0.15, 0.15))
        persons.append(
            make_person(i, 100 + i * 70 + noise_x, 300 + noise_y,
                        angle=0.0 + angle_noise)
        )

    result = compute_pose_queue(persons, min_queue_size=3)
    assert result is not None, "Should detect a queue in this clean scenario"
    assert 'queue_persons'   in result
    assert 'queue_size'      in result
    assert 'queue_length_px' in result
    assert 'graph'           in result
    assert result['queue_size'] >= 3
    assert result['queue_length_px'] > 0


def test_compute_pose_queue_too_few():
    persons = [make_person(i, i * 60, 300, angle=0.0) for i in range(2)]
    result = compute_pose_queue(persons, min_queue_size=3)
    assert result is None, "Below minimum size should return None"


def test_compute_pose_queue_scattered_crowd():
    # random scatter - should either return None or a small queue
    rng = np.random.default_rng(99)
    persons = []
    for i in range(8):
        persons.append(
            make_person(
                i,
                float(rng.uniform(0, 800)),
                float(rng.uniform(0, 600)),
                angle=float(rng.uniform(0, 2 * np.pi))  # random facing
            )
        )
    # just check it doesn't crash - scattered crowd may or may not form a queue
    result = compute_pose_queue(persons, min_queue_size=3)
    # no assertion on None/not-None - just verify return type is correct
    if result is not None:
        assert isinstance(result['queue_size'], int)
        assert isinstance(result['queue_length_px'], float)


def test_queue_persons_are_ordered():
    # queue persons should be sorted from one end to the other (monotone positions)
    persons = [make_person(i, i * 80, 300, angle=0.0) for i in range(6)]
    result = compute_pose_queue(persons, min_queue_size=3)
    assert result is not None
    positions = [p['pos'][0] for p in result['queue_persons']]  # x coords
    # check monotone (either increasing or decreasing)
    diffs = [positions[k+1] - positions[k] for k in range(len(positions) - 1)]
    all_pos = all(d >= 0 for d in diffs)
    all_neg = all(d <= 0 for d in diffs)
    assert all_pos or all_neg, f"Queue positions not monotone: {positions}"


# ---------- run all ----------

def main():
    tests = [
        ("compute_facing_angle returns float",        test_facing_angle_returns_float),
        ("compute_facing_angle fallback to hips",     test_facing_angle_fallback_to_hips),
        ("compute_facing_angle None when no joints",  test_facing_angle_none_when_no_joints),
        ("angle_difference same angles",              test_angle_difference_same),
        ("angle_difference opposite angles",          test_angle_difference_opposite),
        ("angle_difference wrap around",              test_angle_difference_wrap_around),
        ("edge weight in [0,1]",                      test_edge_weight_range),
        ("edge weight: same dir > opposite dir",      test_edge_weight_same_direction_higher_than_opposite),
        ("edge weight: near > far",                   test_edge_weight_distance_decay),
        ("graph builds with valid data",              test_graph_builds_with_valid_data),
        ("graph: no crash for opposite facing",       test_graph_no_edges_for_opposite_facing),
        ("graph: handles single person",              test_graph_too_few_persons),
        ("extract queue: straight line",              test_extract_queue_straight_line),
        ("extract queue: returns None below min",     test_extract_queue_returns_none_for_tiny_group),
        ("full pipeline: clean queue",                test_compute_pose_queue_full_pipeline),
        ("full pipeline: too few people",             test_compute_pose_queue_too_few),
        ("full pipeline: scattered crowd",            test_compute_pose_queue_scattered_crowd),
        ("queue persons are ordered",                 test_queue_persons_are_ordered),
    ]

    print("\nRunning Day 1 tests for pose_graph.py\n" + "-" * 45)

    passed = 0
    failed = 0

    for name, fn in tests:
        ok = run_test(name, fn)
        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 45)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")

    if failed == 0:
        print("All tests passed. Day 1 core logic is working correctly.")
    else:
        print("Some tests failed. Fix the issues above before moving to Day 2.")

    return failed


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
