"""
Day 2 test suite for temporal_smoother.py and crowd_classifier.py

Run with:
    python test_day2.py
"""

import sys
import traceback
import numpy as np

sys.path.insert(0, ".")

from queue_detection.temporal_smoother import TemporalSmoother
from queue_detection.crowd_classifier import (
    CrowdClassifier,
    extract_graph_features,
    classify_scene,
    LABEL_QUEUE,
    LABEL_CROWD,
    LABEL_UNCERTAIN,
)
from queue_detection.pose_graph import build_pose_queue_graph


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


# ===========================================================
# TemporalSmoother tests
# ===========================================================

def test_smoother_score_rises_when_in_queue():
    sm = TemporalSmoother()
    sm.update({1})
    sm.update({1})
    sm.update({1})
    score = sm.get_score(1)
    assert score > 0.5, f"Score should be above 0.5 after 3 frames in queue, got {score:.3f}"


def test_smoother_score_decays_when_absent():
    sm = TemporalSmoother()
    # build up score first
    for _ in range(6):
        sm.update({1})
    high_score = sm.get_score(1)

    # now person leaves queue for 6 frames
    for _ in range(6):
        sm.update(set())

    low_score = sm.get_score(1)
    assert low_score < high_score, (
        f"Score should decay after leaving queue. Was {high_score:.3f}, now {low_score:.3f}"
    )


def test_smoother_score_never_exceeds_one():
    sm = TemporalSmoother()
    for _ in range(50):
        sm.update({1})
    score = sm.get_score(1)
    assert score <= 1.0, f"Score must not exceed 1.0, got {score:.4f}"


def test_smoother_score_never_below_zero():
    sm = TemporalSmoother()
    sm.update({1})
    for _ in range(30):
        sm.update(set())
    score = sm.get_score(1)
    assert score >= 0.0, f"Score must not go below 0.0, got {score:.4f}"


def test_smoother_person_forgotten_after_long_absence():
    sm = TemporalSmoother()
    sm.update({1})
    for _ in range(40):
        sm.update(set())
    # score should be below forget threshold, person removed
    score = sm.get_score(1)
    assert score == 0.0, (
        f"Person should be forgotten (score=0.0) after long absence, got {score:.4f}"
    )


def test_smoother_stable_ids_correct():
    sm = TemporalSmoother()
    # person 1 builds up high confidence
    for _ in range(8):
        sm.update({1, 2})
    # person 2 leaves
    for _ in range(6):
        sm.update({1})

    stable = sm.get_stable_ids()
    assert 1 in stable, "Person 1 should be stable (never left)"


def test_smoother_multiple_persons_independent():
    sm = TemporalSmoother()
    sm.update({1, 2, 3})
    sm.update({1, 2, 3})
    sm.update({1})         # 2 and 3 leave

    score_1 = sm.get_score(1)
    score_2 = sm.get_score(2)
    assert score_1 > score_2, (
        f"Person 1 (stayed) should have higher score than person 2 (left). "
        f"Got {score_1:.3f} vs {score_2:.3f}"
    )


def test_smoother_reset_clears_all():
    sm = TemporalSmoother()
    for _ in range(5):
        sm.update({1, 2})
    sm.reset()
    assert sm.get_score(1) == 0.0
    assert sm.get_score(2) == 0.0
    assert sm.get_stable_ids() == []


def test_smoother_returns_scores_dict():
    sm = TemporalSmoother()
    result = sm.update({1, 2})
    assert isinstance(result, dict), "update() should return a dict"
    assert 1 in result and 2 in result


# ===========================================================
# extract_graph_features tests
# ===========================================================

def make_person(pid, x, y, angle=None):
    return {'id': pid, 'pos': (x, y), 'angle': angle, 'keypoints': None}


def make_queue_persons(n=6, spacing=70, angle=0.1):
    return [make_person(i, 100 + i * spacing, 300, angle=angle) for i in range(n)]


def make_scattered_persons(n=6):
    rng = np.random.default_rng(7)
    return [
        make_person(i,
                    float(rng.uniform(0, 800)),
                    float(rng.uniform(0, 600)),
                    angle=float(rng.uniform(0, 2 * np.pi)))
        for i in range(n)
    ]


def test_features_returns_dict_for_valid_input():
    persons = make_queue_persons()
    G = build_pose_queue_graph(persons)
    feats = extract_graph_features(persons, G)
    assert feats is not None, "Should return features for valid input"
    required_keys = {'angle_variance', 'aspect_ratio', 'avg_edge_weight',
                     'density', 'linearity_score'}
    assert required_keys.issubset(feats.keys()), f"Missing keys: {required_keys - feats.keys()}"


def test_features_returns_none_for_too_few():
    persons = [make_person(0, 100, 300), make_person(1, 200, 300)]
    G = build_pose_queue_graph(persons)
    feats = extract_graph_features(persons, G)
    assert feats is None, "Should return None for fewer than 3 persons"


def test_features_queue_has_low_angle_variance():
    persons = make_queue_persons(angle=0.05)
    G = build_pose_queue_graph(persons)
    feats = extract_graph_features(persons, G)
    assert feats['angle_variance'] < 0.3, (
        f"Queue should have low angle variance, got {feats['angle_variance']:.3f}"
    )


def test_features_queue_has_high_linearity():
    persons = make_queue_persons(n=8, spacing=60)
    G = build_pose_queue_graph(persons)
    feats = extract_graph_features(persons, G)
    assert feats['linearity_score'] > 0.7, (
        f"Queue should have high linearity, got {feats['linearity_score']:.3f}"
    )


def test_features_queue_has_high_aspect_ratio():
    # horizontal line: much wider than tall
    persons = make_queue_persons(n=6, spacing=80)
    G = build_pose_queue_graph(persons)
    feats = extract_graph_features(persons, G)
    assert feats['aspect_ratio'] > 2.0, (
        f"Horizontal queue should have high aspect ratio, got {feats['aspect_ratio']:.3f}"
    )


# ===========================================================
# classify_scene tests
# ===========================================================

def test_classify_queue_features_gives_queue_label():
    features = {
        'angle_variance':  0.10,   # low = same direction
        'aspect_ratio':    4.5,    # elongated
        'avg_edge_weight': 0.40,   # strong connections
        'density':         2.5,
        'linearity_score': 0.90,   # very linear
    }
    label, conf, _ = classify_scene(features, n_persons=8)
    assert label == LABEL_QUEUE, f"Expected 'queue', got '{label}' (conf={conf:.2f})"


def test_classify_crowd_features_gives_crowd_label():
    features = {
        'angle_variance':  0.80,   # everyone facing different ways
        'aspect_ratio':    1.1,    # roughly square blob
        'avg_edge_weight': 0.05,   # weak connections
        'density':         8.0,
        'linearity_score': 0.52,   # not linear
    }
    label, conf, _ = classify_scene(features, n_persons=10)
    assert label == LABEL_CROWD, f"Expected 'crowd', got '{label}' (conf={conf:.2f})"


def test_classify_returns_uncertain_for_none_features():
    label, conf, _ = classify_scene(None, n_persons=5)
    assert label == LABEL_UNCERTAIN


def test_classify_confidence_in_range():
    features = {
        'angle_variance':  0.2,
        'aspect_ratio':    3.0,
        'avg_edge_weight': 0.3,
        'density':         2.0,
        'linearity_score': 0.85,
    }
    _, conf, _ = classify_scene(features, n_persons=6)
    assert 0.0 <= conf <= 1.0, f"Confidence must be in [0, 1], got {conf}"


def test_classify_too_few_persons_is_uncertain():
    features = {
        'angle_variance':  0.1,
        'aspect_ratio':    5.0,
        'avg_edge_weight': 0.5,
        'density':         1.0,
        'linearity_score': 0.95,
    }
    label, _, _ = classify_scene(features, n_persons=2)
    assert label == LABEL_UNCERTAIN, "Fewer than 3 persons should be uncertain"


# ===========================================================
# CrowdClassifier integration tests
# ===========================================================

def test_classifier_returns_queue_for_line():
    clf = CrowdClassifier(history_len=5)
    persons = make_queue_persons(n=7, spacing=65, angle=0.05)
    G = build_pose_queue_graph(persons)

    final_label = None
    for _ in range(8):
        final_label, conf, feats = clf.update(persons, G)

    assert final_label == LABEL_QUEUE, (
        f"Repeated queue frames should stabilise as 'queue', got '{final_label}'"
    )


def test_classifier_smoothing_ignores_single_bad_frame():
    clf = CrowdClassifier(history_len=6)
    queue_persons = make_queue_persons(n=7, spacing=65, angle=0.05)
    G_queue = build_pose_queue_graph(queue_persons)

    # feed 5 clear queue frames
    for _ in range(5):
        clf.update(queue_persons, G_queue)

    # one noisy frame with scattered persons
    scattered = make_scattered_persons(n=7)
    G_scattered = build_pose_queue_graph(scattered)
    clf.update(scattered, G_scattered)

    # majority should still be queue
    label, _, _ = clf.update(queue_persons, G_queue)
    assert label == LABEL_QUEUE, (
        f"One bad frame should not flip label from queue to '{label}'"
    )


def test_classifier_reset_clears_history():
    clf = CrowdClassifier()
    persons = make_queue_persons()
    G = build_pose_queue_graph(persons)
    for _ in range(5):
        clf.update(persons, G)
    clf.reset()
    assert clf._label_history == []


# ===========================================================
# run all
# ===========================================================

def main():
    tests = [
        # temporal smoother
        ("smoother: score rises when in queue",              test_smoother_score_rises_when_in_queue),
        ("smoother: score decays when absent",               test_smoother_score_decays_when_absent),
        ("smoother: score never exceeds 1.0",                test_smoother_score_never_exceeds_one),
        ("smoother: score never below 0.0",                  test_smoother_score_never_below_zero),
        ("smoother: person forgotten after long absence",    test_smoother_person_forgotten_after_long_absence),
        ("smoother: stable_ids returns correct persons",     test_smoother_stable_ids_correct),
        ("smoother: multiple persons tracked independently", test_smoother_multiple_persons_independent),
        ("smoother: reset clears all state",                 test_smoother_reset_clears_all),
        ("smoother: update returns dict",                    test_smoother_returns_scores_dict),
        # feature extraction
        ("features: returns dict for valid input",           test_features_returns_dict_for_valid_input),
        ("features: returns None for too few persons",       test_features_returns_none_for_too_few),
        ("features: queue has low angle variance",           test_features_queue_has_low_angle_variance),
        ("features: queue has high linearity",               test_features_queue_has_high_linearity),
        ("features: queue has high aspect ratio",            test_features_queue_has_high_aspect_ratio),
        # classifier
        ("classify: queue features -> queue label",          test_classify_queue_features_gives_queue_label),
        ("classify: crowd features -> crowd label",          test_classify_crowd_features_gives_crowd_label),
        ("classify: None features -> uncertain",             test_classify_returns_uncertain_for_none_features),
        ("classify: confidence in [0,1]",                    test_classify_confidence_in_range),
        ("classify: too few persons -> uncertain",           test_classify_too_few_persons_is_uncertain),
        # integration
        ("classifier: stable queue -> queue label",          test_classifier_returns_queue_for_line),
        ("classifier: smoothing ignores one bad frame",      test_classifier_smoothing_ignores_single_bad_frame),
        ("classifier: reset clears history",                 test_classifier_reset_clears_history),
    ]

    print("\nRunning Day 2 tests\n" + "-" * 50)

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
        print("All tests passed. Day 2 modules are working correctly.")
    else:
        print("Some tests failed. Fix before moving to Day 3.")

    return failed


if __name__ == "__main__":
    sys.exit(main())
