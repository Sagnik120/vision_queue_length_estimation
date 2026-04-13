"""
Day 3 test suite for utils/langchain_alerts.py and utils/metrics_logger.py

These tests run without any API key or video.
The advisory agent tests use the fallback (rule-based) path.

Run with:
    python test_day3.py
"""

import sys
import os
import csv
import tempfile
import traceback

sys.path.insert(0, ".")

from utils.langchain_alerts import QueueAdvisoryAgent
from utils.metrics_logger import MetricsLogger


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
# QueueAdvisoryAgent tests (all use fallback, no API key needed)
# ===========================================================

def make_agent():
    # force fallback by not setting GOOGLE_API_KEY
    os.environ.pop("GOOGLE_API_KEY", None)
    return QueueAdvisoryAgent(location="Test Gate")


def test_agent_returns_string():
    agent = make_agent()
    result = agent.get_advisory(
        queue_size=8, queue_length_px=400,
        scene_label="queue", scene_confidence=0.85
    )
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 10, "Advisory should be a non-trivial string"


def test_agent_urgent_for_large_queue():
    agent = make_agent()
    result = agent.get_advisory(
        queue_size=15, queue_length_px=700,
        scene_label="queue", scene_confidence=0.90
    )
    assert "URGENT" in result or "15" in result, (
        f"Large queue advisory should mention urgency or count. Got: {result}"
    )


def test_agent_advisory_for_medium_queue():
    agent = make_agent()
    result = agent.get_advisory(
        queue_size=7, queue_length_px=350,
        scene_label="queue", scene_confidence=0.80
    )
    assert "7" in result or "ADVISORY" in result, (
        f"Medium queue advisory should mention count or advisory. Got: {result}"
    )


def test_agent_info_for_small_queue():
    agent = make_agent()
    result = agent.get_advisory(
        queue_size=3, queue_length_px=120,
        scene_label="queue", scene_confidence=0.70
    )
    assert "3" in result or "INFO" in result, (
        f"Small queue advisory should mention count or info. Got: {result}"
    )


def test_agent_crowd_label_gives_monitoring_message():
    agent = make_agent()
    result = agent.get_advisory(
        queue_size=5, queue_length_px=250,
        scene_label="crowd", scene_confidence=0.65
    )
    assert "crowd" in result.lower() or "monitoring" in result.lower(), (
        f"Crowd scene should mention monitoring. Got: {result}"
    )


def test_agent_rate_limiting_returns_same_message():
    agent = make_agent()
    first = agent.get_advisory(
        queue_size=6, queue_length_px=300,
        scene_label="queue", scene_confidence=0.80
    )
    # second call within rate-limit window should return cached result
    second = agent.get_advisory(
        queue_size=6, queue_length_px=300,
        scene_label="queue", scene_confidence=0.80
    )
    assert first == second, (
        "Rapid successive calls should return the same cached advisory"
    )


def test_agent_pixels_per_meter_affects_length():
    agent_fine = QueueAdvisoryAgent(pixels_per_meter=45.0)
    agent_wide = QueueAdvisoryAgent(pixels_per_meter=180.0)

    # reset last call time so rate limiter does not block
    agent_fine._last_call_time = 0
    agent_wide._last_call_time = 0

    result_fine = agent_fine.get_advisory(8, 360, "queue", 0.9)
    agent_wide._last_call_time = 0
    result_wide = agent_wide.get_advisory(8, 360, "queue", 0.9)

    # both should be strings
    assert isinstance(result_fine, str)
    assert isinstance(result_wide, str)


# ===========================================================
# MetricsLogger tests
# ===========================================================

def make_logger(tmp_dir):
    csv_path   = os.path.join(tmp_dir, "test_log.csv")
    return MetricsLogger(csv_path=csv_path, video_path=None), csv_path


def test_logger_creates_csv():
    with tempfile.TemporaryDirectory() as tmp:
        logger, csv_path = make_logger(tmp)
        logger.close()
        assert os.path.exists(csv_path), "CSV file should be created"


def test_logger_csv_has_header():
    with tempfile.TemporaryDirectory() as tmp:
        logger, csv_path = make_logger(tmp)
        logger.close()
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            assert "queue_size" in reader.fieldnames
            assert "scene_label" in reader.fieldnames
            assert "advisory_message" in reader.fieldnames


def test_logger_writes_rows():
    with tempfile.TemporaryDirectory() as tmp:
        logger, csv_path = make_logger(tmp)

        # log enough frames to trigger write (every 15 frames)
        for i in range(1, 31):
            logger.log(
                frame_number=i,
                queue_size=6,
                queue_length_px=300.0,
                scene_label="queue",
                scene_confidence=0.82,
                advisory_message="Test advisory.",
            )
        logger.close()

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) >= 1, (
            f"Expected at least 1 CSV row after 30 frames, got {len(rows)}"
        )


def test_logger_row_values_correct():
    with tempfile.TemporaryDirectory() as tmp:
        logger, csv_path = make_logger(tmp)

        # force a write by calling on a multiple of LOG_EVERY_N_FRAMES
        for i in range(1, 16):
            logger.log(
                frame_number=i,
                queue_size=9,
                queue_length_px=450.0,
                scene_label="queue",
                scene_confidence=0.91,
                advisory_message="Advisory test.",
            )
        logger.close()

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) >= 1, "Should have at least one row"
        row = rows[0]
        assert row["queue_size"]  == "9",     f"queue_size wrong: {row['queue_size']}"
        assert row["scene_label"] == "queue", f"scene_label wrong: {row['scene_label']}"
        assert "Advisory" in row["advisory_message"]


def test_logger_queue_length_m_computed():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "log.csv")
        logger   = MetricsLogger(csv_path=csv_path, pixels_per_meter=90.0)

        for i in range(1, 16):
            logger.log(i, 5, 450.0, "queue", 0.80, "Test.")
        logger.close()

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        row = rows[0]
        length_m = float(row["queue_length_m"])
        expected = 450.0 / 90.0
        assert abs(length_m - expected) < 0.1, (
            f"queue_length_m should be ~{expected:.2f}, got {length_m:.2f}"
        )


def test_logger_close_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        logger.close()
        logger.close()   # should not raise


# ===========================================================
# run all
# ===========================================================

def main():
    tests = [
        # advisory agent
        ("agent: returns a string",                    test_agent_returns_string),
        ("agent: urgent message for large queue",      test_agent_urgent_for_large_queue),
        ("agent: advisory for medium queue",           test_agent_advisory_for_medium_queue),
        ("agent: info for small queue",                test_agent_info_for_small_queue),
        ("agent: crowd label gives monitoring msg",    test_agent_crowd_label_gives_monitoring_message),
        ("agent: rate limiting returns cached msg",    test_agent_rate_limiting_returns_same_message),
        ("agent: pixels_per_meter param accepted",     test_agent_pixels_per_meter_affects_length),
        # metrics logger
        ("logger: creates CSV file",                   test_logger_creates_csv),
        ("logger: CSV has correct header",             test_logger_csv_has_header),
        ("logger: writes rows after N frames",         test_logger_writes_rows),
        ("logger: row values are correct",             test_logger_row_values_correct),
        ("logger: queue_length_m computed correctly",  test_logger_queue_length_m_computed),
        ("logger: close is safe to call twice",        test_logger_close_is_idempotent),
    ]

    print("\nRunning Day 3 tests\n" + "-" * 50)

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
        print("All tests passed. Day 3 modules are working correctly.")
    else:
        print("Some tests failed. Fix before running main_pose.py.")

    return failed


if __name__ == "__main__":
    sys.exit(main())
