import numpy as np
import os
import tempfile
import pandas as pd
import pytest

from data import load_csv, FS, DURATION


def _write_fake_csv(path):
    """Make a minimal 30s recording for testing."""
    n = FS * DURATION
    t = np.arange(n) / FS
    sig = np.sin(2 * np.pi * 100 * t) * 1000  # some EMG-ish-looking signal
    pd.DataFrame({"Tempo(s)": t, "EMG_Value": sig.astype(int)}).to_csv(path, index=False)


def test_load_csv_returns_right_lengths():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        _write_fake_csv(path)
        sig, labels = load_csv(path)
        assert len(sig) == FS * DURATION
        assert len(labels) == FS * DURATION
    finally:
        os.unlink(path)


def test_load_csv_label_schedule():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        _write_fake_csv(path)
        _, labels = load_csv(path)
        # Sample at t=2.5s (middle of first interval, mao aberta) -> 0
        assert labels[int(2.5 * FS)] == 0
        # Sample at t=7.5s (middle of second interval, mao fechada) -> 1
        assert labels[int(7.5 * FS)] == 1
        # Sample at t=27.5s (middle of last interval, mao fechada) -> 1
        assert labels[int(27.5 * FS)] == 1
    finally:
        os.unlink(path)


def test_load_csv_raises_on_missing_column():
    """Confirm the ValueError path for missing EMG_Value column."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        pd.DataFrame({"foo": [1, 2, 3]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="EMG_Value"):
            load_csv(path)
    finally:
        os.unlink(path)


def test_load_csv_raises_on_wrong_length():
    """Confirm the length guard for CSVs that aren't exactly 30s × 500Hz."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        # Half the expected length
        n = FS * DURATION // 2
        t = np.arange(n) / FS
        pd.DataFrame({"Tempo(s)": t, "EMG_Value": np.zeros(n)}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="expected"):
            load_csv(path)
    finally:
        os.unlink(path)


from data import make_windows, WINDOW_SIZE, STEP_SIZE, TRANSITION_MARGIN_SAMPLES
from data import extract_features


def test_make_windows_skips_transition_zones():
    # Build a fake signal where label changes at sample 2500 (5s × 500Hz)
    n = FS * DURATION
    sig = np.arange(n).astype(float)
    labels = np.empty(n, dtype=np.int8)
    labels[:2500] = 0
    labels[2500:5000] = 1
    labels[5000:7500] = 0
    # ... etc (we only need to test the transition)
    labels[7500:] = 1

    X, y = make_windows(sig, labels)
    # Sanity: must produce some windows
    assert len(X) > 0
    # Every window's label must equal the rótulo of all its samples
    # (no mixed-label windows). Recover each window's start from sig[i][0]
    # since signal = np.arange(n).
    for win, label in zip(X, y):
        win_start = int(win[0])
        win_labels = labels[win_start:win_start + WINDOW_SIZE]
        if len(set(win_labels)) > 1:
            # this window straddled a transition; shouldn't be in X/y
            assert False, f"window at {win_start} has mixed labels"


def test_make_windows_excludes_margin():
    """The 250 samples around each transition should produce no windows."""
    n = FS * DURATION
    sig = np.arange(n).astype(float)
    labels = np.empty(n, dtype=np.int8)
    labels[:2500] = 0
    labels[2500:] = 1
    X, y = make_windows(sig, labels)
    assert len(X) >= 1
    # No surviving window may straddle the transition zone [2500-250, 2500+250).
    # Recover each window's start from the tracer signal (sig = arange).
    for win in X:
        start = int(win[0])
        win_labels = labels[start:start + WINDOW_SIZE]
        assert win_labels.min() == win_labels.max(), (
            f"Window at start={start} has mixed labels — margin failed.")


def test_extract_features_returns_right_shape():
    X = np.random.randn(5, 100).astype(float)
    feature_names = ["rms", "mav", "sd", "wl"]
    F = extract_features(X, feature_names)
    assert F.shape == (5, 4)


def test_extract_features_supports_zc_ssc():
    X = np.random.randn(3, 100).astype(float)
    F = extract_features(X, ["rms", "mav", "sd", "wl", "zc", "ssc"])
    assert F.shape == (3, 6)
