import numpy as np

from filter import filter_emg


def test_filter_attenuates_60hz():
    # Pure 60 Hz sine at fs=500 should be massively attenuated.
    # Use a 4-second signal so the Q=30 notch's ring-down has time to die
    # away before the central window we inspect.
    fs = 500
    t = np.arange(0, 4, 1 / fs)
    x = np.sin(2 * np.pi * 60 * t)
    y = filter_emg(x, fs=fs)
    # Ignore the first/last 500 samples (edge effects of a high-Q notch
    # under filtfilt are visible for ~1 second on each side).
    assert np.max(np.abs(y[500:-500])) < 0.1


def test_filter_attenuates_dc():
    # A constant signal should be removed (high-pass 20 Hz blocks DC)
    fs = 500
    x = np.full(1000, 5.0)
    y = filter_emg(x, fs=fs)
    assert np.max(np.abs(y[50:-50])) < 0.05


def test_filter_preserves_100hz():
    # Pure 100 Hz sine (within the passband 20-240) should be roughly preserved
    fs = 500
    t = np.arange(0, 1, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t)
    y = filter_emg(x, fs=fs)
    # Peak-to-peak roughly preserved (within 30% — filtfilt has some passband ripple)
    assert 1.4 < np.ptp(y[100:-100]) < 2.0
