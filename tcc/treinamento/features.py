"""Feature functions for sEMG window analysis.

Each function takes a 1-D numpy array (one window of samples) and returns
a single scalar feature. All functions are pure — no I/O, no globals.
"""

import numpy as np


def rms(x: np.ndarray) -> float:
    """Root Mean Square — sqrt(mean(x**2))."""
    return float(np.sqrt(np.mean(x.astype(float) ** 2)))


def mav(x: np.ndarray) -> float:
    """Mean Absolute Value — mean(|x|). Distinct from arithmetic mean."""
    return float(np.mean(np.abs(x.astype(float))))


def sd(x: np.ndarray) -> float:
    """Standard deviation, population (ddof=0)."""
    return float(np.std(x.astype(float), ddof=0))


def wl(x: np.ndarray) -> float:
    """Waveform Length — sum of absolute consecutive differences."""
    return float(np.sum(np.abs(np.diff(x.astype(float)))))


def var(x: np.ndarray) -> float:
    """Variance, population (ddof=0). Equal to sd(x)**2."""
    return float(np.var(x.astype(float), ddof=0))


def zc(x: np.ndarray, threshold: float = 0.0) -> int:
    """Zero Crossings — number of sign changes in x where
    |x[i+1] - x[i]| exceeds threshold (deadzone to suppress noise)."""
    x = x.astype(float)
    sign_changed = x[:-1] * x[1:] < 0
    large_enough = np.abs(x[1:] - x[:-1]) > threshold
    return int(np.sum(sign_changed & large_enough))


def ssc(x: np.ndarray, threshold: float = 0.0) -> int:
    """Slope Sign Changes — number of times the discrete derivative
    changes sign, with a deadzone on the change magnitude."""
    d = np.diff(x.astype(float))
    sign_changed = d[:-1] * d[1:] < 0
    large_enough = np.abs(d[1:] - d[:-1]) > threshold
    return int(np.sum(sign_changed & large_enough))


def wamp(x: np.ndarray, threshold: float) -> int:
    """Willison Amplitude — number of consecutive |diff| > threshold."""
    return int(np.sum(np.abs(np.diff(x.astype(float))) > threshold))
