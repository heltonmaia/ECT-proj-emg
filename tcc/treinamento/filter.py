"""Offline EMG filtering — matches the TCC's pipeline.

Notch 60 Hz (Q=30) + Butterworth band-pass 20–240 Hz, 4th order,
applied zero-phase via filtfilt / sosfiltfilt.

Design rationale
----------------
* Q=30 gives a narrow notch (~2 Hz bandwidth around 60 Hz), removing
  Brazilian mains interference while preserving nearby EMG content.
* 20 Hz high-pass cutoff removes motion artefacts and DC drift.
* 240 Hz low-pass cutoff stays below Nyquist (250 Hz @ fs=500) and
  retains the bulk of sEMG spectral energy (typically 20–250 Hz).
* The notch is a 2nd-order IIR — the transfer-function (b, a) form is
  numerically fine, so we use ``filtfilt`` for zero-phase response.
* The band-pass is a 4th-order Butterworth — at this order the tf form
  is sensitive to coefficient rounding, so we factor it into 2nd-order
  sections (SOS) and apply ``sosfiltfilt`` for both zero-phase response
  and improved numerical stability.
"""

import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, sosfiltfilt


def filter_emg(signal: np.ndarray, fs: int = 500,
               notch_freq: float = 60.0, notch_q: float = 30.0,
               lowcut: float = 20.0, highcut: float = 240.0,
               order: int = 4) -> np.ndarray:
    """Apply notch + band-pass to an EMG signal, zero-phase.

    Parameters
    ----------
    signal : np.ndarray
        1-D EMG samples.
    fs : int
        Sampling rate in Hz (default 500).
    notch_freq : float
        Mains frequency to reject (default 60 Hz).
    notch_q : float
        Quality factor of the notch (default 30).
    lowcut, highcut : float
        Band-pass cutoffs in Hz (default 20–240).
    order : int
        Butterworth order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as ``signal``.
    """
    x = signal.astype(float)
    # 1. Notch (IIR 2nd order) via filtfilt
    b, a = iirnotch(notch_freq, notch_q, fs)
    x = filtfilt(b, a, x)
    # 2. Band-pass Butterworth via sosfiltfilt (numerically stable)
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)
