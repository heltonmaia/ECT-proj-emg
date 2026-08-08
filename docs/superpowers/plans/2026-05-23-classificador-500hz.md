# Classificador EMG 500 Hz — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir o pipeline de treino reproduzível para o classificador binário EMG (mão aberta/fechada) a 500 Hz, regenerando o `prediction.py` deployado para que ele bata com o modelo treinado, e rodar um sweep modesto pra tentar superar o baseline de 83% do TCC.

**Architecture:** Pasta nova `tcc/treinamento/` com módulos separados por responsabilidade (`features.py`, `filter.py`, `data.py`, `train.py`, template MicroPython). Treino offline em scikit-learn, validação via Leave-One-Group-Out, regeneração automática do `prediction.py` a partir de um template + tree exportada como if/else.

**Tech Stack:** Python (scipy, numpy, pandas, scikit-learn, matplotlib, joblib, pytest). MicroPython no destino do `prediction.py` (sem dependências sklearn na placa).

**Spec:** `docs/superpowers/specs/2026-05-23-classificador-500hz-design.md`

---

## Task 1: Bootstrap `tcc/treinamento/` + primeiro teste passando

**Files:**
- Create: `tcc/treinamento/features.py`
- Create: `tcc/treinamento/test_features.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Adicionar dependências ao requirements.txt**

Edit `requirements.txt`, adicionando alfabeticamente:

```
joblib==1.4.2
pytest==8.3.5
scikit-learn==1.5.2
```

(Use as últimas versões estáveis no momento da execução; valores acima são exemplos.)

- [ ] **Step 2: Instalar deps**

Run:
```bash
VIRTUAL_ENV=$(pwd)/.venv uv pip install scikit-learn joblib pytest
```
Expected: instala sem erro. Confere com `python -c "import sklearn, joblib, pytest"`.

- [ ] **Step 3: Criar `features.py` inicial com docstring e `rms()`**

```python
"""Feature functions for sEMG window analysis.

Each function takes a 1-D numpy array (one window of samples) and returns
a single scalar feature. All functions are pure — no I/O, no globals.
"""

import numpy as np


def rms(x: np.ndarray) -> float:
    """Root Mean Square — sqrt(mean(x**2))."""
    return float(np.sqrt(np.mean(x.astype(float) ** 2)))
```

- [ ] **Step 4: Escrever primeiro teste em `test_features.py`**

```python
import numpy as np
import pytest

from features import rms


def test_rms_constant_signal():
    # RMS of constant 3 is 3
    assert rms(np.array([3, 3, 3, 3])) == pytest.approx(3.0)


def test_rms_zero_signal():
    assert rms(np.zeros(10)) == 0.0


def test_rms_handles_negative():
    # RMS of [-3, 3] = sqrt((9+9)/2) = 3
    assert rms(np.array([-3, 3])) == pytest.approx(3.0)
```

- [ ] **Step 5: Rodar pytest, confirmar 3 passes**

Run:
```bash
cd tcc/treinamento && python -m pytest test_features.py -v
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add tcc/treinamento/features.py tcc/treinamento/test_features.py requirements.txt
git commit -m "feat(treinamento): scaffold features.py with rms + tests"
```

---

## Task 2: Implementar features de amplitude (MAV, SD, WL, VAR)

**Files:**
- Modify: `tcc/treinamento/features.py`
- Modify: `tcc/treinamento/test_features.py`

- [ ] **Step 1: Escrever teste pra `mav` (incluindo o caso crítico MAV ≠ mean)**

Append to `test_features.py`:

```python
def test_mav_simple():
    assert mav(np.array([2, 2, 2])) == pytest.approx(2.0)


def test_mav_differs_from_mean_on_signed_values():
    # MAV uses abs(); mean does not. This is the bug in prediction.py.
    x = np.array([-2, 2])
    assert mav(x) == pytest.approx(2.0)
    assert np.mean(x) == 0.0          # confirma que mean simples é diferente
```

Update import line:
```python
from features import rms, mav
```

- [ ] **Step 2: Rodar teste, confirmar FAIL (`mav` não existe)**

Run:
```bash
python -m pytest test_features.py -v
```
Expected: ImportError / failure for the new tests.

- [ ] **Step 3: Implementar `mav` em `features.py`**

```python
def mav(x: np.ndarray) -> float:
    """Mean Absolute Value — mean(|x|). Distinct from arithmetic mean."""
    return float(np.mean(np.abs(x.astype(float))))
```

- [ ] **Step 4: Rodar testes, confirmar todos passam**

Run: `python -m pytest test_features.py -v`
Expected: `5 passed`

- [ ] **Step 5: Escrever testes para `sd`, `wl`, `var`**

Append:
```python
def test_sd_constant_signal_is_zero():
    assert sd(np.array([5, 5, 5, 5])) == 0.0


def test_sd_simple():
    # std of [1,2,3,4,5] with ddof=0 = sqrt(2)
    assert sd(np.array([1, 2, 3, 4, 5])) == pytest.approx(np.sqrt(2))


def test_wl_total_variation():
    # WL = sum of absolute consecutive differences
    # [1, 3, 2, 5] -> |3-1| + |2-3| + |5-2| = 2 + 1 + 3 = 6
    assert wl(np.array([1, 3, 2, 5])) == pytest.approx(6.0)


def test_wl_constant_signal_is_zero():
    assert wl(np.array([7, 7, 7, 7])) == 0.0


def test_var_is_sd_squared():
    x = np.array([1, 2, 3, 4, 5])
    assert var(x) == pytest.approx(sd(x) ** 2)
```

Update import:
```python
from features import rms, mav, sd, wl, var
```

- [ ] **Step 6: Implementar `sd`, `wl`, `var`**

```python
def sd(x: np.ndarray) -> float:
    """Standard deviation, population (ddof=0)."""
    return float(np.std(x.astype(float), ddof=0))


def wl(x: np.ndarray) -> float:
    """Waveform Length — sum of absolute consecutive differences."""
    return float(np.sum(np.abs(np.diff(x.astype(float)))))


def var(x: np.ndarray) -> float:
    """Variance, population (ddof=0). Equal to sd(x)**2."""
    return float(np.var(x.astype(float), ddof=0))
```

- [ ] **Step 7: Rodar testes, confirmar tudo passa**

Run: `python -m pytest test_features.py -v`
Expected: `10 passed`

- [ ] **Step 8: Commit**

```bash
git add tcc/treinamento/features.py tcc/treinamento/test_features.py
git commit -m "feat(treinamento): add MAV/SD/WL/VAR features with tests"
```

---

## Task 3: Implementar features espectrais (ZC, SSC, WAMP)

**Files:**
- Modify: `tcc/treinamento/features.py`
- Modify: `tcc/treinamento/test_features.py`

- [ ] **Step 1: Escrever testes para `zc` e `ssc`**

Append:
```python
def test_zc_no_crossings_when_all_positive():
    assert zc(np.array([1, 2, 3, 4, 5])) == 0


def test_zc_counts_sign_changes():
    # [1, -1, 1, -1, 1] has 4 sign changes
    assert zc(np.array([1, -1, 1, -1, 1])) == 4


def test_zc_threshold_suppresses_small_changes():
    # diff = 0.1; with threshold 0.5, no crossings count
    assert zc(np.array([0.1, -0.1, 0.1]), threshold=0.5) == 0


def test_ssc_counts_slope_reversals():
    # [1, 2, 1, 2] -> diff = [1, -1, 1] -> 2 sign changes in diff
    assert ssc(np.array([1, 2, 1, 2])) == 2


def test_ssc_monotonic_signal_no_reversals():
    assert ssc(np.array([1, 2, 3, 4, 5])) == 0
```

Update import:
```python
from features import rms, mav, sd, wl, var, zc, ssc
```

- [ ] **Step 2: Rodar, confirmar FAIL**

Run: `python -m pytest test_features.py -v`
Expected: failures for new tests.

- [ ] **Step 3: Implementar `zc` e `ssc`**

```python
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
```

- [ ] **Step 4: Rodar, confirmar passa**

Run: `python -m pytest test_features.py -v`
Expected: `15 passed`

- [ ] **Step 5: Escrever teste de `wamp`**

```python
def test_wamp_counts_amplitude_jumps():
    # [0, 1, 0, 5, 0] -> diffs = [1, -1, 5, -5] -> abs = [1,1,5,5]
    # with threshold 2: 2 jumps (the 5s)
    assert wamp(np.array([0, 1, 0, 5, 0]), threshold=2.0) == 2
```

Update import: `from features import rms, mav, sd, wl, var, zc, ssc, wamp`.

- [ ] **Step 6: Implementar `wamp`**

```python
def wamp(x: np.ndarray, threshold: float) -> int:
    """Willison Amplitude — number of consecutive |diff| > threshold."""
    return int(np.sum(np.abs(np.diff(x.astype(float))) > threshold))
```

- [ ] **Step 7: Rodar testes**

Run: `python -m pytest test_features.py -v`
Expected: `16 passed`

- [ ] **Step 8: Commit**

```bash
git add tcc/treinamento/features.py tcc/treinamento/test_features.py
git commit -m "feat(treinamento): add ZC/SSC/WAMP spectral-domain features"
```

---

## Task 4: Implementar `filter.py` (notch + passa-faixa)

**Files:**
- Create: `tcc/treinamento/filter.py`
- Modify: `tcc/treinamento/test_features.py` (renomear pra test_treinamento.py? — não, mantém features ali, cria test pra filter separado)
- Create: `tcc/treinamento/test_filter.py`

- [ ] **Step 1: Criar `filter.py`**

```python
"""Offline EMG filtering — matches the TCC's pipeline.

Notch 60 Hz (Q=30) + Butterworth band-pass 20–240 Hz, 4th order,
applied zero-phase via filtfilt / sosfiltfilt.
"""

import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, sosfiltfilt


def filter_emg(signal: np.ndarray, fs: int = 500,
               notch_freq: float = 60.0, notch_q: float = 30.0,
               lowcut: float = 20.0, highcut: float = 240.0,
               order: int = 4) -> np.ndarray:
    """Apply notch + band-pass to an EMG signal, zero-phase."""
    x = signal.astype(float)
    # 1. Notch (IIR 2nd order) via filtfilt
    b, a = iirnotch(notch_freq, notch_q, fs)
    x = filtfilt(b, a, x)
    # 2. Band-pass Butterworth via sosfiltfilt (numerically stable)
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)
```

- [ ] **Step 2: Criar `test_filter.py`**

```python
import numpy as np

from filter import filter_emg


def test_filter_attenuates_60hz():
    # Pure 60 Hz sine at fs=500 should be massively attenuated
    fs = 500
    t = np.arange(0, 1, 1 / fs)
    x = np.sin(2 * np.pi * 60 * t)
    y = filter_emg(x, fs=fs)
    # Ignore the first/last 50 samples (edge effects of filtfilt)
    assert np.max(np.abs(y[50:-50])) < 0.1


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
```

- [ ] **Step 3: Rodar testes, confirmar passa**

Run:
```bash
cd tcc/treinamento && python -m pytest -v
```
Expected: 19 passed (16 features + 3 filter).

- [ ] **Step 4: Commit**

```bash
git add tcc/treinamento/filter.py tcc/treinamento/test_filter.py
git commit -m "feat(treinamento): add offline EMG filter (notch 60 + bandpass 20-240)"
```

---

## Task 5: Implementar carregamento de dados e rotulagem

**Files:**
- Create: `tcc/treinamento/data.py`
- Create: `tcc/treinamento/test_data.py`

- [ ] **Step 1: Criar `data.py` com função `load_csv`**

```python
"""Load and label EMG recordings from rec_emg/.

Each recording is 30 s × 500 Hz with prompts at 0/5/10/15/20/25 s
alternating: open (0s) → closed (5s) → open (10s) → closed (15s) →
open (20s) → closed (25s).
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

FS = 500
DURATION = 30
PROMPT_TIMES = [0, 6, 11, 16, 21, 25]
PROMPT_LABELS = [0, 1, 0, 1, 0, 1]   # 0 = aberta, 1 = fechada


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CSV; return (signal, per-sample labels).

    Labels are derived from the prompt schedule: each sample belongs to
    the prompt interval containing its timestamp.
    """
    df = pd.read_csv(path)
    if "EMG_Value" not in df.columns:
        raise ValueError(f"{path}: missing EMG_Value column")
    sig = df["EMG_Value"].values.astype(float)
    n = len(sig)
    # Build per-sample timestamps from index (more robust than reading Tempo column)
    t = np.arange(n) / FS
    labels = np.empty(n, dtype=np.int8)
    for i, ts in enumerate(PROMPT_TIMES):
        end = PROMPT_TIMES[i + 1] if i + 1 < len(PROMPT_TIMES) else DURATION
        mask = (t >= ts) & (t < end)
        labels[mask] = PROMPT_LABELS[i]
    return sig, labels


def list_csvs(dir_path: str = "rec_emg") -> List[str]:
    """Return sorted absolute paths of rec_emg/new_emg_data*.csv."""
    import glob
    pattern = os.path.join(dir_path, "new_emg_data*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No CSVs found at {pattern}. Aborting.")
    return paths
```

- [ ] **Step 2: Criar `test_data.py`**

```python
import numpy as np
import os
import tempfile
import pandas as pd

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
        # Sample at t=2.5s (middle of first interval, mão aberta) -> 0
        assert labels[int(2.5 * FS)] == 0
        # Sample at t=7.5s (middle of second interval, mão fechada) -> 1
        assert labels[int(7.5 * FS)] == 1
        # Sample at t=27.5s (middle of last interval, mão fechada) -> 1
        assert labels[int(27.5 * FS)] == 1
    finally:
        os.unlink(path)
```

- [ ] **Step 3: Rodar testes**

Run: `cd tcc/treinamento && python -m pytest -v`
Expected: 21 passed.

- [ ] **Step 4: Commit**

```bash
git add tcc/treinamento/data.py tcc/treinamento/test_data.py
git commit -m "feat(treinamento): add CSV loader with per-sample label derivation"
```

---

## Task 6: Implementar janelamento com margem de transição

**Files:**
- Modify: `tcc/treinamento/data.py`
- Modify: `tcc/treinamento/test_data.py`

- [ ] **Step 1: Escrever teste de janelamento**

Append to `test_data.py`:

```python
from data import make_windows, WINDOW_SIZE, STEP_SIZE, TRANSITION_MARGIN_SAMPLES


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
    # (no mixed-label windows)
    for win_start, label in zip(range(0, n - WINDOW_SIZE, STEP_SIZE)[: len(X)], y):
        win = labels[win_start:win_start + WINDOW_SIZE]
        if len(set(win)) > 1:
            # this window straddled a transition; shouldn't be in X/y
            assert False, f"window at {win_start} has mixed labels"


def test_make_windows_excludes_margin():
    """The 250 samples around each transition should produce no windows."""
    n = FS * DURATION
    sig = np.zeros(n)
    labels = np.empty(n, dtype=np.int8)
    labels[:2500] = 0
    labels[2500:] = 1
    X, y = make_windows(sig, labels)
    # No window should start in [2500 - margin - WINDOW_SIZE + 1, 2500 + margin)
    # since those would either straddle the transition or overlap the margin.
    # Just confirm the transition itself never appears in a window's labels.
    assert len(X) >= 1
```

- [ ] **Step 2: Rodar, confirmar FAIL**

Expected: ImportError for `make_windows`.

- [ ] **Step 3: Adicionar janelamento em `data.py`**

Append:
```python
WINDOW_SIZE = 100              # 200 ms at 500 Hz, matches prediction.py
STEP_SIZE = 50                 # 50% overlap
TRANSITION_MARGIN_SAMPLES = 250   # 500 ms each side of a label change


def make_windows(signal: np.ndarray,
                 labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window of WINDOW_SIZE samples with STEP_SIZE step.

    Skips any window whose start lies within TRANSITION_MARGIN_SAMPLES
    of a label change, and any window that internally spans more than
    one label (defensive — shouldn't occur given the margin).
    """
    n = len(signal)
    assert n == len(labels)
    # Indexes where label changes (transitions)
    transitions = np.where(np.diff(labels) != 0)[0] + 1   # index of new-label start
    transition_zones = []
    for t in transitions:
        transition_zones.append((t - TRANSITION_MARGIN_SAMPLES,
                                 t + TRANSITION_MARGIN_SAMPLES))

    def in_transition_zone(start: int) -> bool:
        end = start + WINDOW_SIZE
        for lo, hi in transition_zones:
            if start < hi and end > lo:
                return True
        return False

    Xs, ys = [], []
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        if in_transition_zone(start):
            continue
        win = signal[start:start + WINDOW_SIZE]
        win_labels = labels[start:start + WINDOW_SIZE]
        if win_labels.min() != win_labels.max():
            continue          # mixed-label window (defensive)
        Xs.append(win)
        ys.append(int(win_labels[0]))
    return np.asarray(Xs), np.asarray(ys, dtype=np.int8)
```

- [ ] **Step 4: Rodar tudo**

Run: `cd tcc/treinamento && python -m pytest -v`
Expected: 23 passed.

- [ ] **Step 5: Commit**

```bash
git add tcc/treinamento/data.py tcc/treinamento/test_data.py
git commit -m "feat(treinamento): window slicer with transition-margin filtering"
```

---

## Task 7: Função de extração de features por janela

**Files:**
- Modify: `tcc/treinamento/data.py`
- Modify: `tcc/treinamento/test_data.py`

- [ ] **Step 1: Escrever teste**

```python
from data import extract_features

def test_extract_features_returns_right_shape():
    X = np.random.randn(5, 100).astype(float)
    feature_names = ["rms", "mav", "sd", "wl"]
    F = extract_features(X, feature_names)
    assert F.shape == (5, 4)


def test_extract_features_supports_zc_ssc():
    X = np.random.randn(3, 100).astype(float)
    F = extract_features(X, ["rms", "mav", "sd", "wl", "zc", "ssc"])
    assert F.shape == (3, 6)
```

- [ ] **Step 2: Implementar em `data.py`**

Append:
```python
from typing import Sequence
import features as feat_module


FEATURE_FUNCS = {
    "rms": feat_module.rms,
    "mav": feat_module.mav,
    "sd": feat_module.sd,
    "wl": feat_module.wl,
    "var": feat_module.var,
    "zc": feat_module.zc,
    "ssc": feat_module.ssc,
    # wamp omitted from default registry — it needs a per-call threshold
}


def extract_features(windows: np.ndarray,
                     feature_names: Sequence[str]) -> np.ndarray:
    """Apply each feature function to each window. Returns (N_windows, N_features)."""
    out = np.zeros((len(windows), len(feature_names)))
    for j, name in enumerate(feature_names):
        fn = FEATURE_FUNCS[name]
        for i, w in enumerate(windows):
            out[i, j] = fn(w)
    return out
```

- [ ] **Step 3: Rodar**

Run: `cd tcc/treinamento && python -m pytest -v`
Expected: 25 passed.

- [ ] **Step 4: Commit**

```bash
git add tcc/treinamento/data.py tcc/treinamento/test_data.py
git commit -m "feat(treinamento): per-window feature extraction registry"
```

---

## Task 8: Pipeline de treino single-config (sem sweep ainda)

**Files:**
- Create: `tcc/treinamento/train.py`

- [ ] **Step 1: Criar `train.py` com função `train_one_config`**

```python
"""Training orchestrator for the EMG binary classifier.

Loads all rec_emg/ CSVs, filters them, slices into labeled windows,
extracts features, then runs Leave-One-Group-Out CV on each (feature_set,
max_depth) config in a sweep. Picks the best, retrains on all data,
saves artifacts, and regenerates prediction.py.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix

# allow running as `python train.py` from tcc/treinamento/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data import load_csv, list_csvs, make_windows, extract_features  # noqa: E402
from filter import filter_emg  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # .../proj-emg


@dataclass
class Dataset:
    X: np.ndarray         # (N, n_features) features
    y: np.ndarray         # (N,) labels 0/1
    groups: np.ndarray    # (N,) group id per window (= CSV index)


def build_dataset(csv_paths: List[str], feature_names: List[str]) -> Dataset:
    """Load + filter + window + extract features from every CSV."""
    all_X, all_y, all_g = [], [], []
    for g, path in enumerate(csv_paths):
        sig, labels = load_csv(path)
        filtered = filter_emg(sig)
        wins, win_y = make_windows(filtered, labels)
        feats = extract_features(wins, feature_names)
        all_X.append(feats)
        all_y.append(win_y)
        all_g.append(np.full(len(win_y), g, dtype=np.int8))
    return Dataset(
        X=np.vstack(all_X),
        y=np.concatenate(all_y),
        groups=np.concatenate(all_g),
    )


def logo_accuracy(ds: Dataset, max_depth) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Run LOGO CV; return mean accuracy, std accuracy, aggregated y_true and y_pred."""
    logo = LeaveOneGroupOut()
    fold_acc = []
    y_true_all, y_pred_all = [], []
    for tr, te in logo.split(ds.X, ds.y, ds.groups):
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(ds.X[tr], ds.y[tr])
        y_pred = clf.predict(ds.X[te])
        fold_acc.append(accuracy_score(ds.y[te], y_pred))
        y_true_all.append(ds.y[te])
        y_pred_all.append(y_pred)
    return (float(np.mean(fold_acc)),
            float(np.std(fold_acc)),
            np.concatenate(y_true_all),
            np.concatenate(y_pred_all))


if __name__ == "__main__":
    csvs = list_csvs(os.path.join(PROJECT_ROOT, "rec_emg"))
    print(f"Loaded {len(csvs)} CSVs from rec_emg/")
    ds = build_dataset(csvs, ["rms", "mav", "sd", "wl"])
    print(f"Dataset: {ds.X.shape[0]} windows × {ds.X.shape[1]} features")
    mean_acc, std_acc, y_true, y_pred = logo_accuracy(ds, max_depth=5)
    print(f"Baseline (4 features, max_depth=5): {mean_acc:.3f} ± {std_acc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
```

- [ ] **Step 2: Rodar `train.py`, verificar baseline funciona**

Run:
```bash
cd tcc/treinamento && python train.py
```
Expected: imprime acurácia média, std, matriz de confusão. Sem crash.

(Acurácia esperada: alguma coisa entre ~75% e ~85% via LOGO.)

- [ ] **Step 3: Commit**

```bash
git add tcc/treinamento/train.py
git commit -m "feat(treinamento): single-config training + LOGO CV"
```

---

## Task 9: Sweep de hiperparâmetros

**Files:**
- Modify: `tcc/treinamento/train.py`

- [ ] **Step 1: Adicionar configurações do sweep em `train.py`**

Add (after the imports, before `__main__`):

```python
FEATURE_SETS = {
    "baseline":      ["rms", "mav", "sd", "wl"],
    "+ZC":           ["rms", "mav", "sd", "wl", "zc"],
    "+SSC":          ["rms", "mav", "sd", "wl", "ssc"],
    "+ZC+SSC":       ["rms", "mav", "sd", "wl", "zc", "ssc"],
    "+VAR":          ["rms", "mav", "sd", "wl", "var"],
    "tcc-1000hz":    ["rms", "mav", "wl", "ssc", "zc"],   # set used at 1000Hz in TCC
}
MAX_DEPTHS = [3, 5, 7, 10, None]


@dataclass
class ConfigResult:
    feature_set: str
    max_depth: object        # int or None
    mean_acc: float
    std_acc: float
    n_features: int

    @property
    def depth_str(self) -> str:
        return "None" if self.max_depth is None else str(self.max_depth)


def run_sweep(csv_paths: List[str]) -> List[ConfigResult]:
    """Run every (feature_set, max_depth) combination. Returns list of results."""
    results = []
    # Pre-build datasets per feature_set (so we don't re-filter for each depth)
    datasets = {}
    for name, feats in FEATURE_SETS.items():
        print(f"  Building dataset for feature_set={name} ...")
        datasets[name] = build_dataset(csv_paths, feats)
    # Sweep
    for name, feats in FEATURE_SETS.items():
        for depth in MAX_DEPTHS:
            ds = datasets[name]
            mean_acc, std_acc, _, _ = logo_accuracy(ds, max_depth=depth)
            r = ConfigResult(
                feature_set=name, max_depth=depth,
                mean_acc=mean_acc, std_acc=std_acc, n_features=len(feats),
            )
            results.append(r)
            print(f"  feature_set={name:<12} max_depth={r.depth_str:<5} "
                  f"→ {r.mean_acc:.3f} ± {r.std_acc:.3f}")
    return results


def select_winner(results: List[ConfigResult]) -> ConfigResult:
    """Highest mean accuracy, tiebreak by std, then n_features, then depth."""
    def sort_key(r: ConfigResult):
        depth = 999 if r.max_depth is None else r.max_depth
        return (-r.mean_acc, r.std_acc, r.n_features, depth)
    return sorted(results, key=sort_key)[0]
```

Replace the `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    csvs = list_csvs(os.path.join(PROJECT_ROOT, "rec_emg"))
    print(f"Loaded {len(csvs)} CSVs from rec_emg/")
    results = run_sweep(csvs)
    winner = select_winner(results)
    print()
    print(f"Winner: feature_set={winner.feature_set}, max_depth={winner.depth_str}")
    print(f"  Mean LOGO accuracy: {winner.mean_acc:.3f} ± {winner.std_acc:.3f}")
```

- [ ] **Step 2: Rodar sweep**

Run: `cd tcc/treinamento && python train.py`
Expected: imprime 30 linhas de config (6 feature sets × 5 depths) e o winner. Demora ~1-3 min.

- [ ] **Step 3: Commit**

```bash
git add tcc/treinamento/train.py
git commit -m "feat(treinamento): hyperparameter sweep (6 feature sets x 5 depths)"
```

---

## Task 10: Gerar `metrics.txt` com top-5 + winner

**Files:**
- Modify: `tcc/treinamento/train.py`

- [ ] **Step 1: Adicionar `write_metrics`**

Add (before `__main__`):

```python
from datetime import datetime
from sklearn.metrics import classification_report

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def write_metrics(results: List[ConfigResult], winner: ConfigResult,
                  ds_winner: Dataset, y_true: np.ndarray, y_pred: np.ndarray):
    """Write metrics.txt with top 5 + winner + classification report."""
    ensure_results_dir()
    out = os.path.join(RESULTS_DIR, "metrics.txt")

    # Sort by accuracy descending
    def sort_key(r):
        depth = 999 if r.max_depth is None else r.max_depth
        return (-r.mean_acc, r.std_acc, r.n_features, depth)
    sorted_results = sorted(results, key=sort_key)
    top5 = sorted_results[:5]

    with open(out, "w") as f:
        f.write(f"# Sweep results — Classificador EMG 500 Hz\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total configs tested: {len(results)}\n")
        f.write(f"# Dataset: {ds_winner.X.shape[0]} windows × {ds_winner.X.shape[1]} features\n")
        f.write(f"\n")
        f.write(f"## Top 5 configs (by mean LOGO accuracy)\n\n")
        f.write(f"{'#':<4}{'feature_set':<14}{'depth':<8}{'n_feat':<8}{'mean_acc':<12}{'std_acc':<10}\n")
        for i, r in enumerate(top5, start=1):
            f.write(f"{i:<4}{r.feature_set:<14}{r.depth_str:<8}{r.n_features:<8}"
                    f"{r.mean_acc:.4f}      {r.std_acc:.4f}\n")
        f.write(f"\n")
        f.write(f"## Winner\n\n")
        f.write(f"feature_set: {winner.feature_set}\n")
        f.write(f"max_depth:   {winner.depth_str}\n")
        f.write(f"n_features:  {winner.n_features}\n")
        f.write(f"mean_acc:    {winner.mean_acc:.4f}\n")
        f.write(f"std_acc:     {winner.std_acc:.4f}\n")
        f.write(f"\n")
        f.write(f"## Classification report (LOGO aggregated)\n\n")
        f.write(classification_report(y_true, y_pred,
                                       target_names=["aberta (0)", "fechada (1)"],
                                       digits=4))
        cm = confusion_matrix(y_true, y_pred)
        f.write(f"\n## Confusion matrix (absolute counts)\n\n")
        f.write(f"             Predicted aberta  Predicted fechada\n")
        f.write(f"True aberta       {cm[0,0]:>5}             {cm[0,1]:>5}\n")
        f.write(f"True fechada      {cm[1,0]:>5}             {cm[1,1]:>5}\n")
    print(f"Saved {out}")
```

Update `__main__` block — add after winner selection:

```python
    # Re-run LOGO on the winner config to get aggregated y_true/y_pred for metrics
    ds_winner = build_dataset(csvs, FEATURE_SETS[winner.feature_set])
    _, _, y_true, y_pred = logo_accuracy(ds_winner, max_depth=winner.max_depth)
    write_metrics(results, winner, ds_winner, y_true, y_pred)
```

- [ ] **Step 2: Rodar e inspecionar `metrics.txt`**

Run: `cd tcc/treinamento && python train.py`
Expected: `results/metrics.txt` aparece com top 5, winner, classification report, matriz de confusão.

- [ ] **Step 3: Commit**

```bash
git add tcc/treinamento/train.py tcc/treinamento/results/metrics.txt
git commit -m "feat(treinamento): write metrics.txt with top 5 + winner + classification report"
```

---

## Task 11: Figura — matriz de confusão

**Files:**
- Modify: `tcc/treinamento/train.py`

- [ ] **Step 1: Adicionar `plot_confusion_matrix`**

Add (before `__main__`):

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred):
    """Save absolute + normalized confusion matrix side by side."""
    ensure_results_dir()
    cm_abs = confusion_matrix(y_true, y_pred)
    cm_norm = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title, fmt in [
        (axes[0], cm_abs, "Absolute counts", "d"),
        (axes[1], cm_norm, "Normalized per true class", ".2f"),
    ]:
        im = ax.imshow(mat, cmap="Blues", aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["aberta", "fechada"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["aberta", "fechada"])
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                txt = format(val, fmt)
                color = "white" if val > mat.max() * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    for ext in ("png", "svg"):
        out = os.path.join(RESULTS_DIR, f"confusion_matrix.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)
```

Update `__main__` — add after `write_metrics`:
```python
    plot_confusion_matrix(y_true, y_pred)
```

- [ ] **Step 2: Rodar e ver figura**

Run: `cd tcc/treinamento && python train.py`
Verify: `results/confusion_matrix.{png,svg}` aparecem.

- [ ] **Step 3: Commit**

```bash
git add tcc/treinamento/train.py tcc/treinamento/results/
git commit -m "feat(treinamento): confusion matrix figure (absolute + normalized)"
```

---

## Task 12: Figuras — árvore de decisão e feature importance

**Files:**
- Modify: `tcc/treinamento/train.py`

- [ ] **Step 1: Adicionar funções de plot**

Add (before `__main__`):

```python
from sklearn.tree import plot_tree


def fit_final_model(ds: Dataset, max_depth) -> DecisionTreeClassifier:
    """Train on all data (no CV) — used for export."""
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(ds.X, ds.y)
    return clf


def plot_decision_tree(clf: DecisionTreeClassifier, feature_names: List[str]):
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(clf, ax=ax, feature_names=feature_names,
              class_names=["aberta", "fechada"], filled=True, rounded=True,
              fontsize=9)
    plt.tight_layout()
    for ext in ("png", "svg"):
        out = os.path.join(RESULTS_DIR, f"decision_tree.{ext}")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def plot_feature_importance(clf: DecisionTreeClassifier, feature_names: List[str]):
    ensure_results_dir()
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(imp)), imp[order], color="tab:blue")
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=0)
    ax.set_ylabel("Importance")
    ax.set_title("Decision Tree — feature importance")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "svg"):
        out = os.path.join(RESULTS_DIR, f"feature_importance.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)
```

Update `__main__` — add after `plot_confusion_matrix`:
```python
    clf_final = fit_final_model(ds_winner, winner.max_depth)
    winning_features = FEATURE_SETS[winner.feature_set]
    plot_decision_tree(clf_final, winning_features)
    plot_feature_importance(clf_final, winning_features)
```

- [ ] **Step 2: Rodar e inspecionar figuras**

Run: `cd tcc/treinamento && python train.py`
Verify: `results/decision_tree.{png,svg}`, `results/feature_importance.{png,svg}`.

- [ ] **Step 3: Commit**

```bash
git add tcc/treinamento/train.py tcc/treinamento/results/
git commit -m "feat(treinamento): decision tree + feature importance figures"
```

---

## Task 13: Salvar `.pkl`

**Files:**
- Modify: `tcc/treinamento/train.py`

- [ ] **Step 1: Adicionar `save_pkl`**

Add (before `__main__`):

```python
import joblib

PKL_PATH = os.path.join(SCRIPT_DIR, "dt_500hz.pkl")


def save_pkl(clf: DecisionTreeClassifier, feature_names: List[str]):
    """Save model + feature names in a single dict via joblib."""
    payload = {"model": clf, "features": feature_names, "fs": 500}
    joblib.dump(payload, PKL_PATH)
    print(f"Saved {PKL_PATH}")
```

Update `__main__` — add after `plot_feature_importance`:
```python
    save_pkl(clf_final, winning_features)
```

- [ ] **Step 2: Rodar e verificar**

Run:
```bash
cd tcc/treinamento && python train.py
python -c "import joblib; p=joblib.load('dt_500hz.pkl'); print(p['features'], type(p['model']).__name__)"
```
Expected: imprime as features e `DecisionTreeClassifier`.

- [ ] **Step 3: Commit**

```bash
git add tcc/treinamento/train.py tcc/treinamento/dt_500hz.pkl
git commit -m "feat(treinamento): export winning model as .pkl"
```

---

## Task 14: Template MicroPython e gerador de árvore inline

**Files:**
- Create: `tcc/treinamento/prediction_template.py`
- Modify: `tcc/treinamento/train.py`

- [ ] **Step 1: Criar `prediction_template.py`**

```python
"""Template for the regenerated prediction.py at the project root.

Placeholders (in curly-brace string-format style) are filled in by train.py:
  {HEADER}                  - auto-generated metadata comment
  {NUM_AMOSTRAS}            - window size in samples
  {FEATURE_FUNCTIONS}       - bodies of the feature funcs used by the model
  {PREDICT_SIGNATURE}       - prever_movimento(rms, mav, ...) — kwargs only
  {PREDICT_BODY}            - the decision tree as nested if/else (indented 4 spaces)
  {FEATURE_COMPUTE_BLOCK}   - lines that compute the feature kwargs from the window
"""

TEMPLATE = '''\
{HEADER}
from machine import ADC, Pin
import time
import math

# ---------- Parâmetros ----------
NUM_AMOSTRAS = {NUM_AMOSTRAS}
TAMANHO_FILTRO = 5

# ---------- Inicialização ----------
adc = ADC(26)

# ---------- Feature functions ----------
{FEATURE_FUNCTIONS}

# ---------- Decision tree (auto-generated) ----------
def prever_movimento({PREDICT_SIGNATURE}):
{PREDICT_BODY}

# ---------- Funções auxiliares ----------
def media_movel(dados, tamanho):
    filtrado = []
    for i in range(len(dados)):
        janela = dados[max(0, i - tamanho + 1):i + 1]
        filtrado.append(sum(janela) / len(janela))
    return filtrado

# ---------- Loop principal ----------
print("Iniciando leitura EMG com filtro digital...")

while True:
    janela = []
    for _ in range(NUM_AMOSTRAS):
        leitura = adc.read_u16() >> 4   # 12-bit (0-4095)
        janela.append(leitura * 10)     # amplification (matches original code)
        time.sleep(1/500)
    filtrado = media_movel(janela, TAMANHO_FILTRO)

{FEATURE_COMPUTE_BLOCK}

    classe = prever_movimento({PREDICT_KWARGS})
    if classe == 1:
        print("🖐️  Mão FECHADA")
    else:
        print("✋ Mão ABERTA")
    time.sleep(0.1)
'''
```

- [ ] **Step 2: Adicionar exporter em `train.py`**

Add at top of `train.py` (after other imports):
```python
from sklearn.tree import _tree
```

Add helper functions (before `__main__`):

```python
# Feature function bodies in MicroPython-friendly form (no numpy on the Pico)
MICROPYTHON_FEATURE_BODIES = {
    "rms": (
        "def calcula_rms(valores):\n"
        "    return math.sqrt(sum(x*x for x in valores) / len(valores))\n"
    ),
    "mav": (
        "def calcula_mav(valores):\n"
        "    return sum(abs(x) for x in valores) / len(valores)\n"
    ),
    "sd": (
        "def calcula_sd(valores):\n"
        "    m = sum(valores) / len(valores)\n"
        "    return math.sqrt(sum((x-m)*(x-m) for x in valores) / len(valores))\n"
    ),
    "wl": (
        "def calcula_wl(valores):\n"
        "    return sum(abs(valores[i] - valores[i-1]) for i in range(1, len(valores)))\n"
    ),
    "var": (
        "def calcula_var(valores):\n"
        "    m = sum(valores) / len(valores)\n"
        "    return sum((x-m)*(x-m) for x in valores) / len(valores)\n"
    ),
    "zc": (
        "def calcula_zc(valores, thresh=0):\n"
        "    n = 0\n"
        "    for i in range(1, len(valores)):\n"
        "        if valores[i-1] * valores[i] < 0 and abs(valores[i]-valores[i-1]) > thresh:\n"
        "            n += 1\n"
        "    return n\n"
    ),
    "ssc": (
        "def calcula_ssc(valores, thresh=0):\n"
        "    n = 0\n"
        "    for i in range(2, len(valores)):\n"
        "        d1 = valores[i-1] - valores[i-2]\n"
        "        d2 = valores[i]   - valores[i-1]\n"
        "        if d1 * d2 < 0 and abs(d2-d1) > thresh:\n"
        "            n += 1\n"
        "    return n\n"
    ),
}


def export_tree_as_ifelse(clf: DecisionTreeClassifier,
                          feature_names: List[str], indent: int = 4) -> str:
    """Convert sklearn DT into nested if/else MicroPython source."""
    tree = clf.tree_
    pad = " " * indent

    def recurse(node: int, depth: int) -> List[str]:
        lines = []
        ind = " " * (indent + depth * 4)
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            thr = tree.threshold[node]
            lines.append(f"{ind}if {name} <= {thr:.6f}:")
            lines.extend(recurse(tree.children_left[node], depth + 1))
            lines.append(f"{ind}else:")
            lines.extend(recurse(tree.children_right[node], depth + 1))
        else:
            counts = tree.value[node][0]
            cls = int(np.argmax(counts))
            lines.append(f"{ind}return {cls}")
        return lines

    return "\n".join(recurse(0, 0))
```

- [ ] **Step 3: Teste rápido manual**

Run quick smoke test:
```bash
cd tcc/treinamento && python -c "
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from train import export_tree_as_ifelse
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
X = np.random.randn(50, 2); y = (X[:,0] > 0).astype(int)
clf.fit(X, y)
print(export_tree_as_ifelse(clf, ['a','b']))
"
```
Expected: imprime if/else aninhado com `if a <= 0.XXX:` etc.

- [ ] **Step 4: Commit**

```bash
git add tcc/treinamento/prediction_template.py tcc/treinamento/train.py
git commit -m "feat(treinamento): MicroPython template + sklearn-to-ifelse exporter"
```

---

## Task 15: Regeneração do `prediction.py`

**Files:**
- Modify: `tcc/treinamento/train.py`
- Modify: `prediction.py` (na raiz — regerado)

- [ ] **Step 1: Adicionar `regenerate_prediction_py` em train.py**

Add (before `__main__`):

```python
from prediction_template import TEMPLATE

PREDICTION_PY = os.path.join(PROJECT_ROOT, "prediction.py")


def build_header(winner: ConfigResult, features: List[str]) -> str:
    return (
        f"# AUTO-GENERATED by tcc/treinamento/train.py at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Model: DecisionTreeClassifier, max_depth={winner.depth_str}, "
        f"random_state=42\n"
        f"# Features: {', '.join(features)}\n"
        f"# LOGO CV accuracy: {winner.mean_acc:.4f} ± {winner.std_acc:.4f}\n"
        f"# Trained on: rec_emg/new_emg_data{{1..10}}.csv\n"
    )


def regenerate_prediction_py(clf: DecisionTreeClassifier, winner: ConfigResult,
                              features: List[str], window_size: int):
    header = build_header(winner, features)
    feat_bodies = "".join(MICROPYTHON_FEATURE_BODIES[f] for f in features)
    signature = ", ".join(features)              # e.g. "rms, mav, sd, wl"
    body = export_tree_as_ifelse(clf, features, indent=4)
    # Block in the sample loop that computes each feature kwarg
    compute_lines = []
    for f in features:
        if f in ("zc", "ssc"):
            compute_lines.append(f"    {f} = calcula_{f}(filtrado)")
        else:
            compute_lines.append(f"    {f} = calcula_{f}(filtrado)")
    compute_block = "\n".join(compute_lines)
    kwargs = ", ".join(f"{f}={f}" for f in features)

    rendered = TEMPLATE.format(
        HEADER=header,
        NUM_AMOSTRAS=window_size,
        FEATURE_FUNCTIONS=feat_bodies.rstrip(),
        PREDICT_SIGNATURE=signature,
        PREDICT_BODY=body,
        FEATURE_COMPUTE_BLOCK=compute_block,
        PREDICT_KWARGS=kwargs,
    )
    with open(PREDICTION_PY, "w") as f:
        f.write(rendered)
    print(f"Regenerated {PREDICTION_PY}")
```

Update `__main__` — add after `save_pkl`:

```python
    from data import WINDOW_SIZE
    regenerate_prediction_py(clf_final, winner, winning_features, WINDOW_SIZE)
```

- [ ] **Step 2: Rodar e inspecionar o novo `prediction.py`**

Run:
```bash
cd tcc/treinamento && python train.py
head -20 ../../prediction.py
```
Expected: header com timestamp/features/accuracy, depois imports + funções.

- [ ] **Step 3: Sintaxe Python do prediction.py é válida**

Run:
```bash
python -c "import ast; ast.parse(open('prediction.py').read()); print('OK')"
```
Expected: `OK`. (Não vai *rodar* sem MicroPython, mas o syntax check confirma que o gerador escreveu Python válido.)

- [ ] **Step 4: Commit**

```bash
git add tcc/treinamento/train.py prediction.py
git commit -m "feat(treinamento): regenerate prediction.py from trained model"
```

---

## Task 16: README do `tcc/treinamento/`

**Files:**
- Create: `tcc/treinamento/README.md`

- [ ] **Step 1: Escrever README com todas as decisões da spec**

```markdown
# `tcc/treinamento/` — pipeline de treino do classificador EMG (500 Hz)

Pipeline reproduzível para o classificador binário (mão aberta / mão
fechada) descrito no TCC. Carrega as 10 gravações em `rec_emg/`, aplica
filtragem offline, fatia em janelas rotuladas, extrai features, roda um
sweep de 30 configs com Leave-One-Group-Out CV, escolhe a vencedora,
treina o modelo final, salva artefatos e regenera `prediction.py` na
raiz do repositório.

**Spec do design**: `docs/superpowers/specs/2026-05-23-classificador-500hz-design.md`

## Como rodar

```bash
source ../../.venv/bin/activate
cd tcc/treinamento
python train.py
```

Demora ~1-3 minutos. Saídas em `results/` + `dt_500hz.pkl` + `prediction.py` (na raiz).

## Decisões de projeto

### Dados
- Origem: `rec_emg/new_emg_data{1..10}.csv` — 10 CSVs, todos a 500 Hz, 30 s cada.
- Protocolo de prompts: 0s aberta → 5s fecha → 10s abre → 15s fecha → 20s abre → 25s fecha.

### Filtragem (offline, zero-phase, scipy)
- Notch 60 Hz, Q=30, via `filtfilt`.
- Passa-faixa Butterworth 4ª ordem, 20–240 Hz, via `sosfiltfilt`.
- **Nota**: o filtro no `prediction.py` da Pico é só média móvel de 5 amostras. Mismatch herdado do TCC — fica como trabalho futuro.

### Janelamento
- Tamanho: 100 amostras = 200 ms (= `NUM_AMOSTRAS` do `prediction.py`).
- Sobreposição: 50% (step = 50 amostras).
- Margem de transição: 500 ms (= 250 amostras) descartados em cada lado de uma mudança de rótulo. Sobra ~4 s por intervalo de 5 s.
- ~240 janelas por CSV × 10 CSVs ≈ 2400 janelas totais.

### Features (`features.py`)
- `rms`, `mav`, `sd`, `wl`, `var`, `zc`, `ssc`, `wamp`.
- **Crítico**: `mav` ≠ `mean`. O `prediction.py` antigo usava `mean`; aqui usamos a definição correta com `abs()`. Testes em `test_features.py` cobrem esse caso.
- Thresholds de `zc`, `ssc`, `wamp`: começam em 0 (sem deadzone).

### Sweep
- 7 feature sets × 5 valores de `max_depth` = 35 configs.
- Cada uma roda 10-fold Leave-One-Group-Out CV (cada CSV é um group).
- Selector: maior `mean_acc` no LOGO, tiebreak por `std`, `n_features`, `max_depth`.

### Modelo final
- Re-treinado no dataset inteiro com a config vencedora.
- Salvo em `dt_500hz.pkl` (joblib, payload: model + features + fs).
- Exportado como if/else inline para `prediction.py`.

### Regeneração do `prediction.py`
- Template em `prediction_template.py`.
- Esqueleto MicroPython (ADC, loop, prints) preservado.
- Bloco modelo (features + árvore) regerado a cada `python train.py`.
- Header com timestamp + accuracy + features pra auditoria visual.

## Estrutura

```
tcc/treinamento/
├── README.md                       (este arquivo)
├── features.py                     funções puras de feature
├── filter.py                       design do filtro EMG
├── data.py                         loader + janelamento + extração
├── train.py                        orchestrator + sweep + export
├── prediction_template.py          template do prediction.py
├── test_features.py                testes unitários
├── test_filter.py                  testes do filtro
├── test_data.py                    testes de loader + janelas
├── dt_500hz.pkl                    modelo final (gerado)
└── results/                        gerados por train.py
    ├── metrics.txt                 top 5 configs + winner + relatório
    ├── confusion_matrix.{png,svg}  da config vencedora
    ├── decision_tree.{png,svg}     visualização da árvore
    └── feature_importance.{png,svg} importância de cada feature
```

## Rodar testes

```bash
cd tcc/treinamento && python -m pytest -v
```
```

- [ ] **Step 2: Commit**

```bash
git add tcc/treinamento/README.md
git commit -m "docs(treinamento): README with pipeline decisions and how-to-run"
```

---

## Task 17: Execução final + verificação

**Files:** nenhum novo

- [ ] **Step 1: Limpar artefatos gerados pra rodar do zero**

```bash
rm -f tcc/treinamento/dt_500hz.pkl
rm -rf tcc/treinamento/results/
```

- [ ] **Step 2: Rodar pipeline completo**

```bash
cd tcc/treinamento && python train.py
```
Expected:
- Sweep imprime 30 linhas
- Winner identificado
- Artefatos criados em `results/`
- `prediction.py` regerado

- [ ] **Step 3: Conferir todos os artefatos**

```bash
ls -la tcc/treinamento/results/
ls -la tcc/treinamento/dt_500hz.pkl
head -10 prediction.py
```
Expected: 7 arquivos em results/ (metrics.txt + 6 figuras), .pkl existe, prediction.py começa com o header auto-generated.

- [ ] **Step 4: Rodar testes uma última vez**

```bash
cd tcc/treinamento && python -m pytest -v
```
Expected: todos os testes passam.

- [ ] **Step 5: Commit final dos artefatos**

```bash
git add tcc/treinamento/dt_500hz.pkl tcc/treinamento/results/ prediction.py
git commit -m "feat(treinamento): end-to-end pipeline run — final artifacts"
```

- [ ] **Step 6: Push**

```bash
git push
```
