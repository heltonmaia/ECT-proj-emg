"""Training orchestrator for the EMG binary classifier.

Current scope (Task 9): loads all rec_emg/ CSVs, filters them, slices into
labeled windows, extracts features, then runs Leave-One-Group-Out CV across
a sweep of (feature_set, max_depth) configurations. Prints the sweep table
and identifies the winning config.

Planned (Tasks 10-15): write metrics.txt, generate figures (confusion matrix,
decision tree, feature importance), save the winning .pkl, and regenerate
prediction.py at the project root with the trained tree.
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# allow running as `python train.py` from tcc/treinamento/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data import load_csv, list_csvs, make_windows, extract_features, WINDOW_SIZE  # noqa: E402
from filter import filter_emg  # noqa: E402
from prediction_template import TEMPLATE  # noqa: E402

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


def logo_accuracy(ds: Dataset, max_depth: Optional[int]) -> Tuple[float, float, np.ndarray, np.ndarray]:
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


import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PKL_PATH = os.path.join(SCRIPT_DIR, "dt_500hz.pkl")


def save_pkl(clf: DecisionTreeClassifier, feature_names: List[str]):
    """Save model + feature names in a single dict via joblib."""
    payload = {"model": clf, "features": feature_names, "fs": 500}
    joblib.dump(payload, PKL_PATH)
    print(f"Saved {PKL_PATH}")


# MicroPython-friendly bodies (no numpy on the Pico).
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


PREDICTION_PY = os.path.join(PROJECT_ROOT, "prediction.py")


def build_header(winner: ConfigResult, features: List[str]) -> str:
    return (
        f"# AUTO-GENERATED by tcc/treinamento/train.py at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Model: DecisionTreeClassifier, max_depth={winner.depth_str}, "
        f"random_state=42\n"
        f"# Features: {', '.join(features)}\n"
        f"# LOGO CV accuracy: {winner.mean_acc:.4f} +/- {winner.std_acc:.4f}\n"
        f"# Trained on: rec_emg/new_emg_data{{1..10}}.csv\n"
    )


def regenerate_prediction_py(clf: DecisionTreeClassifier, winner: ConfigResult,
                              features: List[str], window_size: int):
    header = build_header(winner, features)
    feat_bodies = "".join(MICROPYTHON_FEATURE_BODIES[f] for f in features)
    signature = ", ".join(features)
    body = export_tree_as_ifelse(clf, features, indent=4)
    compute_block = "\n".join(f"    {f} = calcula_{f}(filtrado)" for f in features)
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


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred):
    """Save absolute + normalized confusion matrix side by side."""
    ensure_results_dir()
    cm_abs = confusion_matrix(y_true, y_pred)
    cm_norm = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title, fmt in [
        (axes[0], cm_abs, "Contagens absolutas", "d"),
        (axes[1], cm_norm, "Normalizada por classe real", ".2f"),
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


def write_metrics(results: List[ConfigResult], winner: ConfigResult,
                  ds_winner: Dataset, y_true: np.ndarray, y_pred: np.ndarray):
    """Write metrics.txt with top 5 + winner + classification report."""
    ensure_results_dir()
    out = os.path.join(RESULTS_DIR, "metrics.txt")

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
              impurity=False, fontsize=9)
    # sklearn renders node text in English; rewrite in place to PT
    for text in ax.texts:
        s = text.get_text()
        s = s.replace("samples = ", "amostras = ")
        s = s.replace("value = ", "valor = ")
        s = s.replace("class = ", "classe = ")
        text.set_text(s)
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
    ax.set_ylabel("Importância")
    ax.set_title("Árvore de decisão — importância das features")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "svg"):
        out = os.path.join(RESULTS_DIR, f"feature_importance.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    csvs = list_csvs(os.path.join(PROJECT_ROOT, "rec_emg"))
    print(f"Loaded {len(csvs)} CSVs from rec_emg/")
    results = run_sweep(csvs)
    winner = select_winner(results)
    print()
    print(f"Winner: feature_set={winner.feature_set}, max_depth={winner.depth_str}")
    print(f"  Mean LOGO accuracy: {winner.mean_acc:.3f} ± {winner.std_acc:.3f}")

    # Re-run LOGO on the winner config to aggregate y_true/y_pred for metrics
    ds_winner = build_dataset(csvs, FEATURE_SETS[winner.feature_set])
    _, _, y_true, y_pred = logo_accuracy(ds_winner, max_depth=winner.max_depth)
    write_metrics(results, winner, ds_winner, y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)

    clf_final = fit_final_model(ds_winner, winner.max_depth)
    winning_features = FEATURE_SETS[winner.feature_set]
    plot_decision_tree(clf_final, winning_features)
    plot_feature_importance(clf_final, winning_features)
    save_pkl(clf_final, winning_features)
    regenerate_prediction_py(clf_final, winner, winning_features, WINDOW_SIZE)
