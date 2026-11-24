"""Per-minute and per-recording evaluation on held-out SUBJECTS."""
from __future__ import annotations

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from src.data.apnea import CLASS_NAMES, record_class
from src.training.train import get_data
from src.utils.common import get_logger, load_json, project_path, save_json
from src.utils.plotting import INK_2, SERIES, plt, save

log = get_logger(__name__)
REC_CLASSES = ["A", "B", "C"]


def minute_metrics(y, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "precision_apnea": p[1], "recall_apnea_sensitivity": r[1], "f1_apnea": f[1],
            "precision_normal": p[0], "recall_normal_specificity": r[0], "f1_normal": f[0],
            "roc_auc": roc_auc_score(y, prob), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "n": int(len(y))}


def fit_record_thresholds(frac: np.ndarray, cls: np.ndarray) -> tuple[float, float]:
    """Choose two cut-points on the predicted apnea-minute fraction (C < t1 <= B < t2 <= A) that maximise
    record-level accuracy on TRAINING recordings."""
    cands = np.unique(np.r_[0, frac, 1])
    best, best_acc = (0.05, 0.2), -1
    for i, t1 in enumerate(cands):
        for t2 in cands[i:]:
            pred = np.where(frac >= t2, "A", np.where(frac >= t1, "B", "C"))
            acc = np.mean(pred == cls)
            if acc > best_acc:
                best, best_acc = (float(t1), float(t2)), acc
    return best


def classify_records(frac, t1, t2):
    return np.where(frac >= t2, "A", np.where(frac >= t1, "B", "C"))


def plot_cm(cm, labels, path, title):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "#0b0b0b", fontsize=10)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)
    save(fig, path)


def evaluate(cfg: dict) -> dict:
    res = project_path("results")
    model = keras.models.load_model(project_path("models", "alexnet_apnea.keras"))
    meta, split, data = get_data(cfg)
    probs = {k: model.predict(v[0].astype(np.float32), batch_size=256, verbose=0)[:, 1] for k, v in data.items()}

    metrics = {k: minute_metrics(data[k][1], probs[k]) for k in data}
    yte, pte = data["test"][1], probs["test"]
    report = classification_report(yte, (pte >= 0.5).astype(int), target_names=CLASS_NAMES, digits=4, output_dict=True)
    pd.DataFrame(report).T.to_csv(res / "metrics" / "test_classification_report.csv")
    (res / "reports" / "test_classification_report.txt").write_text(
        classification_report(yte, (pte >= 0.5).astype(int), target_names=CLASS_NAMES, digits=4))
    cm = confusion_matrix(yte, (pte >= 0.5).astype(int))
    plot_cm(cm, CLASS_NAMES, res / "graphs" / "confusion_matrix_minutes.png", "Per-minute test confusion matrix")

    # ---- per-record analysis (record diagnosis from predicted apnea-minute fraction)
    rows = []
    for k in data:
        X, y, recs, _ = data[k]
        for r in np.unique(recs):
            m = recs == r
            rows.append({"split": k, "record": r, "minutes": int(m.sum()), "true_apnea_fraction": float(y[m].mean()),
                         "pred_apnea_fraction": float((probs[k][m] >= 0.5).mean()),
                         "minute_accuracy": float(((probs[k][m] >= 0.5) == y[m]).mean()),
                         "true_class": record_class(meta.set_index("record").apnea_minutes[r])})
    recdf = pd.DataFrame(rows)
    tr = recdf[recdf.split == "train"]
    t1, t2 = fit_record_thresholds(tr.pred_apnea_fraction.to_numpy(), tr.true_class.to_numpy())
    recdf["pred_class"] = classify_records(recdf.pred_apnea_fraction.to_numpy(), t1, t2)
    recdf.to_csv(res / "metrics" / "per_record_results.csv", index=False)
    te = recdf[recdf.split == "test"]
    rec_cm = confusion_matrix(te.true_class, te.pred_class, labels=REC_CLASSES)
    plot_cm(rec_cm, REC_CLASSES, res / "graphs" / "confusion_matrix_records.png",
            "Record-level diagnosis (test subjects)")
    metrics["test_records"] = {"accuracy": float((te.true_class == te.pred_class).mean()), "n_records": int(len(te)),
                               "thresholds_fit_on_train": [t1, t2],
                               "apnea_fraction_correlation": float(np.corrcoef(te.true_apnea_fraction,
                                                                               te.pred_apnea_fraction)[0, 1])}
    save_json(metrics, res / "metrics" / "evaluation_metrics.json")
    pd.DataFrame(metrics["test"] | {"split": "test"}, index=[0]).to_csv(res / "metrics" / "test_metrics.csv", index=False)

    # ---- graphs
    fpr, tpr, _ = roc_curve(yte, pte)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color=SERIES[0], label=f"AlexNet (AUC {metrics['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=INK_2, lw=1, label="Chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (sensitivity)")
    ax.set_title("ROC, per-minute apnea detection (test)")
    ax.legend(loc="lower right")
    save(fig, res / "graphs" / "roc_curve.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, s in enumerate(["train", "val", "test"]):
        d = recdf[recdf.split == s]
        ax.scatter(d.true_apnea_fraction, d.pred_apnea_fraction, s=30, color=SERIES[i], label=s)
    ax.plot([0, 1], [0, 1], "--", color=INK_2, lw=1)
    ax.set_xlabel("Annotated apnea-minute fraction")
    ax.set_ylabel("Predicted apnea-minute fraction")
    ax.set_title("Per-recording apnea burden")
    ax.legend()
    save(fig, res / "graphs" / "record_apnea_burden.png")

    hist_path = res / "metrics" / "training_history.csv"
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key, lab in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "accuracy", "Accuracy")]:
            ax.plot(np.arange(1, len(h) + 1), h[key], color=SERIES[0], label="Training")
            ax.plot(np.arange(1, len(h) + 1), h[f"val_{key}"], color=SERIES[1], label="Validation")
            n1 = int((h.stage == "1_head").sum())
            ax.axvline(n1 + 0.5, color=INK_2, ls="--", lw=1, label="Start fine-tuning conv4-5")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(lab)
            ax.legend()
        axes[0].set_title("Loss")
        axes[1].set_title("Accuracy")
        save(fig, res / "graphs" / "training_curves.png")

    plot_samples(data["test"], pte, res / "graphs" / "sample_predictions.png")
    log.info("Test per-minute: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics["test"].items()})
    log.info("Test records: %s", metrics["test_records"])
    return metrics


def plot_samples(test, prob, path, n=8, seed=0):
    X, y, recs, mins = test
    rng = np.random.default_rng(seed)
    pred = (prob >= 0.5).astype(int)
    picks = []
    for t in (0, 1):
        for p in (0, 1):
            idx = np.nonzero((y == t) & (pred == p))[0]
            picks += list(rng.choice(idx, min(2, len(idx)), replace=False))
    fig, axes = plt.subplots(2, len(picks), figsize=(2.3 * len(picks), 4.6))
    for j, i in enumerate(picks):
        for row, (c, name) in enumerate([(0, "RR scalogram"), (1, "EDR scalogram")]):
            ax = axes[row, j]
            ax.imshow(X[i][..., c].astype(np.float32), aspect="auto", cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
        ok = "✓" if pred[i] == y[i] else "✗"
        axes[0, j].set_title(f"{recs[i]} min {mins[i]}\ntrue {CLASS_NAMES[y[i]]}\npred {CLASS_NAMES[pred[i]]} "
                             f"{prob[i]:.2f} {ok}", fontsize=8)
    fig.suptitle("Test-subject samples (rows: RR / EDR wavelet power, 0.5 Hz top → 0.01 Hz bottom, 5 min)", fontsize=10)
    save(fig, path)
