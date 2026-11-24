"""Prediction demo: raw ECG recording -> per-minute apnea probabilities -> recording-level class.

Usage: python scripts/predict_record.py --record x05
       python scripts/predict_record.py --record path/to/your_record   (WFDB record with .dat/.hea/.qrs[/.apn])
Research demo only. Not a medical device and not clinically validated.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402

from src.data.apnea import CLASS_NAMES, load_record  # noqa: E402
from src.data.features import record_images  # noqa: E402
from src.evaluation.evaluate import classify_records  # noqa: E402
from src.utils.common import load_config, load_json, project_path  # noqa: E402
from src.utils.plotting import CRITICAL, SERIES, plt, save  # noqa: E402

CLASS_TEXT = {"A": "A: apnea", "B": "B: borderline", "C": "C: control (no apnea)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="x05")
    args = ap.parse_args()
    cfg = load_config()
    path = Path(args.record)
    raw_dir, name = (path.parent, path.name) if path.parent != Path(".") else (project_path(cfg["data"]["raw_dir"]), args.record)
    rec = load_record(raw_dir, name)
    X, y, minutes = record_images(rec, cfg["signal"])
    model = keras.models.load_model(project_path("models", "alexnet_apnea.keras"))
    prob = model.predict(X.astype(np.float32), batch_size=256, verbose=0)[:, 1]
    pred = (prob >= 0.5).astype(int)
    t1, t2 = load_json(project_path("results", "metrics", "evaluation_metrics.json"))["test_records"]["thresholds_fit_on_train"]
    frac = float(pred.mean())
    cls = classify_records(np.array([frac]), t1, t2)[0]
    split = load_json(project_path("results", "metrics", "training_info.json"))["split_records"]
    where = next((k for k, v in split.items() if name in v), "external")
    print(f"Record {name} ({where} split): {len(pred)} minutes analysed")
    print(f"Predicted apnea minutes: {pred.sum()} ({frac:.1%})  ->  recording class {CLASS_TEXT[cls]}")
    if len(y):
        print(f"Annotated apnea minutes: {int(y.sum())} ({y.mean():.1%});  per-minute accuracy {np.mean(pred == y):.3f}")

    fig, ax = plt.subplots(figsize=(12, 3.4))
    hours = minutes / 60
    ax.fill_between(hours, 0, y, step="mid", color=SERIES[0], alpha=0.25, label="Annotated apnea minutes")
    ax.plot(hours, prob, color=SERIES[1], lw=1.2, label="Predicted apnea probability")
    ax.axhline(0.5, color=CRITICAL, lw=1, ls="--", label="Decision threshold")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("P(apnea)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Record {name}: predicted {frac:.0%} apnea minutes → class {CLASS_TEXT[cls]}")
    ax.legend(loc="upper right", ncol=3)
    out = save(fig, project_path("results", "graphs", f"prediction_demo_{name}.png"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
