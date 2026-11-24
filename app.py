"""Interactive demo: whole-night sleep-apnea detection from a single-lead ECG.

Run:  streamlit run app.py        (or double-click run_demo.bat)
Research demo only. Not a medical device.
"""
from __future__ import annotations

import json

import keras
import numpy as np
import pandas as pd
import streamlit as st

from src.data.apnea import CLASS_NAMES, load_record, record_class
from src.data.features import record_images
from src.evaluation.evaluate import classify_records
from src.utils.common import load_config, load_json, project_path
from src.utils.plotting import CRITICAL, SERIES, plt

st.set_page_config(page_title="ECG Sleep-Apnea Demo", layout="wide")
cfg = load_config()
CLASS_TEXT = {"A": "A: apnea", "B": "B: borderline", "C": "C: control (no apnea)"}


@st.cache_resource
def load_model():
    return keras.models.load_model(project_path("models", "alexnet_apnea.keras"))


@st.cache_data(show_spinner="Processing the overnight ECG (R-peaks, RR/EDR series, wavelet images)...")
def analyse(raw_dir: str, name: str):
    rec = load_record(raw_dir, name)
    X, y, minutes = record_images(rec, cfg["signal"])
    prob = load_model().predict(X.astype(np.float32), batch_size=256, verbose=0)[:, 1]
    ecg_preview = rec.ecg[: 30 * rec.fs]
    return X, y, minutes, prob, ecg_preview, rec.fs


def available_records() -> dict[str, str]:
    """Records found locally: the full database in data/raw (if downloaded) plus the bundled sample."""
    out = {p.stem: str(p.parent) for p in sorted(project_path("data", "raw").glob("*.dat"))}
    for p in project_path("samples").glob("*.dat"):
        out.setdefault(p.stem, str(p.parent))
    return out


st.title("Sleep-Apnea Detection from a Single-Lead ECG (AlexNet Transfer Learning)")
st.caption("PhysioNet Apnea-ECG. Each minute is scored from heart-rate (RR) and ECG-derived-respiration wavelet "
           "images by an ImageNet-pretrained AlexNet. Research demo: not a medical device.")

info = load_json(project_path("results", "metrics", "training_info.json"))
split_of = {r: k for k, v in info["split_records"].items() for r in v}
t1, t2 = load_json(project_path("results", "metrics", "evaluation_metrics.json"))["test_records"]["thresholds_fit_on_train"]
meta_path = project_path("data", "raw", "additional-information.txt")

tab_demo, tab_results, tab_how = st.tabs(["Analyse a night", "Experiment results", "How it works"])

with tab_demo:
    recs = available_records()
    test_first = sorted(recs, key=lambda r: (split_of.get(r) != "test", r))
    name = st.sidebar.selectbox("Recording", test_first,
                                format_func=lambda r: f"{r}  ({split_of.get(r, 'external')} split)")
    if len(recs) <= 1:
        st.sidebar.caption("Only the bundled sample is available. Run `python scripts/download_data.py` "
                           "for all 70 recordings.")
    X, y, minutes, prob, ecg, fs = analyse(recs[name], name)
    pred = (prob >= 0.5).astype(int)
    frac = float(pred.mean())
    cls = classify_records(np.array([frac]), t1, t2)[0]
    true_cls = record_class(int(y.sum()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Minutes analysed", f"{len(pred)}", f"{len(pred) / 60:.1f} h", delta_color="off")
    c2.metric("Predicted apnea minutes", f"{pred.sum()} ({frac:.0%})", f"annotated {int(y.sum())} ({y.mean():.0%})",
              delta_color="off")
    c3.metric("Per-minute accuracy", f"{np.mean(pred == y):.1%}")
    c4.metric("Recording diagnosis", CLASS_TEXT[cls], f"annotated {CLASS_TEXT[true_cls]}", delta_color="off")
    if split_of.get(name) != "test":
        st.warning(f"{name} belongs to the **{split_of.get(name, 'unknown')}** split, so the model saw this person "
                   "during training or validation. Pick a test-split recording for an honest demo.")

    fig, ax = plt.subplots(figsize=(12, 3.4))
    hours = minutes / 60
    ax.fill_between(hours, 0, y, step="mid", color=SERIES[0], alpha=0.25, label="Annotated apnea minutes")
    ax.plot(hours, prob, color=SERIES[1], lw=1.1, label="Predicted P(apnea)")
    ax.axhline(0.5, color=CRITICAL, lw=1, ls="--", label="Threshold")
    ax.set_xlabel("Time since recording start (hours)")
    ax.set_ylabel("P(apnea)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", ncol=3)
    ax.set_title(f"Recording {name}: whole-night prediction")
    st.pyplot(fig)

    st.subheader("Inspect one minute")
    i = st.slider("Minute of the night", 0, len(minutes) - 1, int(np.argmax(prob)))
    m = int(minutes[i])
    cc = st.columns(3)
    for col, ch, title in zip(cc, range(3), ["RR-interval scalogram (heart rate)", "EDR scalogram (breathing)",
                                             "cos(phase difference) (coupling)"]):
        f, a = plt.subplots(figsize=(4, 2.6))
        a.imshow(X[i][..., ch].astype(np.float32), aspect="auto", cmap="viridis",
                 extent=[-2.5, 2.5, 0, 1])
        a.set_yticks([0, 1], ["0.01 Hz", "0.5 Hz"])
        a.set_xlabel("minutes from labelled minute")
        a.set_title(title, fontsize=9)
        a.grid(False)
        col.pyplot(f)
    st.write(f"Minute {m}: annotated **{CLASS_NAMES[y[i]]}**, predicted **{CLASS_NAMES[pred[i]]}** "
             f"(P(apnea) = {prob[i]:.2f}). In apnea, the heart rate oscillates every 30–90 s, which shows as a "
             "bright band near the bottom (0.01–0.03 Hz) of the RR scalogram.")
    with st.expander("Raw ECG, first 30 seconds"):
        f, a = plt.subplots(figsize=(12, 2.4))
        a.plot(np.arange(len(ecg)) / fs, ecg, color=SERIES[0], lw=0.7)
        a.set_xlabel("seconds")
        a.set_ylabel("mV")
        st.pyplot(f)

with tab_results:
    res = project_path("results")
    mt = json.loads((res / "metrics" / "evaluation_metrics.json").read_text())
    rows = [{"split": k, **{m: mt[k][m] for m in ("accuracy", "macro_f1", "recall_apnea_sensitivity",
                                                 "recall_normal_specificity", "roc_auc", "n")}}
            for k in ("train", "val", "test")]
    st.dataframe(pd.DataFrame(rows).round(4))
    st.write(f"Recording-level diagnosis on {mt['test_records']['n_records']} test recordings: "
             f"**{mt['test_records']['accuracy']:.1%}**; apnea-burden correlation "
             f"r = {mt['test_records']['apnea_fraction_correlation']:.3f}.")
    c1, c2 = st.columns(2)
    c1.image(str(res / "graphs" / "confusion_matrix_minutes.png"))
    c2.image(str(res / "graphs" / "roc_curve.png"))
    c1.image(str(res / "graphs" / "confusion_matrix_records.png"))
    c2.image(str(res / "graphs" / "record_apnea_burden.png"))
    st.image(str(res / "graphs" / "training_curves.png"))
    st.dataframe(pd.read_csv(res / "metrics" / "per_record_results.csv").round(3))

with tab_how:
    st.markdown("""
1. **ECG → beats:** R-peaks from the database's QRS annotations, refined to the true peak (±50 ms).
2. **Beats → two physiological signals:** RR intervals (heart-rate variability) and R-wave amplitude
   (ECG-derived respiration). Ectopic or missed beats are removed.
3. **Signals → image:** a 5-minute window around each labelled minute, resampled at 4 Hz, then a complex
   Morlet wavelet transform (0.01–0.5 Hz). The RGB channels are RR power, EDR power and their phase coupling.
4. **Image → AlexNet:** conv layers pretrained on ImageNet (ported from PyTorch and verified). Stage 1 trains a
   new head; stage 2 fine-tunes conv4–conv5.
5. **Minutes → diagnosis:** the fraction of apnea minutes maps to the PhysioNet classes A / B / C, with cut-points fitted on training recordings.

**No leakage:** recordings of the same person (identical age/sex/height/weight) are kept in one split.
The test set has 7 people the model never saw.

**Scope:** sleep apnea is a breathing disorder, not a lung-tissue disease. No public ECG dataset has
genuine lung-disease labels.
""")
