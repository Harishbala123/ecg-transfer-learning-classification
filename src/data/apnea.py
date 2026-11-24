"""Apnea-ECG records, labels and subject grouping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from src.data.download import RECORDS

LABELS = {"N": 0, "A": 1}
CLASS_NAMES = ["Normal", "Apnea"]


@dataclass
class Record:
    name: str
    ecg: np.ndarray          # single-lead ECG in mV
    fs: int
    qrs: np.ndarray          # sample indices of QRS complexes (database annotations)
    minute_labels: np.ndarray  # 0 = normal minute, 1 = apnea/hypopnea minute


def load_record(raw_dir: str | Path, name: str) -> Record:
    path = str(Path(raw_dir) / name)
    sig = wfdb.rdrecord(path, channels=[0])
    ecg = sig.p_signal[:, 0].astype(np.float32)
    qrs = wfdb.rdann(path, "qrs").sample
    ann = wfdb.rdann(path, "apn")
    labels = np.array([LABELS[s] for s in ann.symbol], dtype=np.int8)
    return Record(name, ecg, int(sig.fs), np.asarray(qrs), labels)


def read_metadata(raw_dir: str | Path) -> pd.DataFrame:
    """Parse additional-information.txt (per-record apnea minutes, AHI, age, sex, height, weight)."""
    rows = []
    for line in (Path(raw_dir) / "additional-information.txt").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in RECORDS and len(parts) >= 12:
            rows.append(parts[:12])
    cols = ["record", "minutes", "normal_minutes", "apnea_minutes", "hours_with_apnea", "AI", "HI", "AHI",
            "age", "sex", "height", "weight"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c not in ("record", "sex"):
            df[c] = pd.to_numeric(df[c])
    df["record_class"] = df["apnea_minutes"].map(record_class)
    # The database does not publish subject IDs, but several recordings share identical demographics
    # (e.g. a01/a14/x?), i.e. the same person recorded on different nights. Grouping on all four attributes
    # is conservative: two different people with identical attributes would only make the split stricter.
    key = df["age"].astype(str) + "_" + df["sex"] + "_" + df["height"].astype(str) + "_" + df["weight"].astype(str)
    df["subject"] = pd.factorize(key)[0]
    return df


def record_class(apnea_minutes: float) -> str:
    """Record-level diagnosis following the PhysioNet/CinC 2000 definitions, using minute counts:
    A (apnea) >= 100 apnea minutes, B (borderline) 5-99, C (control) < 5."""
    if apnea_minutes >= 100:
        return "A"
    if apnea_minutes >= 5:
        return "B"
    return "C"


def subject_split(meta: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> dict[str, list[str]]:
    """Assign whole subjects to train/val/test, stratified by the subject's most severe record class so every
    split contains apnea, borderline and control subjects."""
    rng = np.random.default_rng(seed)
    severity = {"A": 0, "B": 1, "C": 2}
    subj = meta.groupby("subject")["record_class"].agg(lambda s: min(s, key=severity.get))
    split = {"train": [], "val": [], "test": []}
    for cls in ("A", "B", "C"):
        ids = subj[subj == cls].index.to_numpy().copy()   # pandas 3 returns read-only arrays
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * test_fraction)))
        n_val = max(1, int(round(len(ids) * val_fraction)))
        groups = {"test": ids[:n_test], "val": ids[n_test:n_test + n_val], "train": ids[n_test + n_val:]}
        for k, g in groups.items():
            split[k] += meta[meta.subject.isin(g)].record.tolist()
    return {k: sorted(v) for k, v in split.items()}
