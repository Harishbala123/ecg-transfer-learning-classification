"""Build the per-minute image cache and load splits."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.apnea import load_record
from src.data.download import RECORDS
from src.data.features import record_images
from src.utils.common import get_logger

log = get_logger(__name__)


def _process(args):
    raw_dir, out_dir, name, cfg_sig = args
    out = Path(out_dir) / f"{name}.npz"
    if out.exists():
        return name, None
    rec = load_record(raw_dir, name)
    X, y, m = record_images(rec, cfg_sig)
    np.savez_compressed(out, X=X, y=y, minute=m, total_minutes=len(rec.minute_labels))
    return name, (len(y), int(y.sum()), len(rec.minute_labels))


def build_cache(raw_dir: Path, out_dir: Path, cfg_sig: dict, workers: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(raw_dir), str(out_dir), r, cfg_sig) for r in RECORDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, stats in ex.map(_process, jobs):
            if stats:
                log.info("%s: %d minute images (%d apnea) from %d labelled minutes", name, *stats)


def load_split(proc_dir: Path, records: list[str]):
    Xs, ys, recs, mins = [], [], [], []
    for r in records:
        d = np.load(Path(proc_dir) / f"{r}.npz")
        Xs.append(d["X"])
        ys.append(d["y"])
        mins.append(d["minute"])
        recs += [r] * len(d["y"])
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int32), np.array(recs), np.concatenate(mins)


def cache_summary(proc_dir: Path) -> pd.DataFrame:
    rows = []
    for r in RECORDS:
        d = np.load(Path(proc_dir) / f"{r}.npz")
        rows.append({"record": r, "labelled_minutes": int(d["total_minutes"]), "images": len(d["y"]),
                     "apnea_images": int(d["y"].sum())})
    return pd.DataFrame(rows)
