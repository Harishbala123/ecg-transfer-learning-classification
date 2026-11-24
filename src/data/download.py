"""Download the PhysioNet Apnea-ECG Database (v1.0.0) with SHA-256 verification.

Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. The Apnea-ECG Database. Computers in Cardiology 2000;27:255-258.
Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 2000;101(23):e215-e220.
License: Open Data Commons Attribution License v1.0 (ODC-By).

Only the single-lead ECG (.dat/.hea), the per-minute apnea annotations (.apn) and the database's
machine-generated QRS annotations (.qrs) are downloaded (~420 MB);
the respiration-signal records (*r) are not needed. Uses only the Python standard library.

Usage:  python -m src.data.download  [--out data/raw]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/apnea-ecg/1.0.0/"
RECORDS = ([f"a{i:02d}" for i in range(1, 21)] + [f"b{i:02d}" for i in range(1, 6)]
           + [f"c{i:02d}" for i in range(1, 11)] + [f"x{i:02d}" for i in range(1, 36)])
EXTRA = ["additional-information.txt", "test-dataset-annos.txt", "SHA256SUMS.txt"]


def _get(url: str, dest: Path, retries: int = 5) -> None:
    for attempt in range(1, retries + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 16):
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # network hiccups: retry with back-off
            print(f"  retry {attempt}/{retries} for {dest.name}: {e}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"failed to download {url}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in EXTRA:
        if not (out / f).exists():
            _get(BASE + f, out / f)
    sums = {}
    for line in (out / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    files = [f"{r}.{ext}" for r in RECORDS for ext in ("hea", "apn", "qrs", "dat")]
    for i, f in enumerate(files, 1):
        dest = out / f
        if dest.exists() and (f not in sums or _sha256(dest) == sums[f]):
            continue
        print(f"[{i}/{len(files)}] {f}", flush=True)
        _get(BASE + f, dest)
        if f in sums and _sha256(dest) != sums[f]:
            raise RuntimeError(f"checksum mismatch for {f}")
    print(f"Apnea-ECG ready in {out} ({len(files)} files verified)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "raw"))
    download(Path(ap.parse_args().out))
    sys.exit(0)
