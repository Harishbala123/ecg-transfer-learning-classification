"""ECG -> per-minute RR/EDR wavelet images cached in data/processed.  Usage: python scripts/build_dataset.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import build_cache, cache_summary  # noqa: E402
from src.utils.common import load_config, project_path  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    proc = project_path(cfg["data"]["processed_dir"])
    build_cache(project_path(cfg["data"]["raw_dir"]), proc, cfg["signal"])
    s = cache_summary(proc)
    s.to_csv(project_path("results", "metrics", "dataset_summary.csv"), index=False)
    print(s.sum(numeric_only=True))
