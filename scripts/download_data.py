"""Download + verify PhysioNet Apnea-ECG.  Usage: python scripts/download_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download import download  # noqa: E402
from src.utils.common import load_config, project_path  # noqa: E402

if __name__ == "__main__":
    download(project_path(load_config()["data"]["raw_dir"]))
