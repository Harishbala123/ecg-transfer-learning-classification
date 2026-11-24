"""Two-stage AlexNet transfer learning.  Usage: python scripts/train.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import train  # noqa: E402
from src.utils.common import load_config  # noqa: E402

if __name__ == "__main__":
    train(load_config())
