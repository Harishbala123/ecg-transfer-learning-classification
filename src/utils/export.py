"""Save an inference-only copy of a trained model (same weights, no optimizer state, ~half the file size)."""
from __future__ import annotations

from pathlib import Path

import keras


def save_inference_model(model: keras.Model, path: str | Path) -> Path:
    # Rebuild from the architecture config: the new model is never compiled, so no optimizer variables are saved
    # (keras.models.clone_model keeps the optimizer in Keras 3.15).
    clone = model.__class__.from_config(model.get_config())
    clone.set_weights(model.get_weights())
    path = Path(path)
    clone.save(path)          # uncompiled clone -> no optimizer variables in the file
    return path


if __name__ == "__main__":   # python -m src.utils.export in.keras out.keras
    import sys
    m = keras.models.load_model(sys.argv[1])
    print(save_inference_model(m, sys.argv[2]))
