"""One-time export of ImageNet-pretrained AlexNet conv weights from torchvision to a Keras-layout .npz,
followed by a numerical check that the Keras conv stack reproduces torchvision's feature maps.

Requires: torch, torchvision (CPU build is enough).
Usage:    python scripts/export_alexnet_weights.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

from src.models.alexnet import CONV_SPECS, alexnet_features, torch_to_keras_conv  # noqa: E402
from src.utils.common import get_logger, project_path  # noqa: E402

log = get_logger("export_alexnet")
TORCH_CONV_IDX = {"conv1": 0, "conv2": 3, "conv3": 6, "conv4": 8, "conv5": 10}


def main():
    tv = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).eval()
    out = {}
    for name, *_ in CONV_SPECS:
        conv = tv.features[TORCH_CONV_IDX[name]]
        out[f"{name}_kernel"] = torch_to_keras_conv(conv.weight.detach().numpy())
        out[f"{name}_bias"] = conv.bias.detach().numpy()
    dest = project_path("models", "alexnet_imagenet_conv.npz")
    np.savez(dest, **out)
    log.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)

    # Verify the port: identical input -> identical conv5 feature maps
    inp = keras.Input((224, 224, 3))
    k = keras.Model(inp, alexnet_features(inp))
    for name, *_ in CONV_SPECS:
        k.get_layer(name).set_weights([out[f"{name}_kernel"], out[f"{name}_bias"]])
    x = np.random.default_rng(0).normal(size=(2, 224, 224, 3)).astype(np.float32)
    with torch.no_grad():
        ref = tv.features(torch.from_numpy(x.transpose(0, 3, 1, 2))).numpy().transpose(0, 2, 3, 1)
    got = k.predict(x, verbose=0)
    err = float(np.max(np.abs(ref - got)) / (np.max(np.abs(ref)) + 1e-9))
    log.info("Max relative difference Keras vs torchvision conv5 output: %.2e", err)
    if err > 1e-3:
        raise SystemExit("Weight port check FAILED")
    log.info("Weight port check passed")


if __name__ == "__main__":
    main()
