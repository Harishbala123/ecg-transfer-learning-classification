"""AlexNet in Keras, layer-for-layer identical to torchvision's `alexnet` (the single-GPU variant of
Krizhevsky et al., 2012, as re-implemented in "One weird trick", Krizhevsky 2014), so that its
ImageNet-1k weights can be transferred.

Transfer-learning model:
  stored image (64x128x3 log-power / phase)  ->  fixed per-channel scaling to [0, 1]
  -> resize to 224x224 -> ImageNet mean/std normalisation -> AlexNet conv1-conv5 (pretrained)
  -> flatten (6x6x256) -> Dropout -> Dense(256, ReLU) -> Dropout -> Dense(2, softmax)
The original 4096-unit FC layers are replaced by a small head: far fewer parameters for a ~25k-image dataset,
and the trained model stays small enough to keep in the Git repository.
"""
from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
from keras import layers

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
CONV_SPECS = [  # name, filters, kernel, stride, padding
    ("conv1", 64, 11, 4, 2), ("conv2", 192, 5, 1, 2), ("conv3", 384, 3, 1, 1),
    ("conv4", 256, 3, 1, 1), ("conv5", 256, 3, 1, 1),
]
POOL_AFTER = {"conv1", "conv2", "conv5"}


def alexnet_features(x):
    for name, f, k, s, p in CONV_SPECS:
        x = layers.ZeroPadding2D(p, name=f"{name}_pad")(x)
        x = layers.Conv2D(f, k, strides=s, activation="relu", name=name)(x)
        if name in POOL_AFTER:
            x = layers.MaxPooling2D(3, 2, name=f"{name}_pool")(x)
    return x  # 6 x 6 x 256 for a 224 x 224 input


def build_alexnet_classifier(image_shape, n_classes: int, scale_lo, scale_hi, input_size: int = 224,
                             dense_units: int = 256, dropout: float = 0.5, weights_path: str | Path | None = None
                             ) -> keras.Model:
    lo, hi = np.asarray(scale_lo, np.float32), np.asarray(scale_hi, np.float32)
    inp = keras.Input(shape=image_shape, name="ecg_image")
    x = layers.Rescaling(1.0 / (hi - lo), offset=-lo / (hi - lo), name="channel_scaling")(inp)
    x = layers.ReLU(max_value=1.0, name="clip_0_1")(x)
    x = layers.Resizing(input_size, input_size, interpolation="bilinear", name="resize")(x)
    x = layers.Rescaling(1.0 / IMAGENET_STD, offset=-IMAGENET_MEAN / IMAGENET_STD, name="imagenet_norm")(x)
    x = alexnet_features(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="fc_head")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inp, out, name="alexnet_ecg")
    if weights_path:
        load_pretrained_conv(model, weights_path)
    return model


def load_pretrained_conv(model: keras.Model, weights_path: str | Path) -> None:
    """Load conv1-conv5 weights exported by scripts/export_alexnet_weights.py (Keras layout)."""
    w = np.load(weights_path)
    for name, *_ in CONV_SPECS:
        model.get_layer(name).set_weights([w[f"{name}_kernel"], w[f"{name}_bias"]])


def set_backbone_trainable(model: keras.Model, trainable_convs: tuple[str, ...]) -> None:
    for name, *_ in CONV_SPECS:
        model.get_layer(name).trainable = name in trainable_convs


def torch_to_keras_conv(weight: np.ndarray) -> np.ndarray:
    """PyTorch conv kernel (out, in, kh, kw) -> Keras (kh, kw, in, out)."""
    return np.transpose(weight, (2, 3, 1, 0))
