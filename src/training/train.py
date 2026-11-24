"""Two-stage AlexNet transfer learning on per-minute ECG time-frequency images."""
from __future__ import annotations

import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data.apnea import CLASS_NAMES, read_metadata, subject_split
from src.data.dataset import load_split
from src.models.alexnet import build_alexnet_classifier, set_backbone_trainable
from src.utils.common import get_logger, project_path, save_json, set_seed
from src.utils.export import save_inference_model

log = get_logger(__name__)


def scaling_constants(X: np.ndarray) -> tuple[list[float], list[float]]:
    """Channels 0/1 (log power): training-set 1st/99th percentiles; channel 2 (cos phase) is already in [-1, 1]."""
    rng = np.random.default_rng(0)
    sample = X[rng.choice(len(X), min(len(X), 3000), replace=False)].astype(np.float32)
    lo = [float(np.percentile(sample[..., c], 1)) for c in (0, 1)] + [-1.0]
    hi = [float(np.percentile(sample[..., c], 99)) for c in (0, 1)] + [1.0]
    return lo, hi


def make_ds(X, y, batch, shuffle, augment, seed):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)

    def prep(x, t):
        x = tf.cast(x, tf.float32)
        if augment:  # random circular time shift of up to ±8 columns (~19 s): the event's position in the window varies
            x = tf.roll(x, shift=tf.random.uniform([], -8, 9, dtype=tf.int32), axis=1)
        return x, t

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def get_data(cfg):
    raw = project_path(cfg["data"]["raw_dir"])
    proc = project_path(cfg["data"]["processed_dir"])
    meta = read_metadata(raw)
    split = subject_split(meta, cfg["split"]["val_fraction"], cfg["split"]["test_fraction"], cfg["seed"])
    data = {k: load_split(proc, v) for k, v in split.items()}
    return meta, split, data


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    meta, split, data = get_data(cfg)
    (Xtr, ytr, _, _), (Xva, yva, _, _) = data["train"], data["val"]
    for k, (X, y, r, _) in data.items():
        log.info("%-5s %5d images, apnea %.1f %%, records %d, subjects %d", k, len(y), 100 * y.mean(),
                 len(set(r)), meta[meta.record.isin(set(r))].subject.nunique())
    lo, hi = scaling_constants(Xtr)
    tc, mc = cfg["training"], cfg["model"]
    model = build_alexnet_classifier(Xtr.shape[1:], len(CLASS_NAMES), lo, hi, mc["input_size"], mc["dense_units"],
                                     mc["dropout"], project_path(mc["pretrained_weights"]))
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    class_weight = {0: float(cw[0]), 1: float(cw[1])} if tc["class_weight"] == "balanced" else None
    ds_tr = make_ds(Xtr, ytr, tc["batch_size"], True, True, cfg["seed"])
    ds_va = make_ds(Xva, yva, tc["batch_size"], False, False, cfg["seed"])
    models_dir = project_path("models")
    ckpt = models_dir / "alexnet_apnea_best.keras"

    def run_stage(epochs, lr, stage):
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=tc["patience"], restore_best_weights=True),
               keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True)]
        h = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, class_weight=class_weight, callbacks=cbs, verbose=2)
        return pd.DataFrame(h.history).assign(stage=stage)

    t0 = time.time()
    set_backbone_trainable(model, ())
    log.info("Stage 1: frozen AlexNet backbone, training head (%d trainable params)",
             sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    h1 = run_stage(tc["head_epochs"], tc["head_lr"], "1_head")
    stage1_weights, stage1_best = model.get_weights(), float(h1["val_loss"].min())
    set_backbone_trainable(model, ("conv4", "conv5"))
    log.info("Stage 2: fine-tuning conv4-conv5 (%d trainable params)",
             sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    h2 = run_stage(tc["finetune_epochs"], tc["finetune_lr"], "2_finetune")
    # Early stopping restores the best epoch *within* a stage; keep fine-tuning only if it beat stage 1 on validation.
    stage2_kept = float(h2["val_loss"].min()) < stage1_best
    if not stage2_kept:
        model.set_weights(stage1_weights)
        log.info("Fine-tuning did not improve validation loss (%.4f vs %.4f): keeping stage-1 weights",
                 h2["val_loss"].min(), stage1_best)
    hist = pd.concat([h1, h2], ignore_index=True)
    hist.index.name = "epoch"
    save_inference_model(model, models_dir / "alexnet_apnea.keras")   # weights only, no optimizer state
    if ckpt.exists():
        ckpt.unlink()      # the final model already holds the selected best weights
    res = project_path("results")
    hist.to_csv(res / "metrics" / "training_history.csv")
    info = {"train_seconds": round(time.time() - t0, 1), "split_records": split, "scale_lo": lo, "scale_hi": hi,
            "class_weight": class_weight, "epochs": {"stage1": len(h1), "stage2": len(h2)},
            "stage1_best_val_loss": stage1_best, "stage2_best_val_loss": float(h2["val_loss"].min()),
            "stage2_kept": stage2_kept,
            "n_images": {k: int(len(v[1])) for k, v in data.items()}}
    save_json(info, res / "metrics" / "training_info.json")
    log.info("Done in %.0fs", info["train_seconds"])
    return info
