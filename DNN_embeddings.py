#!/usr/bin/env python
# coding: utf-8
"""Export test-set DNN representations for one trained DNN instance.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf


def save_test_embeddings(
    model,
    X_test,
    meta_test,
    output_path,
    batch_size=4096,
    *,
    split_id=None,
    instance_index=None,
    validation_fold=None,
    scaler_mean=None,
    scaler_scale=None,
):
    """Save one instance's test inputs, labels, patient IDs and DNN features."""
    required = {"Label", "Sid"}
    missing = required.difference(meta_test.columns)
    if missing:
        raise ValueError(f"Test metadata is missing columns: {sorted(missing)}")

    X_test = np.asarray(X_test, dtype=np.float32)
    if X_test.ndim != 2 or len(X_test) != len(meta_test):
        raise ValueError("X_test must be 2D and match the number of metadata rows.")
    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains NaN or infinity.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    labels = np.asarray(meta_test["Label"])
    if not np.isfinite(labels.astype(float)).all() or not np.isin(labels.astype(float), [0.0, 1.0]).all():
        raise ValueError("Test labels must contain only finite 0/1 values.")

    # In this architecture, the last layer is the logit output and the
    # penultimate layer is the final hidden block's Dropout layer. At inference
    # Dropout is inactive, so this yields the final hidden representation.
    if len(model.layers) < 2:
        raise ValueError("Model does not contain a penultimate feature layer.")
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-2].output,
        name="dnn_penultimate_feature_extractor",
    )
    X_dnn = feature_extractor.predict(X_test, batch_size=batch_size, verbose=0)

    if scaler_mean is None:
        scaler_mean = np.asarray([], dtype=np.float32)
    else:
        scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
    if scaler_scale is None:
        scaler_scale = np.asarray([], dtype=np.float32)
    else:
        scaler_scale = np.asarray(scaler_scale, dtype=np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        y=labels.astype(np.int32),
        Sid=meta_test["Sid"].astype(str).to_numpy(),
        X_standardized=X_test,
        X_raw=X_test,  # backward-compatible alias; not raw/unscaled input
        X_dnn=np.asarray(X_dnn, dtype=np.float32),
        split_id=np.asarray(-1 if split_id is None else split_id, dtype=np.int32),
        instance_index=np.asarray(-1 if instance_index is None else instance_index, dtype=np.int32),
        validation_fold=np.asarray(-1 if validation_fold is None else validation_fold, dtype=np.int32),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_layer=np.asarray(model.layers[-2].name),
        embedding_scope=np.asarray("single_instance_test_embedding"),
    )
    return str(output_path)
