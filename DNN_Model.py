#!/usr/bin/env python
# coding: utf-8
"""Patient-level cross-validated DWT/PMF DNN for fragmentomics classification.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import ast
import time
import pickle
import logging
import warnings

import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from tensorflow.keras import layers
from DNN_embeddings import save_test_embeddings
# ----------------------------------------------------------------------------
# global constants
# ----------------------------------------------------------------------------
dwt_input = 1                     # 1 -- DWT coefficients; 0 -- FS PMF profiles
batch_size = 32
epochs = 200
min_delta = 1e-4
Nsplits = 20
Ntr_inst = 20

refDir = './data/TrainTestPaths/'
resBase = './data/Test_Results/DWT_DNN/' if dwt_input else './data/Test_Results/PMF_DNN/'
fDatName = 'FSpmf_TrainTestPaths_'
fResName = 'TestPerf_df'

GROUPS = ['Sub', 'Maj', 'Con', 'Avg']
METRICS = ['Acc', 'Spc', 'Sns', 'Auc', 'AucPR']
METRIC_COLS = [m + g for g in GROUPS for m in METRICS]
TRAIN_COLS = ['ValLoss', 'TrLoss', 'TrAcc', 'TrEpochs', 'BestEpoch', 'TrTime']

# ----------------------------------------------------------------------------
# run configuration
# ----------------------------------------------------------------------------
DEFAULT_CFG = dict(
    standardize=True,
    class_weights=True,
    batchnorm=True,
    hidden=(64, 32, 16),
    drop_rate=0.4,
    lr=1e-3,
    cosine=True,
    weight_decay=1e-4,
    cv_folds=5,              # patient-level folds; instance i uses fold i % cv_folds
    min_epochs=10,           # earliest epoch eligible for checkpoint selection
    patience=15,
    use_val_threshold=False, # tune threshold using patient-level OOF probabilities
    save_embeddings=True,    # save final-instance test features for t-SNE visualization
)

PRESETS = {
    'main': {},
    # Original hyperparameter choices, evaluated under the current patient-level
    # validation/early-stopping protocol (legacy train-loss stopping was removed).
    'baseline_orig': dict(standardize=False, class_weights=False, batchnorm=False,
                          hidden=(64, 32, 16), cosine=False, weight_decay=0.0),
    'no_classw': dict(class_weights=False),
    'no_bn': dict(batchnorm=False),
    'small_arch': dict(hidden=(64, 32, 16)),
    'no_norm': dict(standardize=False),
    'no_cosine': dict(cosine=False),
}


# Parse command-line arguments and validate the run configuration.
def parse_args(argv):
    if len(argv) < 2:
        print(__doc__ or "Usage: python DNN_Model.py <preset> [key=value ...]")
        sys.exit(1)
    run = argv[1]
    cfg = dict(DEFAULT_CFG)
    cfg.update(PRESETS.get(run, {}))
    for kv in argv[2:]:
        if '=' not in kv:
            raise ValueError(f"Override must be key=value, got: {kv}")
        k, v = kv.split('=', 1)
        if k not in cfg:
            raise ValueError(f"Unknown config key '{k}'. Valid keys: {sorted(cfg)}")
        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v
    if cfg['cv_folds'] < 2:
        raise ValueError("cv_folds must be >= 2.")

    if cfg['min_epochs'] < 1 or cfg['min_epochs'] > epochs:
        raise ValueError(f"min_epochs must be between 1 and {epochs}.")

    if cfg['patience'] < 1:
        raise ValueError("patience must be >= 1.")

    if cfg['drop_rate'] < 0 or cfg['drop_rate'] >= 1:
        raise ValueError("drop_rate must be in [0, 1).")

    if cfg['lr'] <= 0:
        raise ValueError("lr must be > 0.")

    if not cfg['hidden'] or any(int(u) <= 0 for u in cfg['hidden']):
        raise ValueError("hidden must contain positive layer sizes.")

    cfg['hidden'] = tuple(cfg['hidden'])
    return run, cfg


# ----------------------------------------------------------------------------
# data functions
# ----------------------------------------------------------------------------
# Load and validate fragment-length profiles from serialized files.
def load_data(fPaths):
    FSpdfs = []
    for fPath in fPaths:
        with open(fPath, 'rb') as handle:
            FSpdf = np.asarray(pickle.load(handle))

        # Keep only fragment lengths 51-250 bp (200 bins).
        FSpdf = np.squeeze(FSpdf)
        if FSpdf.ndim != 1:
            raise ValueError(f"Expected a 1D fragment-length profile in {fPath}, "
                             f"but received shape {FSpdf.shape}.")
        if FSpdf.shape[0] == 200:
            pass
        elif FSpdf.shape[0] >= 250:
            FSpdf = FSpdf[50:250]
        else:
            raise ValueError(f"Unsupported profile length {FSpdf.shape[0]} in {fPath}. "
                             "Expected 200 bins or at least 250 bins.")
        if FSpdf.shape[0] != 200:
            raise ValueError(f"After selecting 51-250 bp, {fPath} has {FSpdf.shape[0]} bins instead of 200.")
        FSpdfs.append(FSpdf)
    return np.stack(FSpdfs, axis=0)


# Transform input signals into concatenated discrete wavelet coefficients.
def dec_dwt1D(X_signal, wavelet='db5', n_max=3):
    X = []
    for i in range(X_signal.shape[0]):
        coeff_list = pywt.wavedec(X_signal[i, :], wavelet=wavelet, level=n_max)
        X.append(np.concatenate(coeff_list))
    return np.stack(X, axis=0)


# Extract sample identifiers from a data path.
def parse_path(fPath):
    """Same path parsing logic as the original make_test_dfs."""
    path = fPath.replace("\\", "/").split("_")
    FCid = path[1].split("/")[0]
    Sid = path[1].split("/")[-1]
    if len(path) == 3:
        ExSid = ""
        Subid = path[2].split(".")[0]
    elif len(path) == 4:
        ExSid = "_" + path[2]
        Subid = path[3].split(".")[0]
    else:
        ExSid = "_" + path[2] + "_" + path[3]
        Subid = path[4].split(".")[0]
    return FCid, Sid, Subid, ExSid


# Build a metadata table from paths and binary labels.
def make_meta_df(fpaths, labels):
    if len(fpaths) != len(labels):
        raise ValueError(f"Number of paths ({len(fpaths)}) does not match number of labels ({len(labels)}).")
    rows = []
    for fPath, label in zip(fpaths, labels):
        if int(label) not in (0, 1):
            raise ValueError(f"Unexpected label {label} in {fPath}")
        FCid, Sid, Subid, ExSid = parse_path(fPath)
        rows.append({'FCid': FCid, 'Sid': Sid, 'Subid': Subid, 'ExSid': ExSid, 'Label': int(label)})
    return pd.DataFrame(rows, columns=['FCid', 'Sid', 'Subid', 'ExSid', 'Label'])


# Validate required patient metadata and label consistency.
def validate_patient_metadata(meta, split_name):
    required = {'Sid', 'Label'}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"{split_name}: missing metadata columns: {sorted(missing)}"
        )

    if meta.empty:
        raise ValueError(f"{split_name}: metadata is empty.")

    if meta['Sid'].isna().any():
        raise ValueError(f"{split_name}: missing patient IDs.")

    sid_text = meta['Sid'].astype(str).str.strip()
    if sid_text.eq('').any():
        raise ValueError(f"{split_name}: empty patient IDs.")

    labels = pd.to_numeric(meta['Label'], errors='coerce')
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{split_name}: labels must be 0 or 1.")

    label_counts = meta.assign(Label=labels).groupby('Sid')['Label'].nunique()
    bad = label_counts[label_counts > 1]

    if not bad.empty:
        raise ValueError(
            f"{split_name}: inconsistent labels for patient IDs: "
            f"{bad.index.astype(str).tolist()[:10]}"
        )

# Assign patients to deterministic, label-stratified folds.
def patient_fold_ids(meta, K, seed):
    """Patient-level, label-stratified fold id (0..K-1) for every subsample.
    Depends only on (meta, K, seed): identical for all runs and for any future model line."""
    rng = np.random.RandomState(seed)
    pat_label = meta.groupby('Sid')['Label'].max()
    fold_of = {}
    for lab in (0, 1):
        sids = np.array(sorted(pat_label.index[pat_label.values == lab]))
        rng.shuffle(sids)
        for i, s in enumerate(sids):
            fold_of[s] = i % K
    return meta['Sid'].map(fold_of).values.astype(int)


# Audit patient fold assignments and class coverage.
def validate_patient_folds(meta, fold_ids, K, split_name):
    """Fail-fast audit: every patient belongs to exactly one fold and each
    fold contains patients from both classes.
    """
    fold_ids = np.asarray(fold_ids, dtype=int)
    if len(fold_ids) != len(meta):
        raise ValueError(
            f"{split_name}: fold_ids length ({len(fold_ids)}) does not match metadata ({len(meta)})."
        )
    if np.any((fold_ids < 0) | (fold_ids >= K)):
        raise ValueError(f"{split_name}: fold IDs must be in the range 0..{K - 1}.")

    audit = meta[['Sid', 'Label']].copy()
    audit['Fold'] = fold_ids

    patient_fold_counts = audit.groupby('Sid')['Fold'].nunique()
    if (patient_fold_counts != 1).any():
        bad = patient_fold_counts[patient_fold_counts != 1].index.astype(str).tolist()
        raise ValueError(
            f"{split_name}: patient(s) assigned to more than one fold: {bad[:10]}"
        )

    patient_table = audit.groupby('Sid', as_index=False).agg(
        Label=('Label', 'first'),
        Fold=('Fold', 'first')
    )
    fold_class_counts = pd.crosstab(
        patient_table['Fold'], patient_table['Label']
    ).reindex(index=range(K), columns=[0, 1], fill_value=0)

    if (fold_class_counts == 0).any().any():
        raise ValueError(
            f"{split_name}: at least one fold has no patients from one class. "
            f"Fold/class counts: {fold_class_counts.to_dict(orient='index')}"
        )

    return fold_class_counts


# Compute balanced sample weights from class frequencies.
def balanced_weights(y, reference_y=None):
    """Return balanced sample weights.

    If reference_y is supplied, class frequencies are estimated from that
    reference set (the training fold) and applied to y (e.g. validation data).
    """
    y = np.asarray(y).astype(int)
    reference_y = y if reference_y is None else np.asarray(reference_y).astype(int)
    n = len(reference_y)
    n1 = float(np.sum(reference_y == 1))
    n0 = float(np.sum(reference_y == 0))
    w1 = n / (2.0 * max(n1, 1.0))
    w0 = n / (2.0 * max(n0, 1.0))
    return np.where(y == 1, w1, w0).astype(np.float32)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
# Construct the configured neural network and optimizer.
def build_model(in_dim, cfg, steps_total):
    inputs = layers.Input(shape=(in_dim,))
    x = inputs
    for u in cfg['hidden']:
        x = layers.Dense(u, use_bias=not cfg['batchnorm'])(x)
        if cfg['batchnorm']:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(tf.nn.leaky_relu)(x)
        x = layers.Dropout(cfg['drop_rate'])(x)
    outputs = layers.Dense(1)(x)  # logits
    model = tf.keras.Model(inputs, outputs)

    if cfg['cosine']:
        lr = tf.keras.optimizers.schedules.CosineDecay(cfg['lr'], decay_steps=max(steps_total, 1), alpha=0.01)
    else:
        lr = cfg['lr']
    if cfg['weight_decay'] > 0:
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg['weight_decay'])
        except AttributeError:
            optimizer = tf.keras.optimizers.Adam(lr)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    return model, optimizer


# Create a compiled training step for the supplied model.
def make_train_step(model, optimizer):
    @tf.function
    # Run one weighted binary-classification optimization step.
    def train_step(x, y, w):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)[:, 0]
            per_ex = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_sum(w * per_ex) / tf.reduce_sum(w)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, logits
    return train_step


# Compute model logits for input data in batches.
def predict_logits(model, X, bs=4096):
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs], training=False).numpy()[:, 0])
    return np.concatenate(out)


# Calculate weighted binary cross-entropy from logits.
def np_weighted_bce(z, y, w):
    per_ex = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(np.sum(w * per_ex) / np.sum(w))


# Convert logits to probabilities with the sigmoid function.
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
# Calculate ROC AUC for the supplied labels and scores.
def roc_auc(y, s):
    """Exact ROC AUC (used for the OOF diagnostic)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return np.nan
    r = pd.Series(s).rank(method='average').values
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


# Calculate the configured Keras AUC metric.
def _keras_auc(y, score, curve='ROC'):
    """
    TensorFlow/Keras AUC using the same metric family as the official
    DeepFRAG implementation (tf.keras.metrics.AUC; default 200 thresholds).

    The official implementation uses this metric for both ROC and PR AUC,
    including patient-level binary consensus outputs.
    """
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    score = np.asarray(score, dtype=np.float32).reshape(-1)

    if len(y) != len(score):
        raise ValueError("Labels and scores must have equal length.")
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    if not np.isfinite(score).all():
        raise ValueError("AUC scores contain NaN or infinity.")

    metric = tf.keras.metrics.AUC(
        curve=curve,
        num_thresholds=200,
        summation_method='interpolation'
    )
    metric.update_state(y, score)
    return float(metric.result().numpy())


# Compute classification metrics at the selected threshold.
def basic_metrics(y, pred, score):
    """Metrics using the official DeepFRAG TensorFlow AUC implementation."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    score = np.asarray(score, dtype=float)

    if len(y) != len(pred) or len(y) != len(score):
        raise ValueError("y, pred, and score must have equal lengths.")
    if len(y) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    P = int((y == 1).sum())
    N = int((y == 0).sum())
    acc = (pred == y).mean() * 100
    spc = (
        ((pred == 0) & (y == 0)).sum() / N * 100
        if N else np.nan
    )
    sns = (
        ((pred == 1) & (y == 1)).sum() / P * 100
        if P else np.nan
    )

    return [
        np.round(acc, 3),
        np.round(spc, 3) if np.isfinite(spc) else np.nan,
        np.round(sns, 3) if np.isfinite(sns) else np.nan,
        np.round(_keras_auc(y, score, curve='ROC'), 5),
        np.round(_keras_auc(y, score, curve='PR'), 5),
    ]


# Compute the configured metrics for all evaluation groups.
def all_metrics(meta, p, thr):
    """
    Subsample-level and patient-level metrics.

    Sub:
        Per-subsample classification.
    Maj:
        Majority-of-vote, preserving the DeepFRAG 60% support threshold.
        The fraction of positive subsamples is the ranking score.
    Con:
        Strict consensus exactly as in the DeepFRAG reference code:
        any positive subsample among a healthy patient's subsamples is an FP;
        any negative subsample among a cancer patient's subsamples is an FN.
        The resulting binary consensus vector is used for ROC/PR AUC, as in
        the reference implementation.
    Avg:
        Mean patient probability for hard metrics, ROC AUC and PR AUC.
    """
    y = meta['Label'].to_numpy(dtype=int)
    p = np.asarray(p, dtype=float).reshape(-1)

    if len(y) != len(p):
        raise ValueError(
            f"Length mismatch: {len(y)} labels vs {len(p)} probabilities."
        )
    if len(y) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.isfinite(p).all():
        raise ValueError("Predicted probabilities contain NaN or infinity.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be in [0, 1].")
    if not np.isfinite(thr) or not (0 <= thr <= 1):
        raise ValueError("Threshold must be finite and in [0, 1].")
    if 'Sid' not in meta.columns or 'Label' not in meta.columns:
        raise ValueError("meta must contain 'Sid' and 'Label' columns.")

    pred = (p > thr).astype(int)

    d = pd.DataFrame({
        'Sid': meta['Sid'].to_numpy(),
        'Label': y,
        'p': p,
        'pos': pred
    })

    # Every patient must have one consistent ground-truth label.
    label_counts = d.groupby('Sid')['Label'].nunique()
    bad_patients = label_counts[label_counts > 1].index.tolist()
    if bad_patients:
        raise ValueError(
            "Inconsistent labels within patient(s): "
            + ", ".join(map(str, bad_patients[:10]))
        )

    g = d.groupby('Sid', sort=False).agg(
        Label=('Label', 'first'),
        frac_pos=('pos', 'mean'),
        mean_p=('p', 'mean'),
        any_pos=('pos', 'max'),
        all_pos=('pos', 'min')
    )

    lab = g['Label'].to_numpy(dtype=int)

    # 1) Subsample-level performance.
    res = basic_metrics(y, pred, p)

    # 2) Majority-of-vote: keep the paper's >= 60% support rule.
    maj_score = g['frac_pos'].to_numpy(dtype=float)
    maj_pred = (maj_score >= 0.6).astype(int)
    res += basic_metrics(lab, maj_pred, maj_score)

    # 3) Strict consensus, following the official DeepFRAG implementation.
    #    For healthy patients, any positive subsample means FP (score 1).
    #    For cancer patients, any negative subsample means FN (score 0).
    #    This vector is the paper's operational encoding of consensus errors.
    con_pred = np.where(
        lab == 1,
        g['all_pos'].to_numpy(dtype=int),
        g['any_pos'].to_numpy(dtype=int)
    ).astype(int)

    n_pos = int((lab == 1).sum())
    n_neg = int((lab == 0).sum())
    con_acc = (con_pred == lab).mean() * 100
    con_spc = (
        ((con_pred == 0) & (lab == 0)).sum() / n_neg * 100
        if n_neg else np.nan
    )
    con_sns = (
        ((con_pred == 1) & (lab == 1)).sum() / n_pos * 100
        if n_pos else np.nan
    )

    # As in the official reference code, calculate both AUCs from the
    # binary consensus outcome vector (not from the underlying probabilities).
    con_auc = _keras_auc(lab, con_pred, curve='ROC')
    con_auprc = _keras_auc(lab, con_pred, curve='PR')

    con_metrics = [
        np.round(con_acc, 3),
        np.round(con_spc, 3) if np.isfinite(con_spc) else np.nan,
        np.round(con_sns, 3) if np.isfinite(con_sns) else np.nan,
        np.round(con_auc, 5) if np.isfinite(con_auc) else np.nan,
        np.round(con_auprc, 5) if np.isfinite(con_auprc) else np.nan,
    ]
    res += con_metrics

    # 4) Mean predicted probability across all subsamples of each patient.
    avg_score = g['mean_p'].to_numpy(dtype=float)
    avg_pred = (avg_score > thr).astype(int)
    res += basic_metrics(lab, avg_pred, avg_score)

    return dict(zip(METRIC_COLS, res))

# Calculate AUC using patient-level aggregated predictions.
def patient_auc(meta, p):
    """Patient-level AUC from mean subsample probability (used for the OOF diagnostic)."""
    d = pd.DataFrame({'Sid': meta['Sid'].values, 'Label': meta['Label'].values, 'p': p})
    g = d.groupby('Sid').agg(Label=('Label', 'max'), mean_p=('p', 'mean'))
    return roc_auc(g['Label'].values, g['mean_p'].values)


# Select a classification threshold from the supplied scores.
def pick_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    y = np.asarray(y).astype(int)
    # Compute balanced accuracy for the supplied predictions.
    def bal_acc(t):
        sn = ((p > t) & (y == 1)).sum() / max((y == 1).sum(), 1)
        sp = ((p <= t) & (y == 0)).sum() / max((y == 0).sum(), 1)
        return 0.5 * (sn + sp)
    return float(max(grid, key=bal_acc))


# Select a threshold using patient-level predictions.
def pick_patient_threshold(meta, p):
    """
    Select threshold using only out-of-fold training predictions,
    after aggregating probabilities at patient level.
    """
    p = np.asarray(p, dtype=float).reshape(-1)

    if len(meta) != len(p):
        raise ValueError("Metadata and OOF probabilities have different lengths.")

    if not np.isfinite(p).all():
        raise ValueError("OOF probabilities contain NaN or infinity.")

    d = pd.DataFrame({
        'Sid': meta['Sid'].to_numpy(),
        'Label': meta['Label'].to_numpy(dtype=int),
        'p': p
    })

    label_counts = d.groupby('Sid')['Label'].nunique()
    if (label_counts > 1).any():
        raise ValueError("Inconsistent labels in patient-level threshold tuning.")

    patient_df = d.groupby('Sid', sort=False).agg(
        Label=('Label', 'first'),
        mean_p=('p', 'mean')
    )

    return pick_threshold(
        patient_df['Label'].to_numpy(dtype=int),
        patient_df['mean_p'].to_numpy(dtype=float)
    )

# ----------------------------------------------------------------------------
# one training instance
# ----------------------------------------------------------------------------
# Train and evaluate one model instance across its data splits.
def train_instance(X_tr, y_tr, w_tr, X_val, y_val, w_val, seed, cfg):
    """Train one model using patient-held-out validation and early stopping."""
    tf.keras.utils.set_random_seed(seed)

    n_train = X_tr.shape[0]
    if n_train == 0:
        raise ValueError("Training set is empty.")
    if len(y_tr) != n_train or len(w_tr) != n_train:
        raise ValueError("X_tr, y_tr and w_tr must have matching lengths.")
    if len(X_val) == 0:
        raise ValueError("Validation set is empty.")
    if len(y_val) != len(X_val) or len(w_val) != len(X_val):
        raise ValueError("X_val, y_val and w_val must have matching lengths.")

    steps_per_epoch = int(np.ceil(n_train / batch_size))
    model, optimizer = build_model(
        X_tr.shape[1], cfg, epochs * steps_per_epoch
    )
    train_step = make_train_step(model, optimizer)

    ds = (
        tf.data.Dataset
        .from_tensor_slices((X_tr, y_tr, w_tr))
        .shuffle(n_train, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
    )

    tr_loss_hist, tr_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
    best_val, best_weights, best_epoch, wait = np.inf, None, 0, 0
    time0 = time.time()

    for epoch in range(epochs):
        loss_avg = tf.keras.metrics.Mean()
        acc = tf.keras.metrics.BinaryAccuracy(threshold=0.0)  # logits threshold
        for x, y, w in ds:
            loss, logits = train_step(x, y, w)
            loss_avg.update_state(
                loss,
                sample_weight=tf.cast(tf.shape(y)[0], tf.float32)
            )
            acc.update_state(y, logits)

        tr_loss_hist.append(float(loss_avg.result()))
        tr_acc_hist.append(float(acc.result()) * 100)

        val_logits = predict_logits(model, X_val)
        val_loss = np_weighted_bce(val_logits, y_val, w_val)
        val_loss_hist.append(float(val_loss))
        val_acc_hist.append(
            float(((val_logits > 0).astype(int) == y_val.astype(int)).mean() * 100)
        )

        # Record validation metrics from epoch 1, but select checkpoints only
        # from min_epochs onward.
        if epoch + 1 < cfg['min_epochs']:
            continue

        if not np.isfinite(val_loss):
            wait += 1
        elif best_weights is None or val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch + 1
            wait = 0
            best_weights = model.get_weights()
        else:
            wait += 1

        if wait >= cfg['patience']:
            break

    if best_weights is None or best_epoch == 0:
        raise RuntimeError(
            "Early stopping failed: no valid checkpoint was selected. "
            "Check validation data, labels and validation loss."
        )

    model.set_weights(best_weights)
    tr_time = np.round(time.time() - time0, 2)

    info = {
        'ValLoss': np.round(best_val, 4),
        'TrLoss': np.round(tr_loss_hist[best_epoch - 1], 4),
        'TrAcc': np.round(tr_acc_hist[best_epoch - 1], 3),
        'TrEpochs': epoch + 1,
        'BestEpoch': best_epoch,
        'TrTime': tr_time,
    }
    history = {
        'train_loss': tr_loss_hist[:best_epoch],
        'val_loss': val_loss_hist[:best_epoch],
        'train_acc': tr_acc_hist[:best_epoch],
        'val_acc': val_acc_hist[:best_epoch],
    }
    return model, info, history


# Calculate the median curve across the supplied curves.
def _median_curve(curves):
    """Median by epoch, padding shorter runs with NaN."""
    if not curves:
        return np.array([], dtype=float)
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        arr[i, :len(curve)] = np.asarray(curve, dtype=float)
    with np.errstate(all='ignore'):
        return np.nanmedian(arr, axis=0)

# Calculate quantile bounds across the supplied curves.
def _quantile_curves(curves, q):
    """Epoch-wise quantile (e.g. Q1/Q3 for the IQR band), padding shorter runs with NaN."""
    if not curves:
        return np.array([], dtype=float)
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        arr[i, :len(curve)] = np.asarray(curve, dtype=float)
    with np.errstate(all='ignore'):
        return np.nanquantile(arr, q, axis=0)

# Plot a curve with its uncertainty band.
def _band_plot(ax, epochs, med, q1, q3, label, color, linewidth=2.0):
    """Median line + shaded Q1-Q3 IQR band (plt.fill_between)."""
    ax.plot(epochs, med, color=color, linewidth=linewidth, label=label)
    if len(med) and np.isfinite(q1).any() and np.isfinite(q3).any():
        ax.fill_between(epochs, q1, q3, color=color, alpha=0.20,
                        linewidth=0, label=f'{label} IQR (Q1-Q3)')

# Save metric curves for individual data splits.
def save_split_curves_plot(split_no, histories, out_dir):
    """Per-split median train/validation loss AND accuracy curves across instances,
    each with a shaded Q1-Q3 IQR band (across instances, epoch-wise)."""
    curves = {key: [h[key] for h in histories if h.get(key)]
              for key in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}

    # ---- loss plot ----
    tr, va = _median_curve(curves['train_loss']), _median_curve(curves['val_loss'])
    tr_q1, tr_q3 = _quantile_curves(curves['train_loss'], 0.25), _quantile_curves(curves['train_loss'], 0.75)
    va_q1, va_q3 = _quantile_curves(curves['val_loss'], 0.25), _quantile_curves(curves['val_loss'], 0.75)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n = max(len(tr), len(va))
    if n:
        ep = np.arange(1, n + 1)
        if len(tr):
            _band_plot(ax, ep[:len(tr)], tr, tr_q1, tr_q3, 'Median training loss', 'tab:blue')
        if len(va):
            _band_plot(ax, ep[:len(va)], va, va_q1, va_q3, 'Median validation loss', 'tab:orange')
    ax.set(title=f'Split {split_no:02d}: median loss across training instances (Q1-Q3 IQR band)',
           xlabel='Epoch', ylabel='Binary cross-entropy loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'Loss_Split_{split_no:02d}.png'), dpi=180)
    plt.close(fig)

    # ---- accuracy plot ----
    tr_a, va_a = _median_curve(curves['train_acc']), _median_curve(curves['val_acc'])
    tra_q1, tra_q3 = _quantile_curves(curves['train_acc'], 0.25), _quantile_curves(curves['train_acc'], 0.75)
    vaa_q1, vaa_q3 = _quantile_curves(curves['val_acc'], 0.25), _quantile_curves(curves['val_acc'], 0.75)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n = max(len(tr_a), len(va_a))
    if n:
        ep = np.arange(1, n + 1)
        if len(tr_a):
            _band_plot(ax, ep[:len(tr_a)], tr_a, tra_q1, tra_q3, 'Median training accuracy', 'tab:blue')
        if len(va_a):
            _band_plot(ax, ep[:len(va_a)], va_a, vaa_q1, vaa_q3, 'Median validation accuracy', 'tab:orange')
    ax.set(title=f'Split {split_no:02d}: median accuracy across training instances (Q1-Q3 IQR band)',
           xlabel='Epoch', ylabel='Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'Acc_Split_{split_no:02d}.png'), dpi=180)
    plt.close(fig)

    return {'train_loss': tr, 'val_loss': va, 'train_acc': tr_a, 'val_acc': va_a}

# Save overall median metric curves.
def save_overall_median_curves_plot(split_curves, out_dir):
    """Epoch-wise median of the 20 per-split median curves for loss and accuracy,
    each with a shaded Q1-Q3 IQR band computed across the 20 split curves."""
    for kind, fname, ylabel in (
            ('loss', 'Loss_Median_20_Splits', 'Binary cross-entropy loss'),
            ('acc', 'Acc_Median_20_Splits', 'Accuracy (%)')):
        keys = ('train_' + kind, 'val_' + kind)
        tr_c = [x[keys[0]] for x in split_curves if len(x.get(keys[0], []))]
        va_c = [x[keys[1]] for x in split_curves if len(x.get(keys[1], []))]
        tr, va = _median_curve(tr_c), _median_curve(va_c)
        tr_q1, tr_q3 = _quantile_curves(tr_c, 0.25), _quantile_curves(tr_c, 0.75)
        va_q1, va_q3 = _quantile_curves(va_c, 0.25), _quantile_curves(va_c, 0.75)

        fig, ax = plt.subplots(figsize=(10, 6))
        n = max(len(tr), len(va))
        if n:
            ep = np.arange(1, n + 1)
            if len(tr):
                _band_plot(ax, ep[:len(tr)], tr, tr_q1, tr_q3,
                           f'Median training {kind} (20 splits)', 'tab:blue', linewidth=2.2)
            if len(va):
                _band_plot(ax, ep[:len(va)], va, va_q1, va_q3,
                           f'Median validation {kind} (20 splits)', 'tab:orange', linewidth=2.2)
        ax.set(title=f'Overall median {kind} across 20 splits (Q1-Q3 IQR band across splits)',
               xlabel='Epoch', ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname + '.png'), dpi=200)
        plt.close(fig)

        # Export the plotted medians and IQR bands for reproducibility.
        max_len = max(len(tr), len(va))
        table = pd.DataFrame({'Epoch': np.arange(1, max_len + 1)})
        for name, arr in (('MedianTrain', tr), ('MedianVal', va),
                          ('TrainQ1', tr_q1), ('TrainQ3', tr_q3),
                          ('ValQ1', va_q1), ('ValQ3', va_q3)):
            table[name] = np.pad(arr.astype(float), (0, max_len - len(arr)), constant_values=np.nan)
        table.to_csv(os.path.join(out_dir, fname + '.csv'), index=False)

# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------
# Run the configured training and evaluation workflow.
def main():
    run, cfg = parse_args(sys.argv)
    resDir = os.path.join(resBase, run) + '/'
    os.makedirs(resDir, exist_ok=True)
    print('Run:', run)
    print('Config:', cfg)
    with open(resDir + 'config.pickle', 'wb') as handle:
        pickle.dump(cfg, handle)

    K = cfg['cv_folds']
    if Ntr_inst < K or Ntr_inst % K != 0:
        raise ValueError(
            f"Ntr_inst ({Ntr_inst}) must be >= cv_folds ({K}) and divisible by it "
            "so every training subsample receives the same number of OOF predictions."
        )
    expected_oof_repeats = Ntr_inst // K
    time000 = time.time()
    ens_rows, inst_mean_rows, inst_median_rows = [], [], []
    inst_detail_rows = []
    split_curve_data = []

    for v in range(1, Nsplits + 1):
        vers = str(v).zfill(2)
        with open(refDir + fDatName + vers + '.pickle', 'rb') as handle:
            fpathsTrn, labelsTrn, fpathsTst, labelsTst = pickle.load(handle)

        X_all = load_data(fpathsTrn)
        X_test = load_data(fpathsTst)
        if dwt_input:
            X_all = dec_dwt1D(X_all)
            X_test = dec_dwt1D(X_test)
        X_all = X_all.astype(np.float32)
        X_test = X_test.astype(np.float32)

        y_all = np.asarray(labelsTrn).astype(np.float32)

        meta_tr = make_meta_df(fpathsTrn, labelsTrn)
        meta_te = make_meta_df(fpathsTst, labelsTst)

        if v == 1:
            print("\n=== Patient ID parsing audit: first 10 training paths ===")
            for fp in fpathsTrn[:10]:
                print(
                    "PATH:", fp,
                    "| PARSED:", parse_path(fp)
                )

        print("\nUnique TRAIN patients:", meta_tr['Sid'].nunique())
        print("Unique TEST patients :", meta_te['Sid'].nunique())
        print("TRAIN subsamples      :", len(meta_tr))
        print("TEST subsamples       :", len(meta_te))

        validate_patient_metadata(meta_tr, "TRAIN")
        validate_patient_metadata(meta_te, "TEST")

        train_patients = set(meta_tr['Sid'].astype(str))
        test_patients = set(meta_te['Sid'].astype(str))
        overlap = train_patients & test_patients

        if overlap:
            raise ValueError(
                "Patient-level leakage risk: patient IDs occur in both TRAIN "
                f"and TEST. Examples: {sorted(overlap)[:10]}"
            )


        fold_ids = patient_fold_ids(meta_tr, K, seed=v)
        fold_class_counts = validate_patient_folds(
            meta_tr, fold_ids, K, split_name=f'Split {vers}'
        )
        if v == 1:
            print("Patient counts per validation fold (negative, positive):")
            print(fold_class_counts.to_string())
        oof_sum = np.zeros(len(y_all))
        oof_cnt = np.zeros(len(y_all))

        time00 = time.time()
        test_probs, infos, histories = [], [], []
        for itr in range(Ntr_inst):
            vm = fold_ids == (itr % K)

            X_tr, y_tr = X_all[~vm], y_all[~vm]
            X_val, y_val = X_all[vm], y_all[vm]

            if cfg['standardize']:
                mu = X_tr.mean(axis=0, keepdims=True)
                sd = X_tr.std(axis=0, keepdims=True) + 1e-8
            else:
                mu, sd = 0.0, 1.0
            X_tr = (X_tr - mu) / sd
            X_te = (X_test - mu) / sd
            X_val = (X_val - mu) / sd

            if cfg['class_weights']:
                w_tr = balanced_weights(y_tr)
                w_val = balanced_weights(y_val, reference_y=y_tr)
            else:
                w_tr = np.ones_like(y_tr, dtype=np.float32)
                w_val = np.ones_like(y_val, dtype=np.float32)

            model, info, history = train_instance(X_tr, y_tr, w_tr, X_val, y_val, w_val, seed=1000 * v + itr, cfg=cfg)
            histories.append(history)
            test_probs.append(sigmoid(predict_logits(model, X_te)))
            oof_sum[vm] += sigmoid(predict_logits(model, X_val))
            oof_cnt[vm] += 1
            infos.append(info)
            if itr % 5 == 0:
                print("Split {} Inst {} (fold {}): epochs {} (best {}) ValLoss {} TrLoss {:.4f} TrAcc {:.2f}% time {} s".format(
                    v, itr + 1, (itr % K), info['TrEpochs'], info['BestEpoch'],
                    info['ValLoss'], info['TrLoss'], info['TrAcc'], info['TrTime']))

            # Preserve the original behavior: export the final instance's
            # penultimate-layer representation for this split.
            if cfg['save_embeddings'] and itr == Ntr_inst - 1:
                embedding_path = os.path.join(
                    resDir, 'DNN_Emb', f'DNN_embeddings_split_{v:02d}.npz'
                )
                save_test_embeddings(
                    model=model,
                    X_test=X_te,
                    meta_test=meta_te,
                    output_path=embedding_path,
                    batch_size=4096,
                    split_id=v,
                    instance_index=itr + 1,
                    validation_fold=itr % K,
                    scaler_mean=mu,
                    scaler_scale=sd,
                )
                print(f"-> Saved DNN embeddings for Split {v:02d}: {embedding_path}")

            tf.keras.backend.clear_session()
        test_probs = np.stack(test_probs, axis=0)
        split_curve_data.append(save_split_curves_plot(v, histories, resDir))

        if not np.all(oof_cnt == expected_oof_repeats):
            observed = np.unique(oof_cnt.astype(int)).tolist()
            raise ValueError(
                "Incorrect OOF coverage: every training subsample should receive "
                f"exactly {expected_oof_repeats} predictions, but observed counts are {observed}."
            )
        oof_probs = oof_sum / oof_cnt

        thr = 0.5
        if cfg['use_val_threshold']:
            thr = pick_patient_threshold(meta_tr, oof_probs)

        perf_df = pd.DataFrame(index=np.arange(Ntr_inst), columns=METRIC_COLS + TRAIN_COLS, dtype=float)
        for itr in range(Ntr_inst):
            row = all_metrics(meta_te, test_probs[itr], thr)
            row.update(infos[itr])
            perf_df.loc[itr, :] = pd.Series(row)

        p_ens = test_probs.mean(axis=0)
        ens_row = pd.Series(all_metrics(meta_te, p_ens, thr))

        ens_rows.append(ens_row)
        inst_mean_rows.append(perf_df.mean())  # backward-compatible output
        inst_median_rows.append(perf_df[METRIC_COLS].median())
        for instance_idx, instance_row in perf_df.iterrows():
            detail = instance_row.to_dict()
            detail['Split'] = v
            detail['Instance'] = int(instance_idx) + 1
            inst_detail_rows.append(detail)
        
        # Embeddings are exported exactly once, inside the final training
        # instance above. Do not re-export here: this old block wrote to a
        # fixed path, omitted Sid, and could overwrite outputs from other runs.


        oof_msg = ' | OOF patient AUC {:.4f}'.format(
            patient_auc(meta_tr, oof_probs)
        )
        print('Split {}: {} instances in {} min | thr={:.2f} | Ensemble: AucSub {:.4f}, AucAvg {:.4f}, '
              'SpcAvg {:.1f}, SnsAvg {:.1f}{} | median best epoch {:.0f}'.format(
                  v, Ntr_inst, np.round((time.time() - time00) / 60, 2), thr,
                  ens_row['AucSub'], ens_row['AucAvg'], ens_row['SpcAvg'], ens_row['SnsAvg'], oof_msg,
                  perf_df['BestEpoch'].median()))
        print('   ')

        out = {'perf_df': perf_df, 'ens_perf': ens_row, 'threshold': thr, 'cfg': cfg,
               'test_meta': meta_te, 'test_probs': test_probs,
               'train_meta': meta_tr, 'oof_probs': oof_probs, 'fold_ids': fold_ids}
        with open(resDir + fResName + vers + '.pickle', 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # summary over splits (Table 2 style)
    # ------------------------------------------------------------------
    ens_df = pd.DataFrame(ens_rows).reset_index(drop=True)
    ens_df.index = np.arange(1, len(ens_df) + 1)
    inst_mean_df = pd.DataFrame(inst_mean_rows).reset_index(drop=True)
    inst_mean_df.index = np.arange(1, len(inst_mean_df) + 1)
    inst_median_df = pd.DataFrame(inst_median_rows).reset_index(drop=True)
    inst_median_df.index = np.arange(1, len(inst_median_df) + 1)
    inst_detail_df = pd.DataFrame(inst_detail_rows)

    # Summarize evaluation results across data splits.
    def summarize_splits(df):
        """Summary across the 20 official splits."""
        q1 = df.quantile(0.25)
        median = df.median()
        q3 = df.quantile(0.75)
        return pd.DataFrame({
            'median': median,
            'Q1': q1,
            'Q3': q3,
            'IQR_width': q3 - q1,
            'mean': df.mean()
        })

    # Ensemble results: one ensemble prediction per official split.
    sum_ens = summarize_splits(ens_df[METRIC_COLS])

    # For each split, summarize the 20 independent training instances.
    per_split_rows = []
    for split_id, split_group in inst_detail_df.groupby('Split', sort=True):
        row = {'Split': int(split_id)}
        for metric in METRIC_COLS:
            values = split_group[metric].dropna()
            row[f'{metric}_median'] = values.median() if len(values) else np.nan
            row[f'{metric}_Q1'] = values.quantile(0.25) if len(values) else np.nan
            row[f'{metric}_Q3'] = values.quantile(0.75) if len(values) else np.nan
            row[f'{metric}_IQR'] = (
                row[f'{metric}_Q3'] - row[f'{metric}_Q1']
                if len(values) else np.nan
            )
        per_split_rows.append(row)
    per_split_instance_summary = pd.DataFrame(per_split_rows).set_index('Split')

    # Variability caused by model stochasticity:
    # median of the within-split IQRs across the 20 splits.
    stochasticity_summary = pd.DataFrame(index=METRIC_COLS)
    for metric in METRIC_COLS:
        iqr_values = per_split_instance_summary[f'{metric}_IQR'].dropna()
        stochasticity_summary.loc[metric, 'median_within_split_IQR'] = (
            iqr_values.median() if len(iqr_values) else np.nan
        )
        stochasticity_summary.loc[metric, 'Q1_within_split_IQR'] = (
            iqr_values.quantile(0.25) if len(iqr_values) else np.nan
        )
        stochasticity_summary.loc[metric, 'Q3_within_split_IQR'] = (
            iqr_values.quantile(0.75) if len(iqr_values) else np.nan
        )

    # Variability caused by data composition:
    # distribution of the 20 split-specific medians.
    split_median_summary = summarize_splits(inst_median_df[METRIC_COLS])

    # Preserve the previous summary output for backward compatibility.
    sum_inst_mean = summarize_splits(inst_mean_df[METRIC_COLS])

    ens_df.to_csv(resDir + 'PerSplit_ensemble.csv', index_label='Split')
    inst_mean_df.to_csv(resDir + 'PerSplit_instanceMean.csv', index_label='Split')
    inst_median_df.to_csv(resDir + 'PerSplit_instanceMedian.csv', index_label='Split')
    inst_detail_df.to_csv(resDir + 'PerInstance_testMetrics.csv', index=False)
    per_split_instance_summary.to_csv(resDir + 'PerSplit_instanceDistribution.csv', index_label='Split')
    sum_ens.to_csv(resDir + 'Summary_ensemble.csv')
    sum_inst_mean.to_csv(resDir + 'Summary_instanceMean_legacy.csv')
    stochasticity_summary.to_csv(resDir + 'Summary_modelStochasticity.csv')
    split_median_summary.to_csv(resDir + 'Summary_splitComposition.csv')

    save_overall_median_curves_plot(split_curve_data, resDir)

    pd.set_option('display.width', 200)
    print('\n=== Summary over {} splits: ensemble ==='.format(Nsplits))
    print(sum_ens.round(3).to_string())
    print('\n=== Median of 20 instances within each split ===')
    print(inst_median_df[METRIC_COLS].round(3).to_string())
    print('\n=== Model stochasticity: median within-split IQR ===')
    print(stochasticity_summary.round(5).to_string())
    print('\n=== Split-composition variability: distribution of split medians ===')
    print(split_median_summary.round(5).to_string())
    print('\nRun {} completed in {} hrs'.format(run, np.round((time.time() - time000) / 3600, 3)))


if __name__ == '__main__':
    main()