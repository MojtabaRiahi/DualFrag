#!/usr/bin/env python
# coding: utf-8
"""
Graph-only inductive GCN line for the DNN-GCN-Fusion pipeline.

Loads pre-built graphs from GCN_Maker_updated.py; rebuilds fold-specific graph statistics without leakage.
Raw 200-dimensional PMF profiles are encoded by a fold-standardized PMFEncoder
to initialize sample nodes. Token/region message passing, KNN/patient messages,
the separate DNN branch, and adaptive fusion interfaces remain unchanged.

Main safeguards:
  1. Validates train/test patient separation and patient-level CV folds.
  2. Does not load sample_token_weights/region_features as direct model inputs.
  3. Restricts KNN and same-patient message sources to FIT samples per fold.
  4. Uses training labels only in the loss; validation is used for early stopping.
  5. Produces OOF and test probabilities in a pickle layout suitable for fusion.
  6. Uses DNN-aligned patient metric definitions (Maj, class-conditioned Con, Avg).
  7. Defines Con without consulting the true patient label.

Usage:
    python GCN_Model.py main
    python GCN_Model.py adaptive_mean_max  # adaptive mean-max patient aggregation
    python GCN_Model.py no_knn
    python GCN_Model.py main hidden=128 dropout=0.4 lr=5e-4

Results:
    ./data/Test_Results/GCN_Main/<run_name>/
"""

import os
import sys
import ast
import re
import copy
import time
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, eye

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any


# ----------------------------------------------------------------------------
# Constants / configuration
# ----------------------------------------------------------------------------
SEED_BASE = 1000
RES_BASE = "./data/Test_Results/GCN_Graph/"
fResName = "TestPerf_df"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUPS = ["Sub", "Maj", "Con", "Avg"]
METRICS = ["Acc", "Spc", "Sns", "Auc", "AucPR"]
METRIC_COLS = [m + g for g in GROUPS for m in METRICS]

# Keep the decision operating point identical to DNN_Model.py.
DECISION_THRESHOLD = 0.5
TRAIN_COLS = ["ValLoss", "TrLoss", "TrAcc", "TrEpochs", "BestEpoch", "TrTime"]

DEFAULT_CFG = dict(
    graph_dir="./output_graph",
    n_instances=20,
    use_token_msg=True,
    use_knn=True,
    use_patient=False,
    knn_k=10,
    hidden=64,
    layers=2,
    head=16,
    dropout=0.35,
    residual=True,
    class_weights=True,
    lr=0.0002,
    weight_decay=0.001,
    epochs=400,
    min_epochs=20,
    patience=20,
    min_delta=1e-4,
    clip=5.0,
    threshold_method="fixed",  # Fixed at 0.5 to match DNN_Model.py
    patient_aggregation="mean",  # primary method: mean, top2_mean, or max
    aggregation_alpha=0.25,  # adaptive score=(1-alpha)*mean + alpha*max
    target_sensitivity=0.98,
)

PRESETS = {
    "main": dict(use_knn=True),
    "no_knn": dict(use_knn=False),
    "no_patient": dict(use_patient=False),
    "no_classw": dict(class_weights=False),
}


# Parse command-line arguments and validate the run configuration.
def parse_args(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    run = argv[1]
    cfg = dict(DEFAULT_CFG)
    cfg.update(PRESETS.get(run, {}))

    for kv in argv[2:]:
        if "=" not in kv:
            raise ValueError(f"Override must be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise ValueError(f"Unknown config key '{k}'. Valid keys: {sorted(cfg)}")
        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v

    if cfg["min_epochs"] > cfg["epochs"]:
        raise ValueError("min_epochs must be <= epochs.")
    if cfg["n_instances"] < 1:
        raise ValueError("n_instances must be >= 1.")
    if not cfg["use_token_msg"]:
        raise ValueError("Graph-only mode requires use_token_msg=True so sample nodes receive token/region graph messages.")
    if cfg["layers"] < 1:
        raise ValueError("layers must be >= 1.")
    if cfg["hidden"] < 1 or cfg["head"] < 1:
        raise ValueError("hidden and head must be positive.")
    if not (0.0 <= float(cfg["dropout"]) < 1.0):
        raise ValueError("dropout must be in [0, 1).")
    if cfg["patience"] < 1:
        raise ValueError("patience must be >= 1.")
    # The DNN branch uses a pre-specified 0.5 cutoff by default. Keep GCN
    # identical; data-driven threshold tuning is deliberately disabled.
    if cfg.get("threshold_method", "fixed") != "fixed":
        raise ValueError(
            "GCN threshold is fixed at 0.5 to remain consistent with DNN_Model.py; "
            "threshold_method must be 'fixed'."
        )
    cfg["threshold_method"] = "fixed"
    if cfg.get("patient_aggregation", "mean") != "mean":
        raise ValueError(
            "The primary GCN patient aggregation is fixed to 'mean' to match "
            "DNN_Model.py; top2_mean/max are available only in supplementary "
            "aggregation-comparison diagnostics."
        )
    cfg["patient_aggregation"] = "mean"
    cfg["aggregation_alpha"] = float(cfg.get("aggregation_alpha", 0.25))
    if not (0.0 <= cfg["aggregation_alpha"] <= 1.0):
        raise ValueError("aggregation_alpha must be in [0, 1].")
    target_sensitivity = float(cfg.get("target_sensitivity", 0.95))
    if not (0.0 < target_sensitivity <= 1.0):
        raise ValueError("target_sensitivity must be in (0, 1].")
    cfg["target_sensitivity"] = target_sensitivity
    return run, cfg


# Seed the random number generators used by the model.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Graph loading, validation and sparse matrices
# ----------------------------------------------------------------------------
# Load a graph artifact and validate its required fields.
def load_graph(path):
    with open(path, "rb") as f:
        g = pickle.load(f)

    needed = [
        "labels", "train_mask", "test_mask", "profiles", "patient_ids",
        "subsample_ids", "ex_ids", "fold_ids", "cv_folds", "split_number",
    ]
    missing = [k for k in needed if k not in g]
    if missing:
        raise KeyError(
            f"{os.path.basename(path)} is missing required graph keys: {missing}"
        )
    return g


# Validate patient separation and cross-validation assignments.
def validate_patient_splits(g, train_mask, test_mask, fold_ids, K):
    """Validate patient separation and that each training patient has one CV fold."""
    labels = np.asarray(g["labels"])
    patients = np.asarray(g["patient_ids"], dtype=object)

    if labels.ndim != 1 or patients.ndim != 1:
        raise ValueError("labels and patient_ids must be one-dimensional.")
    if len(patients) != len(labels):
        raise ValueError("patient_ids and labels have different lengths.")
    if pd.isna(patients).any():
        raise ValueError("Missing patient IDs found in graph.")

    patient_str = np.asarray([str(x).strip() for x in patients], dtype=object)
    if np.any(patient_str == ""):
        raise ValueError("Empty patient IDs found in graph.")

    try:
        labels = labels.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be integer-like 0/1 values.") from exc

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary: 0=Healthy, 1=Cancer.")

    if train_mask.shape != labels.shape or test_mask.shape != labels.shape:
        raise ValueError("train_mask/test_mask lengths must match labels.")
    if np.any(train_mask & test_mask):
        raise ValueError("A sample is marked as both TRAIN and TEST.")
    if not np.all(train_mask | test_mask):
        raise ValueError("Some samples are neither TRAIN nor TEST.")

    train_patients = set(patient_str[train_mask])
    test_patients = set(patient_str[test_mask])
    overlap = train_patients.intersection(test_patients)
    if overlap:
        raise ValueError(
            "Patient leakage risk: patient IDs occur in both TRAIN and TEST. "
            f"Examples: {sorted(overlap)[:10]}"
        )

    fold_ids = np.asarray(fold_ids)
    if len(fold_ids) != int(train_mask.sum()):
        raise ValueError("fold_ids length does not match the number of TRAIN samples.")
    if not np.isfinite(fold_ids).all():
        raise ValueError("fold_ids contains non-finite values.")

    fold_ids = fold_ids.astype(int)
    if K < 2:
        raise ValueError("cv_folds must be >= 2.")
    if not np.isin(fold_ids, np.arange(K)).all():
        raise ValueError(f"fold_ids must be in the range 0..{K - 1}.")

    train_patient_ids = patient_str[train_mask]
    train_labels = labels[train_mask]

    patient_fold_counts = pd.DataFrame({
        "Sid": train_patient_ids,
        "fold": fold_ids,
    }).groupby("Sid")["fold"].nunique()

    bad_patients = patient_fold_counts[patient_fold_counts != 1]
    if not bad_patients.empty:
        raise ValueError(
            "Patient-level CV leakage risk: a patient occurs in multiple folds. "
            f"Examples: {bad_patients.index.tolist()[:10]}"
        )

    patient_label_counts = pd.DataFrame({
        "Sid": train_patient_ids,
        "Label": train_labels,
    }).groupby("Sid")["Label"].nunique()

    if (patient_label_counts != 1).any():
        raise ValueError("Some training patients have inconsistent labels.")

    # Also validate label consistency in the test patients.
    test_label_counts = pd.DataFrame({
        "Sid": patient_str[test_mask],
        "Label": labels[test_mask],
    }).groupby("Sid")["Label"].nunique()

    if (test_label_counts != 1).any():
        raise ValueError("Some test patients have inconsistent labels.")

    print(
        f"Split audit: train samples={int(train_mask.sum())}, "
        f"test samples={int(test_mask.sum())}, "
        f"train patients={len(train_patients)}, "
        f"test patients={len(test_patients)}, patient overlap=0, CV folds={K}"
    )


# Normalize sparse matrix rows.
def row_normalize(A):
    A = A.tocsr().astype(np.float32)
    A.eliminate_zeros()
    d = np.asarray(A.sum(axis=1)).ravel()
    d[d <= 0] = 1.0
    return (diags((1.0 / d).astype(np.float32)) @ A).tocsr()


# Normalize sparse rows after applying a column mask.
def masked_row_normalize(A, col_mask):
    """Keep only columns in col_mask, then row-normalise."""
    B = (A.tocsr().astype(np.float32) @ diags(col_mask.astype(np.float32))).tocsr()
    B.eliminate_zeros()
    return row_normalize(B)


# Remove self-loops from a sparse adjacency matrix.
def remove_diagonal(A):
    """Remove sample self-loops from a sample-to-sample adjacency matrix."""
    A = A.tocsr().astype(np.float32).copy()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


# Validate the raw graph split inputs.
def validate_graph_inputs(g):
    """Validate raw split data; no precomputed adjacency is required."""
    required = ("profiles", "labels", "train_mask", "test_mask")
    missing = [key for key in required if key not in g]
    if missing:
        raise KeyError(f"Split data is missing required keys: {missing}")
    labels = np.asarray(g["labels"])
    profiles = np.asarray(g["profiles"])
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if profiles.shape != (len(labels), 200):
        raise ValueError(f"Expected profiles with shape ({len(labels)}, 200), got {profiles.shape}.")
    if not np.isfinite(profiles).all():
        raise ValueError("profiles contain NaN or infinite values.")
    for key in ("train_mask", "test_mask", "patient_ids", "subsample_ids", "ex_ids", "fold_ids"):
        if key not in g or len(g[key]) != len(labels):
            raise ValueError(f"{key} must be present and have one entry per sample.")
    if "cv_folds" not in g or "split_number" not in g:
        raise KeyError("Split data must include cv_folds and split_number.")
    return len(labels)


# Build sample metadata for the selected indices.
def make_meta(g, idx):
    n = len(g["labels"])
    fc = np.asarray(g["fc_ids"], dtype=object) if "fc_ids" in g else np.full(n, "", dtype=object)
    return pd.DataFrame({
        "FCid": fc[idx],
        "Sid": np.asarray(g["patient_ids"], dtype=object)[idx],
        "Subid": np.asarray(g["subsample_ids"], dtype=object)[idx],
        "ExSid": np.asarray(g["ex_ids"], dtype=object)[idx],
        "Label": np.asarray(g["labels"])[idx].astype(int),
    }).reset_index(drop=True)


# Compute inverse-frequency class weights.
def balanced_weights(y):
    y = np.asarray(y).astype(int)
    n = len(y)
    if n == 0:
        raise ValueError("Cannot calculate class weights for an empty set.")
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    return np.where(
        y == 1,
        n / (2 * max(n1, 1.0)),
        n / (2 * max(n0, 1.0)),
    ).astype(np.float32)


# Convert a SciPy sparse matrix to a Torch sparse tensor.
def to_torch_sparse(A):
    A = A.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, size=A.shape).coalesce().to(DEVICE)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
# Calculate the ROC AUC using ranked scores.
def roc_auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(y) != len(s):
        raise ValueError("y and score have different lengths.")
    if not np.isfinite(s).all():
        raise ValueError("Non-finite scores passed to ROC AUC.")
    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    if P == 0 or N == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").values
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


# Calculate average precision from ranked scores.
def pr_auc(y, s):
    """Average Precision (AP), grouping tied scores at the same threshold."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(y) != len(s):
        raise ValueError("y and score have different lengths.")
    if not np.isfinite(s).all():
        raise ValueError("Non-finite scores passed to PR AUC.")
    P = int(np.sum(y == 1))
    if P == 0:
        return np.nan

    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / P
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


# Calculate the standard binary classification metrics.
def basic_metrics(y, pred, score):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    score = np.asarray(score, dtype=float)

    if not (len(y) == len(pred) == len(score)):
        raise ValueError("y, pred and score must have equal lengths.")
    if len(y) == 0:
        raise ValueError("Cannot calculate metrics on an empty set.")

    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    acc = np.mean(pred == y) * 100.0
    spc = np.sum((pred == 0) & (y == 0)) / max(N, 1) * 100.0
    sns = np.sum((pred == 1) & (y == 1)) / max(P, 1) * 100.0

    return [
        np.round(acc, 3),
        np.round(spc, 3),
        np.round(sns, 3),
        np.round(roc_auc(y, score), 5),
        np.round(pr_auc(y, score), 5),
    ]


# Aggregate subsample probabilities into patient scores.
def aggregate_patient_scores(meta, probabilities, method="mean", alpha=0.25):
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(meta) != len(p):
        raise ValueError("Metadata and probability array lengths differ.")
    allowed = ("mean", "top2_mean", "max", "adaptive_mean_max")
    if method not in allowed:
        raise ValueError(f"Patient aggregation must be one of {allowed}.")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    d = pd.DataFrame({"Sid": meta["Sid"].astype(str).to_numpy(),
                      "Label": meta["Label"].astype(int).to_numpy(), "p": p})
    if (d.groupby("Sid")["Label"].nunique() != 1).any():
        raise ValueError("Inconsistent labels within patient during aggregation.")

    rows = []
    for sid, group in d.groupby("Sid", sort=False):
        vals = group["p"].to_numpy(dtype=float)
        mean_p = float(np.mean(vals))
        max_p = float(np.max(vals))
        if method == "mean":
            score = mean_p
        elif method == "max":
            score = max_p
        elif method == "adaptive_mean_max":
            score = float((1.0 - alpha) * mean_p + alpha * max_p)
        else:
            top = np.sort(vals)[-min(2, len(vals)):]
            score = float(np.mean(top))
        rows.append((sid, int(group["Label"].iloc[0]), score, len(vals),
                     float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                     mean_p, max_p))
    return pd.DataFrame(rows, columns=["Sid", "Label", "PatientProbability", "NSubsamples",
                                       "SubsampleProbabilitySD", "PatientMeanProbability",
                                       "PatientMaxProbability"])


# Calculate subsample-level and patient-level metrics.
def all_metrics(meta, p, thr, patient_aggregation="mean", aggregation_alpha=0.25):
    """Calculate metrics with exactly the patient-level rules used by DNN_Model.py.

    Sub: per-subsample prediction; score is the subsample probability.
    Maj: hard decision is positive-sub-sample fraction >= 0.6; AUROC/AP
         use that same positive fraction as the continuous ranking score.
    Con: class-conditioned consensus evaluation rule copied from DNN:
         cancer (label=1) is positive only if all subsamples are positive;
         healthy (label=0) is positive if any subsample is positive.
         All five Con metrics, including AUROC/AP, use the resulting binary
         consensus output. This is an evaluation/error-encoding rule, not a
         deployable patient prediction rule because it conditions on true label.
    Avg: patient mean probability; hard decision uses the fixed threshold.

    patient_aggregation and aggregation_alpha remain in the signature for
    backward compatibility with existing call sites, but the reported DNN-
    aligned metrics intentionally use the arithmetic mean for Avg.
    """
    y = np.asarray(meta["Label"].to_numpy(), dtype=int)
    p = np.asarray(p, dtype=float).reshape(-1)

    if len(y) != len(p):
        raise ValueError(f"Length mismatch: {len(y)} labels vs {len(p)} probabilities.")
    if len(y) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Predicted probabilities must be finite and in [0, 1].")
    if not np.isfinite(thr) or not (0 <= float(thr) <= 1):
        raise ValueError("Threshold must be finite and in [0, 1].")
    if "Sid" not in meta.columns or meta["Sid"].isna().any():
        raise ValueError("Missing patient IDs in metric metadata.")

    # Same strict comparison as DNN_Model.py: probability > 0.5.
    pred = (p > float(thr)).astype(int)
    d = pd.DataFrame({
        "Sid": meta["Sid"].to_numpy(),
        "Label": y,
        "p": p,
        "pos": pred,
    })

    label_counts = d.groupby("Sid")["Label"].nunique()
    bad_patients = label_counts[label_counts > 1].index.tolist()
    if bad_patients:
        raise ValueError(
            "Inconsistent labels within patient(s): "
            + ", ".join(map(str, bad_patients[:10]))
        )

    # Keep exactly the same group statistics and patient ordering as DNN.
    g = d.groupby("Sid", sort=False).agg(
        Label=("Label", "first"),
        frac_pos=("pos", "mean"),
        mean_p=("p", "mean"),
        any_pos=("pos", "max"),
        all_pos=("pos", "min"),
    )
    lab = g["Label"].to_numpy(dtype=int)

    # 1) Subsample-level metrics.
    res = basic_metrics(y, pred, p)

    # 2) Majority: same >= 0.6 support rule and same continuous score as DNN.
    maj_score = g["frac_pos"].to_numpy(dtype=float)
    maj_pred = (maj_score >= 0.6).astype(int)
    res += basic_metrics(lab, maj_pred, maj_score)

    # 3) Class-conditioned consensus evaluation: exactly as DNN_Model.py.
    con_pred = np.where(
        lab == 1,
        g["all_pos"].to_numpy(dtype=int),
        g["any_pos"].to_numpy(dtype=int),
    ).astype(int)
    con_acc = np.mean(con_pred == lab) * 100.0
    n_pos = int(np.sum(lab == 1))
    n_neg = int(np.sum(lab == 0))
    con_spc = (
        np.sum((con_pred == 0) & (lab == 0)) / n_neg * 100.0
        if n_neg else np.nan
    )
    con_sns = (
        np.sum((con_pred == 1) & (lab == 1)) / n_pos * 100.0
        if n_pos else np.nan
    )
    if len(np.unique(lab)) == 2:
        con_auc = roc_auc(lab, con_pred)
        con_auprc = pr_auc(lab, con_pred)
    else:
        con_auc = np.nan
        con_auprc = np.nan
    res += [
        np.round(con_acc, 3),
        np.round(con_spc, 3),
        np.round(con_sns, 3),
        np.round(con_auc, 3) if np.isfinite(con_auc) else np.nan,
        np.round(con_auprc, 3) if np.isfinite(con_auprc) else np.nan,
    ]

    # 4) Average: DNN uses the arithmetic mean of subsample probabilities.
    avg_score = g["mean_p"].to_numpy(dtype=float)
    avg_pred = (avg_score > float(thr)).astype(int)
    res += basic_metrics(lab, avg_pred, avg_score)

    return dict(zip(METRIC_COLS, res))


# Calculate ROC AUC from aggregated patient scores.
def patient_auc(meta, p, patient_aggregation="mean", aggregation_alpha=0.25):
    """Patient-level ROC AUC using the selected subsample-to-patient aggregation."""
    patient = aggregate_patient_scores(meta, p, patient_aggregation, aggregation_alpha)
    return roc_auc(patient["Label"].values, patient["PatientProbability"].values)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class GraphLayer(nn.Module):
    """Message-passing layer for sample nodes and optional global nodes."""

    # Initialize the model component.
    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.s_self = nn.Linear(dim, dim)
        self.s_norm = nn.LayerNorm(dim)

        if cfg["use_token_msg"]:
            self.s_tok = nn.Linear(dim, dim, bias=False)
            self.g_self = nn.Linear(dim, dim)
            self.g_msg = nn.Linear(dim, dim, bias=False)
            self.g_norm = nn.LayerNorm(dim)
        if cfg["use_knn"]:
            self.s_knn = nn.Linear(dim, dim, bias=False)
        if cfg["use_patient"]:
            self.s_pat = nn.Linear(dim, dim, bias=False)

        self.drop = nn.Dropout(cfg["dropout"])

    # Compute the model component forward pass.
    def forward(self, s, g, A):
        h = self.s_self(s)

        if self.cfg["use_token_msg"]:
            h = h + self.s_tok(torch.sparse.mm(A["sg"], g))
        if self.cfg["use_knn"]:
            h = h + self.s_knn(torch.sparse.mm(A["knn"], s))
        if self.cfg["use_patient"]:
            h = h + self.s_pat(torch.sparse.mm(A["pat"], s))

        out = self.drop(F.relu(self.s_norm(h)))
        s_new = s + out if self.cfg["residual"] else out

        if self.cfg["use_token_msg"]:
            g_out = self.g_self(g) + self.g_msg(torch.sparse.mm(A["gg"], g))
            g_new = self.drop(F.relu(self.g_norm(g_out)))
        else:
            g_new = g

        return s_new, g_new


class PMFEncoder(nn.Module):
    """Encode each raw 200-bin PMF profile into the GCN hidden dimension."""
    # Initialize the model component.
    def __init__(self, hidden, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    # Compute the model component forward pass.
    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 200:
            raise ValueError(f"PMFEncoder expects [n_samples, 200], got {tuple(x.shape)}")
        return self.net(x)


class GraphNet(nn.Module):
    """Graph-only branch with PMF-encoded sample-node initial states."""
    # Initialize the model component.
    def __init__(self, n_samples, n_global, cfg):
        super().__init__()
        hidden = cfg["hidden"]
        self.n_samples = int(n_samples)
        self.pmf_encoder = PMFEncoder(hidden, dropout=0.2)
        self.sample_init = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.sample_init, mean=0.0, std=0.1)
        self.g_emb = nn.Parameter(torch.randn(n_global, hidden) * 0.1)
        self.layers = nn.ModuleList([
            GraphLayer(hidden, cfg) for _ in range(cfg["layers"])
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, cfg["head"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["head"], 1),
        )

    # Compute the model component forward pass.
    def forward(self, A, pmf_x):
        if pmf_x.shape[0] != self.n_samples:
            raise ValueError("PMF feature row count does not match graph sample nodes")
        # PMF-derived sample state plus a learned shared bias, then unchanged GCN.
        s = self.pmf_encoder(pmf_x) + self.sample_init.unsqueeze(0)
        g = self.g_emb
        for layer in self.layers:
            s, g = layer(s, g, A)
        return self.head(s).squeeze(1)


# Calculate weighted binary cross-entropy loss.
def weighted_bce(logits, y, w):
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (w * per).sum() / torch.clamp(w.sum(), min=1e-8)


# ----------------------------------------------------------------------------
# Fold-specific graph reconstruction: label-derived statistics use FIT only
# ----------------------------------------------------------------------------
NUM_TOKENS_FOLD = 13
WINDOW_SIZE_FOLD = 16
STEP_SIZE_FOLD = 1
CRF_ALPHA_FOLD = 5.0
CRF_GAMMA_FOLD = 0.50
CRF_CLIP_FOLD = 2.50
SAMPLE_WEIGHT_POWER_FOLD = 0.50
SAMPLE_TOKEN_TOP_K_FOLD = 13
TOKEN_TOKEN_TOP_K_FOLD = 4
TOKEN_TOKEN_MIN_SUPPORT_FOLD = 5
TOKEN_TOKEN_MIN_SUPPORT_FRAC_FOLD = 5e-4
TOKEN_CLASS_MIN_SUPPORT_FOLD = 5
SAMPLE_TOKEN_MIN_WEIGHT_FOLD = 0.0


# Collect graph statistics from the fitting samples.
def _collect_fold_statistics(sequences, labels, fit_idx):
    """Count token occurrences and ordered within-window co-occurrences using FIT only."""
    token_window_counts = np.zeros(NUM_TOKENS_FOLD, dtype=np.float64)
    label_window_counts = np.zeros(2, dtype=np.float64)
    token_label_counts = np.zeros((NUM_TOKENS_FOLD, 2), dtype=np.float64)
    token_pair_counts = np.zeros((NUM_TOKENS_FOLD, NUM_TOKENS_FOLD), dtype=np.float64)
    token_sample_counts = np.zeros(NUM_TOKENS_FOLD, dtype=np.int64)
    total_windows = 0
    total_pair_mass = 0.0
    for idx in fit_idx:
        seq = np.asarray(sequences[idx], dtype=np.int64).reshape(-1)
        label = int(labels[idx])
        if label not in (0, 1):
            raise ValueError(f"Unexpected class label {label} at sample {idx}.")
        if seq.size and (seq.min() < 0 or seq.max() >= NUM_TOKENS_FOLD):
            raise ValueError(f"Token IDs must be in 0..{NUM_TOKENS_FOLD-1}.")
        token_sample_counts[np.unique(seq)] += 1
        if seq.size < WINDOW_SIZE_FOLD:
            continue
        for start in range(0, seq.size - WINDOW_SIZE_FOLD + 1, STEP_SIZE_FOLD):
            window = seq[start:start + WINDOW_SIZE_FOLD]
            counts = np.bincount(window, minlength=NUM_TOKENS_FOLD).astype(np.float64)
            total_windows += 1
            label_window_counts[label] += 1
            token_window_counts += counts
            token_label_counts[:, label] += counts
            # Ordered pairs of distinct positions; repeated token occurrences count multiplicatively.
            token_pair_counts += np.outer(counts, counts) - np.diag(counts)
            total_pair_mass += WINDOW_SIZE_FOLD * (WINDOW_SIZE_FOLD - 1)
    return {
        "total_windows": int(total_windows),
        "total_pair_mass": float(total_pair_mass),
        "token_window_counts": token_window_counts,
        "label_window_counts": label_window_counts,
        "token_label_window_counts": token_label_counts,
        "token_pair_window_counts": token_pair_counts,
        "token_sample_counts": token_sample_counts,
    }


# Build the fold-specific token PPMI matrix.
def _fold_ppmi(stats):
    """Match the graph builder's count-based PPMI and top-k pruning."""
    pair_counts = np.asarray(stats["token_pair_window_counts"], dtype=np.float64)
    total_pair_mass = float(stats["total_pair_mass"])
    token_counts = np.asarray(stats["token_window_counts"], dtype=np.float64)
    ppmi = np.zeros((NUM_TOKENS_FOLD, NUM_TOKENS_FOLD), dtype=np.float64)
    if total_pair_mass <= 0 or token_counts.sum() <= 0:
        return ppmi.astype(np.float32)
    marginal = token_counts * (WINDOW_SIZE_FOLD - 1)
    p_token = marginal / max(marginal.sum(), 1.0)
    min_support = max(TOKEN_TOKEN_MIN_SUPPORT_FOLD, TOKEN_TOKEN_MIN_SUPPORT_FRAC_FOLD * total_pair_mass)
    for i in range(NUM_TOKENS_FOLD):
        for j in range(i + 1, NUM_TOKENS_FOLD):
            support = pair_counts[i, j]
            denom = p_token[i] * p_token[j]
            if support >= min_support and denom > 0:
                value = np.log2((support / total_pair_mass) / denom)
                if value > 0:
                    ppmi[i, j] = ppmi[j, i] = value
    pruned = np.zeros_like(ppmi)
    for i in range(NUM_TOKENS_FOLD):
        neighbors = np.flatnonzero(ppmi[i] > 0)
        if neighbors.size > TOKEN_TOKEN_TOP_K_FOLD:
            neighbors = neighbors[np.argsort(ppmi[i, neighbors])[::-1][:TOKEN_TOKEN_TOP_K_FOLD]]
        pruned[i, neighbors] = ppmi[i, neighbors]
    return np.maximum(pruned, pruned.T).astype(np.float32)


# Build fold-specific token-label PPMI values.
def _fold_token_label_ppmi(stats):
    """Leakage-safe token-label PPMI; uses FIT labels only."""
    token_counts = np.asarray(stats["token_window_counts"], dtype=np.float64)
    label_counts = np.asarray(stats["label_window_counts"], dtype=np.float64)
    pair_counts = np.asarray(stats["token_label_window_counts"], dtype=np.float64)
    total = float(token_counts.sum())
    out = np.zeros((NUM_TOKENS_FOLD, 2), dtype=np.float32)
    if total <= 0:
        return out
    p_token = token_counts / total
    p_label = label_counts / max(label_counts.sum(), 1.0)
    for t in range(NUM_TOKENS_FOLD):
        for c in range(2):
            joint = pair_counts[t, c] / total
            denom = p_token[t] * p_label[c]
            if pair_counts[t, c] >= TOKEN_CLASS_MIN_SUPPORT_FOLD and joint > 0 and denom > 0:
                value = np.log2(joint / denom)
                if value > 0:
                    out[t, c] = value
    return out


# Compute fold-specific sample-token weights.
def _fold_sample_token_weights(sequences, stats, n_fit):
    """Rebuild TF x CRF weights without IDF, using FIT-derived CRF only."""
    token_label = np.asarray(stats["token_label_window_counts"], dtype=np.float64)
    cancer, healthy = token_label[:, 1], token_label[:, 0]
    log_odds = np.clip(np.log((cancer + CRF_ALPHA_FOLD) / (healthy + CRF_ALPHA_FOLD)), -CRF_CLIP_FOLD, CRF_CLIP_FOLD)
    crf = np.exp(CRF_GAMMA_FOLD * log_odds)
    weights = np.zeros((len(sequences), NUM_TOKENS_FOLD), dtype=np.float32)
    for sample_idx, sequence in enumerate(sequences):
        seq = np.asarray(sequence, dtype=np.int64).reshape(-1)
        counts = np.bincount(seq, minlength=NUM_TOKENS_FOLD).astype(np.float64)
        tf = counts / max(seq.size, 1)
        raw = np.maximum(tf * crf, 0.0)
        positive = np.flatnonzero(raw > SAMPLE_TOKEN_MIN_WEIGHT_FOLD)
        if positive.size > SAMPLE_TOKEN_TOP_K_FOLD:
            positive = positive[np.argsort(raw[positive])[::-1][:SAMPLE_TOKEN_TOP_K_FOLD]]
        selected = np.zeros(NUM_TOKENS_FOLD, dtype=np.float64)
        selected[positive] = np.power(raw[positive], SAMPLE_WEIGHT_POWER_FOLD)
        total = float(selected.sum())
        if total > 0:
            selected /= total
        weights[sample_idx] = selected.astype(np.float32)
    return weights


# Tokenize profiles for the current fold.
def _fold_tokenize_profiles(profiles, fit_idx):
    """Quantile-tokenize profiles using FIT samples only; apply bins to all rows."""
    profiles = np.asarray(profiles, dtype=np.float64)
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    if profiles.ndim != 2 or len(fit_idx) == 0:
        raise ValueError("profiles must be 2D and FIT must be non-empty")
    values = profiles[fit_idx].reshape(-1)
    edges = np.percentile(values, np.linspace(0.0, 100.0, NUM_TOKENS_FOLD + 1))
    if np.unique(edges).size < NUM_TOKENS_FOLD + 1:
        lo, hi = float(values.min()), float(values.max())
        edges = (np.linspace(lo - 1e-8, hi + 1e-8, NUM_TOKENS_FOLD + 1)
                 if np.isclose(lo, hi) else np.linspace(lo, hi, NUM_TOKENS_FOLD + 1))
    return np.clip(np.digitize(profiles, edges[1:-1]), 0, NUM_TOKENS_FOLD - 1).astype(np.int64)


# Build the fold-specific sample KNN graph.
def _build_fold_knn_from_profiles(g, fit_idx, k=10):

    if "profiles" not in g:
        raise KeyError("Graph lacks raw 'profiles'; rebuild it with GCN_Maker.py")

    X = np.asarray(g["profiles"], dtype=np.float64)
    fit_idx = np.asarray(fit_idx, dtype=np.int64).reshape(-1)
    patient_ids = np.asarray(g["patient_ids"]).astype(str).reshape(-1)

    if X.ndim != 2 or len(X) != len(patient_ids):
        raise ValueError("profiles and patient_ids are misaligned")
    if not np.isfinite(X).all():
        raise ValueError("profiles contain NaN or infinite values")
    if fit_idx.size == 0:
        raise ValueError("No FIT samples available for KNN")
    if np.any(fit_idx < 0) or np.any(fit_idx >= len(X)):
        raise IndexError("fit_idx contains an out-of-range sample index")
    if np.unique(fit_idx).size != fit_idx.size:
        raise ValueError("fit_idx contains duplicate sample indices")
    k = int(k)
    if k < 1:
        raise ValueError("KNN k must be >= 1")

    # Fit the scaler on FIT samples only; apply it to query rows and FIT refs.
    mu = X[fit_idx].mean(axis=0, keepdims=True)
    sd = X[fit_idx].std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(norms, 1e-8)
    sims = Z @ Z[fit_idx].T

    # A sample cannot use any subsample belonging to its own patient.
    same_patient = patient_ids[:, None] == patient_ids[fit_idx][None, :]
    sims[same_patient] = -np.inf

    rows, cols, vals = [], [], []
    for query_idx in range(len(X)):
        scores = sims[query_idx]
        valid_pos = np.flatnonzero(np.isfinite(scores) & (scores > 0.0))
        if valid_pos.size == 0:
            continue

        # Stable sorting makes tie handling deterministic. Select only after
        # all forbidden/invalid candidates have been removed.
        order = np.argsort(-scores[valid_pos], kind="stable")[:k]
        chosen_pos = valid_pos[order]
        chosen_fit_idx = fit_idx[chosen_pos]
        chosen_scores = scores[chosen_pos]

        rows.extend([query_idx] * len(chosen_pos))
        cols.extend(chosen_fit_idx.tolist())
        vals.extend(chosen_scores.tolist())

    A_knn = csr_matrix(
        (np.asarray(vals, dtype=np.float32),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(len(X), len(X)), dtype=np.float32
    )
    A_knn.sum_duplicates()
    A_knn.eliminate_zeros()

    # Defensive invariants: KNN may point only to FIT and never to same patient.
    if A_knn.nnz:
        coo = A_knn.tocoo()
        if not np.isin(coo.col, fit_idx).all():
            raise RuntimeError("KNN contains a neighbor outside FIT")
        if np.any(patient_ids[coo.row] == patient_ids[coo.col]):
            raise RuntimeError("KNN contains a same-patient edge")
        if np.any(coo.data <= 0) or not np.isfinite(coo.data).all():
            raise RuntimeError("KNN contains invalid similarity weights")

    return A_knn


# Construct graph adjacency blocks for the current fold.
def _build_fold_graph_blocks(g, fit_idx):
    """Rebuild sample-token, token-token and token-label blocks using FIT data only."""
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    if "profiles" not in g:
        raise KeyError("Graph lacks raw 'profiles'; cannot fit quantile bins independently inside each fold.")
    sequences = _fold_tokenize_profiles(g["profiles"], fit_idx)
    labels = np.asarray(g["labels"], dtype=np.int64)
    if len(sequences) != len(labels):
        raise ValueError("fold-tokenized profiles and graph['labels'] lengths differ.")
    if fit_idx.size == 0:
        raise ValueError("Cannot build graph statistics from an empty FIT set.")
    stats = _collect_fold_statistics(sequences, labels, fit_idx)
    sample_token = _fold_sample_token_weights(sequences, stats, len(fit_idx))
    token_token = _fold_ppmi(stats)
    token_label = _fold_token_label_ppmi(stats)
    n_all = len(labels)
    n_global = NUM_TOKENS_FOLD + 2

    rows, cols, vals = [], [], []
    for i in range(n_all):
        for t in np.flatnonzero(sample_token[i] > 0):
            rows.append(i); cols.append(int(t)); vals.append(float(sample_token[i, t]))
    A_sg = csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(n_all, n_global), dtype=np.float32)
    A_sg = row_normalize(A_sg)

    gr, gc, gv = [], [], []
    for node in range(n_global):
        gr.append(node); gc.append(node); gv.append(1.0)
    for i in range(NUM_TOKENS_FOLD):
        for j in range(i + 1, NUM_TOKENS_FOLD):
            if token_token[i, j] > 0:
                gr.extend((i, j)); gc.extend((j, i)); gv.extend((float(token_token[i, j]), float(token_token[i, j])))
        for c in range(2):
            if token_label[i, c] > 0:
                label_node = NUM_TOKENS_FOLD + c
                gr.extend((i, label_node)); gc.extend((label_node, i)); gv.extend((float(token_label[i, c]), float(token_label[i, c])))
    A_gg = csr_matrix((np.asarray(gv, dtype=np.float32), (gr, gc)), shape=(n_global, n_global), dtype=np.float32)
    A_gg.sum_duplicates()
    A_gg = row_normalize(A_gg)
    if np.any(np.asarray(A_sg.getnnz(axis=1)).ravel() == 0):
        raise ValueError("Fold-specific graph contains sample nodes without token edges.")
    return A_sg, A_gg, n_global


# Select the patient-level decision threshold.
def select_patient_threshold(
    meta=None,
    probabilities=None,
    method="fixed",
    target_sensitivity=0.95,
    patient_aggregation="mean",
    aggregation_alpha=0.25,
):
   
    if method != "fixed":
        raise ValueError(
            "Only the fixed threshold is supported: threshold=0.5 (DNN-aligned)."
        )
    return DECISION_THRESHOLD


# ----------------------------------------------------------------------------
# Fold context
# ----------------------------------------------------------------------------
# Standardize PMF profiles using fitting samples.
def _standardize_fold_pmf(profiles, fit_idx):
    """Standardize raw PMF using FIT rows only, then transform every graph row."""
    X = np.asarray(profiles, dtype=np.float32)
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 200:
        raise ValueError(f"Expected raw PMF profiles with shape [n, 200], got {X.shape}")
    if fit_idx.size == 0:
        raise ValueError("Cannot standardize PMF without FIT samples")
    mu = X[fit_idx].mean(axis=0, keepdims=True)
    sigma = X[fit_idx].std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return np.nan_to_num((X - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# Build model inputs and graph state for one fold.
def build_fold_context(k: int, g: dict, n_all: int,
                       y_train_only: np.ndarray, fold_ids_train: np.ndarray,
                       n_tr: int, A_pat, cfg: dict):

    fit_idx = np.flatnonzero(fold_ids_train != k)
    val_idx = np.flatnonzero(fold_ids_train == k)

    if len(fit_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Fold {k} has an empty FIT or Validation set "
            f"(FIT={len(fit_idx)}, VAL={len(val_idx)})."
        )

    A = {}
    pmf_x_np = _standardize_fold_pmf(g["profiles"], fit_idx)
    pmf_x = torch.from_numpy(pmf_x_np).to(DEVICE)
    if cfg["use_token_msg"]:
        A_sg_fold, A_gg_fold, _ = _build_fold_graph_blocks(g, fit_idx)
        A["sg"] = to_torch_sparse(A_sg_fold)
        A["gg"] = to_torch_sparse(A_gg_fold)

    fit_cols = np.zeros(n_all, dtype=bool)
    fit_cols[fit_idx] = True

    if cfg["use_knn"]:
        # Recompute KNN for this fold: FIT-only scaling and FIT-only neighbor columns.
        A_knn_fold = _build_fold_knn_from_profiles(g, fit_idx, k=int(cfg.get("knn_k", 10)))
        A["knn"] = to_torch_sparse(masked_row_normalize(A_knn_fold, fit_cols))
    if cfg["use_patient"]:
        A["pat"] = to_torch_sparse(masked_row_normalize(A_pat, fit_cols))

    y_fit = y_train_only[fit_idx]
    y_val = y_train_only[val_idx]

    if cfg["class_weights"]:
        w_fit = balanced_weights(y_fit)
    else:
        w_fit = np.ones_like(y_fit, dtype=np.float32)

    # Unweighted validation loss: class weights are estimated only from FIT.
    w_val = np.ones_like(y_val, dtype=np.float32)

    return {
        "n_samples": int(n_all),
        "A": A,
        "pmf_x": pmf_x,
        "fit_idx": torch.from_numpy(fit_idx).long().to(DEVICE),
        "val_idx": torch.from_numpy(val_idx).long().to(DEVICE),
        "fit_idx_np": fit_idx,
        "val_idx_np": val_idx,
        "y": torch.from_numpy(y_train_only.astype(np.float32)).to(DEVICE),
        "w_fit": torch.from_numpy(w_fit).to(DEVICE),
        "w_val": torch.from_numpy(w_val).to(DEVICE),
    }


# ----------------------------------------------------------------------------
# Train one instance
# ----------------------------------------------------------------------------
# Train one model instance and collect its predictions.
def train_instance(ctx, n_global, seed, cfg):
    """Train one fold-specific GCN and return probabilities, metrics and epoch history."""
    set_seed(seed)

    A, y = ctx["A"], ctx["y"]
    fit_idx, val_idx = ctx["fit_idx"], ctx["val_idx"]

    model = GraphNet(ctx["n_samples"], n_global, cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    epochs_run = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    time0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        epochs_run = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(A, ctx["pmf_x"])
        loss = weighted_bce(logits[fit_idx], y[fit_idx], ctx["w_fit"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        optimizer.step()

        history["train_loss"].append(float(loss.detach().cpu()))
        history["train_acc"].append(float(
            ((logits[fit_idx].detach() > 0).float() == y[fit_idx]).float().mean().cpu()
        ) * 100.0)

        # Evaluate validation metrics every epoch. Validation remains excluded from
        # gradient updates; checkpoint selection starts only at min_epochs.
        model.eval()
        with torch.no_grad():
            val_logits = model(A, ctx["pmf_x"])[val_idx]
            val_loss_t = weighted_bce(val_logits, y[val_idx], ctx["w_val"])
            val_loss = float(val_loss_t.cpu())
            val_acc = float(((val_logits > 0).float() == y[val_idx]).float().mean().cpu()) * 100.0
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch < cfg["min_epochs"]:
            continue

        if not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}.")

        if best_state is None or val_loss < best_val - cfg["min_delta"]:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint was saved. Check epochs/min_epochs and validation data."
        )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(A, ctx["pmf_x"])).cpu().numpy()
        

    info = {
        "ValLoss": np.round(best_val, 4),
        "TrLoss": np.round(history["train_loss"][best_epoch - 1], 4),
        "TrAcc": np.round(history["train_acc"][best_epoch - 1], 3),
        "TrEpochs": epochs_run,
        "BestEpoch": best_epoch,
        "TrTime": np.round(time.time() - time0, 2),
    }
    return probs, info, history,model


# ----------------------------------------------------------------------------
# Training-history plots and CSV exports
# ----------------------------------------------------------------------------
# Summarize curves with their median and interquartile range.
def _median_iqr_curve(curves):
    """Return epoch-wise median, Q1 and Q3; shorter histories are NaN padded."""
    curves = [np.asarray(curve, dtype=float).reshape(-1) for curve in curves if curve is not None and len(curve)]
    if not curves:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()
    arr = np.full((len(curves), max(map(len, curves))), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        arr[i, :len(curve)] = curve
    with np.errstate(all="ignore"):
        median, q1, q3 = np.nanpercentile(arr, [50, 25, 75], axis=0)
    return median, q1, q3


# Draw a training-history panel.
def _draw_history_panel(ax, curves, labels, title, ylabel):
    colors = {"train": "tab:blue", "val": "tab:orange"}
    for key, label in labels:
        median, q1, q3 = curves[key]
        if len(median):
            x = np.arange(1, len(median) + 1)
            color = colors["train" if key.startswith("train") else "val"]
            ax.plot(x, median, label=label, color=color)
            ax.fill_between(x, q1, q3, color=color, alpha=0.2, label=f"{label} IQR (Q1–Q3)")
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()


# Export curve data to CSV.
def _export_curve_csv(curves, out_path, names):
    max_len = max((len(curves[k][0]) for k in names), default=0)
    table = pd.DataFrame({"Epoch": np.arange(1, max_len + 1)})
    for key in names:
        med, q1, q3 = curves[key]
        for suffix, values in (("Median", med), ("Q1", q1), ("Q3", q3)):
            col = np.full(max_len, np.nan)
            col[:len(values)] = values
            table[f"{key}_{suffix}"] = col
    table.to_csv(out_path, index=False)


# Save the loss plot for a split.
def save_split_loss_plot(split_no, histories, out_dir):
    curves = {
        "train_loss": _median_iqr_curve([h["train_loss"] for h in histories]),
        "val_loss": _median_iqr_curve([h["val_loss"] for h in histories]),
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_history_panel(ax, curves, [("train_loss", "Training loss"), ("val_loss", "Validation loss")],
                        f"Split {split_no:02d}: loss across instances", "Weighted binary cross-entropy")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"Loss_Split_{split_no:02d}.png"), dpi=180)
    plt.close(fig)
    _export_curve_csv(curves, os.path.join(out_dir, f"Loss_Split_{split_no:02d}.csv"), ["train_loss", "val_loss"])
    return curves


# Save the accuracy plot for a split.
def save_split_acc_plot(split_no, histories, out_dir):
    curves = {
        "train_acc": _median_iqr_curve([h["train_acc"] for h in histories]),
        "val_acc": _median_iqr_curve([h["val_acc"] for h in histories]),
    }
    # Normalize keys for shared plotting helper, preserving explicit CSV column names.
    plot_curves = {"train": curves["train_acc"], "val": curves["val_acc"]}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_history_panel(ax, plot_curves, [("train", "Training accuracy"), ("val", "Validation accuracy")],
                        f"Split {split_no:02d}: accuracy across instances", "Accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"Accuracy_Split_{split_no:02d}.png"), dpi=180)
    plt.close(fig)
    _export_curve_csv(curves, os.path.join(out_dir, f"Accuracy_Split_{split_no:02d}.csv"), ["train_acc", "val_acc"])
    return curves


# Save the overall median loss plot.
def save_overall_median_loss_plot(split_curves, out_dir):
    curves = {key: _median_iqr_curve([item[key][0] for item in split_curves if len(item[key][0])])
              for key in ("train_loss", "val_loss")}
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_history_panel(ax, curves, [("train_loss", "Training loss"), ("val_loss", "Validation loss")],
                        "Overall median loss across splits", "Weighted binary cross-entropy")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Loss_Median_All_Splits.png"), dpi=200)
    plt.close(fig)
    _export_curve_csv(curves, os.path.join(out_dir, "Loss_Median_All_Splits.csv"), ["train_loss", "val_loss"])


# Save the overall median accuracy plot.
def save_overall_median_acc_plot(split_curves, out_dir):
    curves = {key: _median_iqr_curve([item[key][0] for item in split_curves if len(item[key][0])])
              for key in ("train_acc", "val_acc")}
    plot_curves = {"train": curves["train_acc"], "val": curves["val_acc"]}
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_history_panel(ax, plot_curves, [("train", "Training accuracy"), ("val", "Validation accuracy")],
                        "Overall median accuracy across splits", "Accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Accuracy_Median_All_Splits.png"), dpi=200)
    plt.close(fig)
    _export_curve_csv(curves, os.path.join(out_dir, "Accuracy_Median_All_Splits.csv"), ["train_acc", "val_acc"])


# ----------------------------------------------------------------------------
# Run one outer split
# ----------------------------------------------------------------------------

# Save diagnostic outputs for model errors.
def save_error_diagnostics(g, meta_te, test_probs, p_ens, thr, patient_table, res_dir, split_number):
    """Save post-hoc error-analysis tables; never used to train or tune the model."""
    os.makedirs(res_dir, exist_ok=True)
    n_test = len(meta_te)
    probs = np.asarray(test_probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != n_test:
        raise ValueError(f"Diagnostic probability shape {probs.shape} does not match n_test={n_test}.")

    sample = meta_te.copy().reset_index(drop=True)
    sample['Sid'] = sample['Sid'].astype(str)
    sample['ProbabilityMean'] = probs.mean(axis=0)
    sample['ProbabilitySD'] = probs.std(axis=0, ddof=1) if probs.shape[0] > 1 else 0.0
    sample['Threshold'] = float(thr)
    sample['SamplePrediction'] = (sample['ProbabilityMean'].to_numpy() > thr).astype(int)
    sample['SampleMargin'] = sample['ProbabilityMean'] - float(thr)
    sample['SampleCorrect'] = (sample['SamplePrediction'] == sample['Label']).astype(int)

    # Attach patient-level decision to each subsample row.
    pt = patient_table.copy()
    sample = sample.merge(
        pt[['Sid', 'PatientProbability', 'PatientPrediction', 'PatientErrorType', 'PatientMargin']],
        on='Sid', how='left', validate='many_to_one'
    )

    # Optional sequence-token composition, for descriptive comparison only.
    # This is not fed to the model and must not be used to tune on the held-out test set.
    seqs = g.get('sequences')
    if seqs is not None and len(seqs) == len(g['labels']):
        n_tokens = int(NUM_TOKENS_FOLD)
        sample_indices = np.arange(len(g['labels']))[np.asarray(g['test_mask'], dtype=bool)]
        if len(sample_indices) == len(sample):
            counts = np.zeros((len(sample), n_tokens), dtype=np.float64)
            for row_i, global_i in enumerate(sample_indices):
                seq = np.asarray(seqs[int(global_i)], dtype=np.int64).reshape(-1)
                seq = seq[(seq >= 0) & (seq < n_tokens)]
                if seq.size:
                    counts[row_i] = np.bincount(seq, minlength=n_tokens)[:n_tokens]
            totals = counts.sum(axis=1, keepdims=True)
            fracs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
            for t in range(n_tokens):
                sample[f'T{t:02d}_count'] = counts[:, t].astype(int)
                sample[f'T{t:02d}_fraction'] = fracs[:, t]

    split_tag = str(split_number).zfill(2)
    sample.to_csv(os.path.join(res_dir, f'TestSubsampleDiagnostics_{split_tag}.csv'), index=False)

    patient = patient_table.copy().reset_index(drop=True)
    patient['Threshold'] = float(thr)
    patient['PatientMargin'] = patient['PatientProbability'] - float(thr)
    patient['AbsoluteMargin'] = patient['PatientMargin'].abs()
    patient['NearThreshold_0_05'] = patient['AbsoluteMargin'] <= 0.05
    patient['NearThreshold_0_10'] = patient['AbsoluteMargin'] <= 0.10
    patient.to_csv(os.path.join(res_dir, f'TestPatientDiagnostics_{split_tag}.csv'), index=False)
    patient.loc[patient['PatientErrorType'] == 'FN_Cancer_missed'].to_csv(
        os.path.join(res_dir, f'FalseNegativePatients_{split_tag}.csv'), index=False
    )
    sample.loc[sample['PatientErrorType'] == 'FN_Cancer_missed'].to_csv(
        os.path.join(res_dir, f'FalseNegativeSubsamples_{split_tag}.csv'), index=False
    )

    # Summarize token fractions by patient-level outcome category, if available.
    token_cols = [c for c in sample.columns if c.startswith('T') and c.endswith('_fraction')]
    if token_cols:
        grouped = sample.groupby('PatientErrorType', dropna=False)[token_cols].mean().reset_index()
        grouped.to_csv(os.path.join(res_dir, f'TokenProfile_ByOutcome_{split_tag}.csv'), index=False)

    print(f"Error diagnostics saved: TestPatientDiagnostics_{split_tag}.csv; "
          f"FalseNegativePatients_{split_tag}.csv; TestSubsampleDiagnostics_{split_tag}.csv")

# Run cross-validation and testing for one data split.
def run_split(graph_path, cfg, resDir):
    g = load_graph(graph_path)
    v = int(g["split_number"])
    vers = str(v).zfill(2)

    labels = np.asarray(g["labels"], dtype=np.int64)
    train_mask = np.asarray(g["train_mask"], dtype=bool)
    test_mask = np.asarray(g["test_mask"], dtype=bool)
    n_all = labels.shape[0]

    if len(train_mask) != n_all or len(test_mask) != n_all:
        raise ValueError("Mask lengths must match the number of labels.")
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("TRAIN and TEST must both contain samples.")

    n_tr = int(train_mask.sum())
    if not (
        np.array_equal(np.flatnonzero(train_mask), np.arange(n_tr))
        and np.array_equal(np.flatnonzero(test_mask), np.arange(n_tr, n_all))
    ):
        raise ValueError(
            "Expected graph sample order: all TRAIN samples first, then TEST samples."
        )

    K = int(g["cv_folds"])
    fold_ids_train = np.asarray(g["fold_ids"])[:n_tr]
    validate_patient_splits(g, train_mask, test_mask, fold_ids_train, K)

    N = int(cfg["n_instances"])
    if N < K:
        raise ValueError(
            "n_instances must be >= cv_folds so every training sample receives OOF prediction."
        )

    if cfg["use_knn"] and g.get("profiles") is None and g.get("adj_sample_knn") is None:
        raise KeyError("Need raw profiles to build fold-specific KNN, or a prebuilt adj_sample_knn for legacy mode.")
    if cfg["use_patient"] and g.get("adj_sample_same_patient") is None:
        raise KeyError(
            "adj_sample_same_patient missing; rebuild graph or set use_patient=False."
        )

    validate_graph_inputs(g)
    if "profiles" not in g:
        raise KeyError("Graph pickle is missing raw 'profiles'; rebuild it with GCN_Maker_refactored.py")
    n_global = NUM_TOKENS_FOLD + 2
    A_pat = g.get("adj_sample_same_patient")

    
    if cfg["use_patient"]:
        A_pat = remove_diagonal(A_pat)
        if A_pat.shape != (n_all, n_all):
            raise ValueError(
                f"adj_sample_same_patient has invalid shape {A_pat.shape}."
            )

    # Test labels are retained only for evaluation metadata; training loss uses FIT labels only.
    y_train_only = labels.astype(np.float32).copy()
    y_train_only[n_tr:] = 0.0

    meta_tr = make_meta(g, np.arange(n_tr))
    meta_te = make_meta(g, np.arange(n_tr, n_all))

    contexts = {}
    oof_sum = np.zeros(n_tr, dtype=np.float64)
    oof_cnt = np.zeros(n_tr, dtype=np.int64)
    test_probs, infos, histories = [], [], []

    time00 = time.time()
    for inst in range(N):
        k = inst % K
        if k not in contexts:
            contexts[k] = build_fold_context(
                k, g, n_all, y_train_only, fold_ids_train, n_tr,
                A_pat, cfg
            )

        ctx = contexts[k]
        probs, info, history,last_model = train_instance(
            ctx, n_global, seed=SEED_BASE * v + inst, cfg=cfg
        )
        histories.append(history)

        val_np = ctx["val_idx_np"]
        oof_sum[val_np] += probs[val_np]
        oof_cnt[val_np] += 1
        test_probs.append(probs[n_tr:])
        infos.append(info)

        if inst % 5 == 0:
            print(
                "Split {} Inst {} (fold {}): epochs {} (best {}) "
                "ValLoss {} TrLoss {:.4f} TrAcc {:.2f}% time {} s".format(
                    v, inst + 1, k, info["TrEpochs"], info["BestEpoch"],
                    info["ValLoss"], info["TrLoss"], info["TrAcc"], info["TrTime"]
                )
            )

    test_probs = np.stack(test_probs, axis=0)
    with torch.no_grad():
        last_model.eval()
        pmf_x_tensor = ctx["pmf_x"]
        A_tensor = ctx["A"]
        
        s = last_model.pmf_encoder(pmf_x_tensor) + last_model.sample_init.unsqueeze(0)
        g_emb = last_model.g_emb
        for layer in last_model.layers:
            s, g_emb = layer(s, g_emb, A_tensor)
        
        test_labels = labels[n_tr:]
        raw_test_features = np.asarray(g["profiles"][n_tr:], dtype=np.float32)
        gcn_test_embeddings = s[n_tr:].cpu().numpy()
        
        out_npz_path = os.path.join(resDir, f"GCN_embeddings_split_{vers}.npz")
        np.savez_compressed(
            out_npz_path,
            y=test_labels,
            X_raw=raw_test_features,
            X_gcn=gcn_test_embeddings
        )
        print(f"-> [t-SNE] Successfully saved GCN embeddings to: {out_npz_path}")
    # ----------------------------------------------------
    # -------------------------------------------------------------
    # --------------------------------------------------------
    split_loss_curve = save_split_loss_plot(v, histories, resDir)
    split_acc_curve = save_split_acc_plot(v, histories, resDir)

    # Since each instance is assigned to exactly one fold, every training sample
    # should receive OOF predictions from instances for its own validation fold.
    if np.any(oof_cnt == 0):
        missing = np.flatnonzero(oof_cnt == 0)
        raise ValueError(
            f"Some training samples never received OOF predictions: {missing[:10].tolist()}"
        )
    # Every training sample must receive one validation prediction for each
    # instance assigned to its own fold. Check this unconditionally, including
    # when n_instances is not divisible by cv_folds.
    expected_by_fold = np.bincount(np.arange(N) % K, minlength=K)
    expected_oof_count = expected_by_fold[fold_ids_train]
    if not np.array_equal(oof_cnt, expected_oof_count):
        raise ValueError(
            "OOF prediction counts do not match instance/fold assignments: "
            f"observed={np.unique(oof_cnt).tolist()}, "
            f"expected={np.unique(expected_oof_count).tolist()}"
        )

    oof_probs = oof_sum / oof_cnt

    # Supplementary OOF aggregation diagnostics only. The primary reported
    # patient metrics remain DNN-aligned and always use the arithmetic mean;
    # these comparisons do not change the primary metric definitions.
    oof_comparison_rows = []
    # Compare predeclared alternative scores for transparency only.
    aggregation_candidates = [(name, 0.25) for name in ("mean", "top2_mean", "max")]
    aggregation_candidates += [("adaptive_mean_max", a) for a in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)]
    for aggregation_name, aggregation_alpha in aggregation_candidates:
        agg_oof = aggregate_patient_scores(meta_tr, oof_probs, aggregation_name, aggregation_alpha)
        agg_thr = select_patient_threshold(
            meta=meta_tr,
            probabilities=oof_probs,
            method=cfg.get("threshold_method", "fixed"),
            target_sensitivity=cfg.get("target_sensitivity", 0.95),
            patient_aggregation=aggregation_name,
            aggregation_alpha=aggregation_alpha,
        )
        agg_y = agg_oof["Label"].to_numpy(dtype=int)
        agg_score = agg_oof["PatientProbability"].to_numpy(dtype=float)
        agg_pred = (agg_score > agg_thr).astype(int)
        m = basic_metrics(agg_y, agg_pred, agg_score)
        oof_comparison_rows.append({
            "Aggregation": aggregation_name,
            "AggregationAlpha": aggregation_alpha if aggregation_name == "adaptive_mean_max" else np.nan,
            "ThresholdMethod": cfg.get("threshold_method", "fixed"),
            "OOFThreshold": float(agg_thr),
            "OOFPatientAccPct": m[0],
            "OOFPatientSpecificityPct": m[1],
            "OOFPatientSensitivityPct": m[2],
            "OOFPatientAUROC": m[3],
            "OOFPatientAUPRC": m[4],
            "OOFPatientBalancedAccuracyPct": 0.5 * (m[1] + m[2]),
        })
    os.makedirs(resDir, exist_ok=True)
    pd.DataFrame(oof_comparison_rows).to_csv(
        os.path.join(resDir, f"OOF_AggregationComparison_{vers}.csv"), index=False
    )

    # Use the pre-specified DNN-aligned operating point. OOF predictions are
    # retained for diagnostics and never used to tune the decision threshold.
    thr = select_patient_threshold(
        meta=meta_tr,
        probabilities=oof_probs,
        method=cfg.get("threshold_method", "fixed"),
        target_sensitivity=cfg.get("target_sensitivity", 0.95),
        patient_aggregation=cfg.get("patient_aggregation", "mean"),
        aggregation_alpha=cfg.get("aggregation_alpha", 0.25),
    )
    print(
        f"Split {v}: fixed patient threshold={thr:.4f} "
        f"(method={cfg.get('threshold_method', 'fixed')}, "
        f"aggregation={cfg.get('patient_aggregation', 'mean')}, "
        f"alpha={cfg.get('aggregation_alpha', 0.25):.2f}, "
        f"target_sensitivity={cfg.get('target_sensitivity', 0.95):.1%})"
    )

    perf_df = pd.DataFrame(
        index=np.arange(N), columns=METRIC_COLS + TRAIN_COLS, dtype=float
    )
    for inst in range(N):
        row = all_metrics(meta_te, test_probs[inst], thr, cfg.get("patient_aggregation", "mean"), cfg.get("aggregation_alpha", 0.25))
        row.update(infos[inst])
        perf_df.loc[inst, :] = pd.Series(row)

    # Ensemble probabilities are averaged once across stochastic instances.
    p_ens = test_probs.mean(axis=0)

    # Supplementary outer-test comparison of predeclared aggregation methods.
    # The primary reported metrics remain mean-based, matching DNN. Test
    # labels are used only for evaluation and never for model fitting/tuning.
    aggregation_methods = ("mean", "top2_mean", "max")
    primary_method = cfg.get("patient_aggregation", "mean")
    if primary_method not in aggregation_methods:
        raise ValueError(
            f"patient_aggregation must be one of {aggregation_methods}; got {primary_method!r}."
        )

    comparison_rows = []
    patient_tables = {}
    for method_name in aggregation_methods:
        method_thr = select_patient_threshold(
            meta=meta_tr,
            probabilities=oof_probs,
            method=cfg.get("threshold_method", "fixed"),
            target_sensitivity=cfg.get("target_sensitivity", 0.95),
            patient_aggregation=method_name,
            aggregation_alpha=cfg.get("aggregation_alpha", 0.25),
        )
        patient_table = aggregate_patient_scores(meta_te, p_ens, method_name)
        patient_y_method = patient_table["Label"].to_numpy(dtype=int)
        patient_scores_method = patient_table["PatientProbability"].to_numpy(dtype=float)
        patient_pred_method = (patient_scores_method > method_thr).astype(int)
        m = basic_metrics(patient_y_method, patient_pred_method, patient_scores_method)
        tp_m = int(np.sum((patient_y_method == 1) & (patient_pred_method == 1)))
        fn_m = int(np.sum((patient_y_method == 1) & (patient_pred_method == 0)))
        tn_m = int(np.sum((patient_y_method == 0) & (patient_pred_method == 0)))
        fp_m = int(np.sum((patient_y_method == 0) & (patient_pred_method == 1)))
        comparison_rows.append({
            "Split": v,
            "Aggregation": method_name,
            "ThresholdMethod": cfg.get("threshold_method", "fixed"),
            "ThresholdSource": "predefined fixed threshold 0.5 (DNN-aligned)",
            "PatientThreshold": float(method_thr),
            "PatientAccPct": m[0],
            "PatientSpecificityPct": m[1],
            "PatientSensitivityPct": m[2],
            "PatientAUROC": m[3],
            "PatientAUPRC": m[4],
            "PatientBalancedAccuracyPct": 0.5 * (m[1] + m[2]),
            "PatientTP": tp_m, "PatientFN": fn_m,
            "PatientTN": tn_m, "PatientFP": fp_m,
            "NTestPatients": int(len(patient_table)),
        })
        patient_table = patient_table.copy()
        patient_table["PatientAggregation"] = method_name
        patient_table["PatientThreshold"] = float(method_thr)
        patient_table["PatientPrediction"] = patient_pred_method
        patient_table["PatientMargin"] = patient_scores_method - float(method_thr)
        patient_table["PatientErrorType"] = np.select(
            [((patient_y_method == 1) & (patient_pred_method == 0)),
             ((patient_y_method == 0) & (patient_pred_method == 1)),
             ((patient_y_method == 1) & (patient_pred_method == 1)),
             ((patient_y_method == 0) & (patient_pred_method == 0))],
            ["FN_Cancer_missed", "FP_healthy_flagged", "TP_Cancer_detected", "TN_healthy_correct"],
            default="UNKNOWN",
        )
        patient_tables[method_name] = patient_table
        split_tag = str(v).zfill(2)
        patient_table.to_csv(
            os.path.join(resDir, f"TestPatients_{method_name}_{split_tag}.csv"), index=False
        )
        patient_table.loc[patient_table["PatientErrorType"] == "FN_Cancer_missed"].to_csv(
            os.path.join(resDir, f"FalseNegativePatients_{method_name}_{split_tag}.csv"), index=False
        )
        patient_table.loc[patient_table["PatientErrorType"] == "FP_healthy_flagged"].to_csv(
            os.path.join(resDir, f"FalsePositivePatients_{method_name}_{split_tag}.csv"), index=False
        )

    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(resDir, f"TestAggregationComparison_{vers}.csv"), index=False
    )

    # Keep arithmetic mean as the primary patient aggregation, exactly as DNN.
    primary_row = next(row for row in comparison_rows if row["Aggregation"] == "mean")
    thr = float(primary_row["PatientThreshold"])
    p_patient = patient_tables[primary_method].copy()
    tp = int(primary_row["PatientTP"])
    fn = int(primary_row["PatientFN"])
    tn = int(primary_row["PatientTN"])
    fp = int(primary_row["PatientFP"])
    sns_pct = float(primary_row["PatientSensitivityPct"])
    spc_pct = float(primary_row["PatientSpecificityPct"])
    print(
        f"TEST patient aggregation comparison | primary={primary_method} | "
        f"TP={tp}, FN={fn}, TN={tn}, FP={fp} | threshold={thr:.4f}"
    )
    print(f"Primary patient SNS={sns_pct:.2f}% | SPC={spc_pct:.2f}%")
    print(f"Ensemble probability range: {p_ens.min():.4f} .. {p_ens.max():.4f}")
    save_error_diagnostics(
        g=g, meta_te=meta_te, test_probs=test_probs, p_ens=p_ens, thr=thr,
        patient_table=p_patient, res_dir=resDir, split_number=v
    )

    ens_row = pd.Series(all_metrics(meta_te, p_ens, thr, cfg.get("patient_aggregation", "mean"), cfg.get("aggregation_alpha", 0.25)))
    # Persist explicit patient-level operating-point metrics for paper reporting.
    ens_row["PatientTP"] = tp
    ens_row["PatientFN"] = fn
    ens_row["PatientTN"] = tn
    ens_row["PatientFP"] = fp
    ens_row["PatientSensitivity"] = sns_pct / 100.0
    ens_row["PatientSpecificity"] = spc_pct / 100.0
    ens_row["PatientBalancedAccuracy"] = 0.5 * (sns_pct + spc_pct) / 100.0
    ens_row["PatientThreshold"] = thr

    print(
        "Split {}: {} instances in {} min | Ensemble: AucSub {:.4f}, "
        "AucAvg {:.4f}, SpcAvg {:.1f}, SnsAvg {:.1f} | "
        "OOF patient AUC {:.4f} | median best epoch {:.0f}".format(
            v, N, np.round((time.time() - time00) / 60, 2),
            ens_row["AucSub"], ens_row["AucAvg"], ens_row["SpcAvg"],
            ens_row["SnsAvg"], patient_auc(meta_tr, oof_probs, cfg.get("patient_aggregation", "mean"), cfg.get("aggregation_alpha", 0.25)),
            perf_df["BestEpoch"].median()
        )
    )

    out = {
        "perf_df": perf_df,
        "ens_perf": ens_row,
        "threshold": thr,
        "threshold_method": "fixed",
        "patient_aggregation": cfg.get("patient_aggregation", "mean"),
        "aggregation_alpha": cfg.get("aggregation_alpha", 0.25),
        "target_sensitivity": cfg.get("target_sensitivity", 0.95),
        "threshold_source": "predefined fixed threshold 0.5 (DNN-aligned)",
        "aggregation_comparison_methods": ["mean", "top2_mean", "max"],
        "cfg": cfg,
        "test_meta": meta_te,
        "test_probs": test_probs,
        "train_meta": meta_tr,
        "oof_probs": oof_probs,
        "oof_count": oof_cnt,
        "fold_ids": fold_ids_train,
        "graph_version": g.get("version"),
    }

    os.makedirs(resDir, exist_ok=True)
    out_path = os.path.join(resDir, f"{fResName}{vers}.pickle")
    with open(out_path, "wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ens_row, perf_df.mean(numeric_only=True), split_loss_curve, split_acc_curve


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
# Run the configured graph model experiment.
def main():
    run, cfg = parse_args(sys.argv)
    resDir = os.path.join(RES_BASE, run)
    os.makedirs(resDir, exist_ok=True)

    print("Run:", run, "| device:", DEVICE)
    print("Config:", cfg)

    with open(os.path.join(resDir, "config.pickle"), "wb") as handle:
        pickle.dump(cfg, handle)

    graph_files = sorted(
        f for f in os.listdir(cfg["graph_dir"])
        if f.startswith("graph_") and f.endswith(".pickle")
    )
    if not graph_files:
        raise FileNotFoundError(f"No graph_*.pickle found in {cfg['graph_dir']}")

    time000 = time.time()
    ens_rows, inst_rows = [], []
    split_loss_curves, split_acc_curves = [], []

    for fname in graph_files:
        print(f"\nLoading split graph: {fname}")
        ens_row, inst_mean, loss_curve, acc_curve = run_split(
            os.path.join(cfg["graph_dir"], fname), cfg, resDir
        )
        ens_rows.append(ens_row)
        inst_rows.append(inst_mean)
        split_loss_curves.append(loss_curve)
        split_acc_curves.append(acc_curve)

    save_overall_median_loss_plot(split_loss_curves, resDir)
    save_overall_median_acc_plot(split_acc_curves, resDir)

    # Combine per-split aggregation comparison files. Only match numeric split
    # filenames so a prior AllSplits output can never be ingested recursively.
    comparison_paths = []
    for name in os.listdir(resDir):
        if re.fullmatch(r"TestAggregationComparison_\d+\.csv", name):
            comparison_paths.append(os.path.join(resDir, name))
    comparison_paths.sort()
    if comparison_paths:
        comparison_all = pd.concat(
            [pd.read_csv(path) for path in comparison_paths], ignore_index=True
        )
        comparison_all.to_csv(
            os.path.join(resDir, "TestAggregationComparison_AllSplits.csv"), index=False
        )
        metric_cols = [
            "PatientAccPct", "PatientSpecificityPct", "PatientSensitivityPct",
            "PatientAUROC", "PatientAUPRC", "PatientBalancedAccuracyPct",
            "PatientFN", "PatientFP", "PatientThreshold",
        ]
        summary_rows = []
        for method_name, group in comparison_all.groupby("Aggregation", sort=False):
            for metric in metric_cols:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                q1 = float(values.quantile(0.25))
                med = float(values.median())
                q3 = float(values.quantile(0.75))
                summary_rows.append({
                    "Aggregation": method_name,
                    "Metric": metric,
                    "N_Splits": int(values.size),
                    "Median": med,
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": q3 - q1,
                    "Median [IQR]": f"{med:.4f} [{q1:.4f}, {q3:.4f}]",
                })
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(resDir, "Summary_TestAggregationComparison_Median_IQR.csv"),
            index=False,
        )

    ens_df = pd.DataFrame(ens_rows).reset_index(drop=True)
    ens_df.index = np.arange(1, len(ens_df) + 1)
    inst_df = pd.DataFrame(inst_rows).reset_index(drop=True)
    inst_df.index = np.arange(1, len(inst_df) + 1)

    sum_ens = pd.DataFrame({
        "median": ens_df.median(numeric_only=True),
        "mean": ens_df.mean(numeric_only=True),
    })
    sum_inst = pd.DataFrame({
        "median": inst_df.median(numeric_only=True),
        "mean": inst_df.mean(numeric_only=True),
    })

    ens_df.to_csv(os.path.join(resDir, "PerSplit_ensemble.csv"))
    inst_df.to_csv(os.path.join(resDir, "PerSplit_instanceMean.csv"))
    sum_ens.to_csv(os.path.join(resDir, "Summary_ensemble.csv"))
    sum_inst.to_csv(os.path.join(resDir, "Summary_instanceMean.csv"))

    pd.set_option("display.width", 220)
    print("\n=== Per-split metrics (ensemble) ===")
    print(ens_df.round(3).to_string())
    print(f"\n=== Summary over {len(ens_df)} splits: ensemble ===")
    print(sum_ens.round(3).to_string())
    print(f"\n=== Summary over {len(inst_df)} splits: mean of single instances ===")
    print(sum_inst.round(3).to_string())
    print(
        "\nRun {} completed in {} hrs".format(
            run, np.round((time.time() - time000) / 3600, 3)
        )
    )

if __name__ == "__main__":
    main()
