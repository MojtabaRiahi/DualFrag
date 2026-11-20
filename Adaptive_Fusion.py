#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adaptive Fusion: DNN + GCN.

Retains only these methods:
    dnn
    gcn
    avg_prob
    avg_logit
    guarded_residual_sample
    guarded_residual_patient

DNN is the anchor model. GCN contributes a bounded correction:
    z_fused = z_dnn + alpha * (z_gcn - z_dnn)

For guarded residual fusion:
    alpha = gate_max_weight * learned_gate * confidence_guard

"""

import os
import sys
import ast
import pickle
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import wilcoxon


GROUPS = ["Sub", "Maj", "Con", "Avg"]
DECISION_THRESHOLD = 0.5
MAJORITY_THRESHOLD = 0.6
METRICS = ["Acc", "Spc", "Sns", "F1", "Auc", "AucPR"]
METRIC_COLS = [m + g for g in GROUPS for m in METRICS]

DEFAULT_CFG = dict(
    dnn_dir="./data/Test_Results/DWT_DNN/DWT",
    gcn_dir="./data/Test_Results/GCN_Graph/GCN",
    out_base="./data/Test_Results/Adaptive_Fusion/",
    main="avg_prob",
    gate_l2=0.005,
    gate_max_weight=0.4,
    confidence_temperature=0.9,
    thr=0.5,
)


METHODS = [
    "dnn",
    "gcn",
    "avg_prob",
    "avg_logit",
    "guarded_residual_sample",
    "guarded_residual_patient",
]


# Implement parse args.
def parse_args(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    run = argv[1]
    cfg = dict(DEFAULT_CFG)

    for kv in argv[2:]:
        if "=" not in kv:
            raise ValueError(f"Override must be key=value: {kv}")

        k, v = kv.split("=", 1)

        if k not in cfg:
            raise ValueError(
                f"Unknown config key '{k}'. Valid keys: {sorted(cfg)}"
            )

        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v

    if cfg["main"] not in METHODS:
        raise ValueError(f"main must be one of {METHODS}")

    if not np.isclose(float(cfg["thr"]), 0.5):
        raise ValueError(
            "Subsample and Average probability thresholds are fixed at 0.5 "
            "to match the final DNN and GCN branches. Majority uses its "
            "separate fixed support threshold of 0.6."
        )
    cfg["thr"] = 0.5

    if float(cfg["gate_l2"]) < 0:
        raise ValueError("gate_l2 must be non-negative.")

    if not 0 < float(cfg["gate_max_weight"]) <= 1:
        raise ValueError("gate_max_weight must be in (0, 1].")

    if float(cfg["confidence_temperature"]) <= 0:
        raise ValueError("confidence_temperature must be positive.")

    return run, cfg


# Implement sigmoid.
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)

    return out


# Implement logit.
def logit(p):
    p = np.asarray(p, dtype=float)

    if not np.all(np.isfinite(p)):
        raise ValueError("Probabilities contain NaN or infinity.")

    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# Implement balanced weights.
def balanced_weights(y):
    y = np.asarray(y, dtype=int)

    n = len(y)
    n1 = float(y.sum())
    n0 = n - n1

    return np.where(
        y == 1,
        n / (2 * max(n1, 1.0)),
        n / (2 * max(n0, 1.0)),
    )


# Implement roc auc.
def roc_auc(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(y) != len(scores):
        return np.nan

    if not np.all(np.isfinite(scores)):
        return np.nan

    P = int(y.sum())
    N = len(y) - P

    if P == 0 or N == 0:
        return np.nan

    ranks = pd.Series(scores).rank(method="average").values

    return float(
        (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)
    )


# Implement pr auc.
def pr_auc(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(y) != len(scores):
        return np.nan

    if not np.all(np.isfinite(scores)):
        return np.nan

    P = int(y.sum())

    if P == 0:
        return np.nan

    order = np.argsort(-scores, kind="mergesort")
    scores, y = scores[order], y[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    last = np.r_[
        np.where(np.diff(scores) != 0)[0],
        len(scores) - 1,
    ]

    precision = tp[last] / np.maximum(tp[last] + fp[last], 1)
    recall = tp[last] / P

    return float(
        np.sum(np.diff(np.r_[0.0, recall]) * precision)
    )


# Implement basic metrics.
def basic_metrics(y, pred, score=None):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    if len(y) == 0 or len(y) != len(pred):
        raise ValueError("Invalid or inconsistent metric inputs.")

    P = int((y == 1).sum())
    N = int((y == 0).sum())

    acc = (pred == y).mean() * 100
    tn = int(((pred == 0) & (y == 0)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    spc = tn / max(N, 1) * 100
    sns = tp / max(P, 1) * 100
    # F1 is reported as a percentage, consistently with Acc/Spc/Sns.
    f1_denom = 2 * tp + fp + fn
    f1 = (2 * tp / f1_denom * 100) if f1_denom else 0.0

    if score is None:
        auc = auc_pr = np.nan
    else:
        score = np.asarray(score, dtype=float)

        if len(score) != len(y):
            raise ValueError("Score and label lengths differ.")

        auc = roc_auc(y, score)
        auc_pr = pr_auc(y, score)

    values = (acc, spc, sns, f1, auc, auc_pr)

    return [
        round(float(v), 3) if np.isfinite(v) else np.nan
        for v in values[:4]
    ] + [
        round(float(v), 5) if np.isfinite(v) else np.nan
        for v in values[4:]
    ]


# Implement validate meta.
def validate_meta(meta, name):
    required = {"FCid", "Sid", "Subid", "ExSid", "Label"}
    missing = required - set(meta.columns)

    if missing:
        raise KeyError(f"{name}: missing columns {sorted(missing)}")

    if meta[list(required)].isnull().any().any():
        raise ValueError(
            f"{name}: missing values in required metadata columns."
        )

    labels = set(
        pd.to_numeric(meta["Label"], errors="raise")
        .astype(int)
        .unique()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(f"{name}: Label must be binary 0/1.")

    if (meta.groupby("Sid")["Label"].nunique() > 1).any():
        raise ValueError(
            f"{name}: a patient has inconsistent labels across subsamples."
        )

# Implement all metrics.
def all_metrics(meta, probabilities, thr):
    validate_meta(meta, "all_metrics")
    if not np.isclose(float(thr), DECISION_THRESHOLD):
        raise ValueError(
            f"Subsample and Average thresholds are fixed at {DECISION_THRESHOLD}."
        )

    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) != len(meta):
        raise ValueError("Probability vector shape does not match metadata.")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities in all_metrics.")

    y = meta["Label"].astype(int).values
    # Same strict probability cutoff as the final DNN and GCN branches.
    pred = (p > thr).astype(int)

    d = pd.DataFrame({
        "Sid": meta["Sid"].values,
        "Label": y,
        "p": p,
        "pos": pred,
    })


    g = d.groupby("Sid", sort=False).agg(
        Label=("Label", "first"),
        frac_pos=("pos", "mean"),
        mean_p=("p", "mean"),
        all_pos=("pos", lambda x: int(np.all(x == 1))),
        all_neg=("pos", lambda x: int(np.all(x == 0))),
    )


    results = basic_metrics(y, pred, p)


    maj_pred = (g.frac_pos.values >= MAJORITY_THRESHOLD).astype(int)
    results += basic_metrics(g.Label.values, maj_pred, g.frac_pos.values)



    con_pred = np.where(
        g.all_pos.values == 1,
        1,
        np.where(g.all_neg.values == 1, 0, 1 - g.Label.values)
    )
    results += basic_metrics(g.Label.values, con_pred, con_pred.astype(float))


    avg_pred = (g.mean_p.values > thr).astype(int)
    results += basic_metrics(g.Label.values, avg_pred, g.mean_p.values)

    return dict(zip(METRIC_COLS, results))

# Implement patient auc.
def patient_auc(sid, y, p):
    d = pd.DataFrame({
        "Sid": sid,
        "Label": np.asarray(y, dtype=int),
        "p": np.asarray(p, dtype=float),
    })

    g = d.groupby("Sid", sort=False).agg(
        Label=("Label", "first"),
        mean_p=("p", "mean"),
    )

    return roc_auc(g.Label.values, g.mean_p.values)

# Implement sample keys.
def sample_keys(meta):
    return (
        meta["FCid"].astype(str)
        + "|"
        + meta["Sid"].astype(str)
        + "|"
        + meta["Subid"].astype(str)
        + "|"
        + meta["ExSid"].astype(str)
    ).values


# Implement align to ref.
def align_to_ref(meta_ref, meta_other, name):
    kr = sample_keys(meta_ref)
    ko = sample_keys(meta_other)

    if len(set(kr)) != len(kr):
        raise ValueError(
            f"{name}: duplicate sample keys in reference metadata."
        )

    if len(set(ko)) != len(ko):
        raise ValueError(
            f"{name}: duplicate sample keys in other metadata."
        )

    if len(kr) != len(ko):
        raise ValueError(
            f"{name}: different sample counts ({len(kr)} vs {len(ko)})."
        )

    positions = pd.Series(np.arange(len(ko)), index=ko)
    idx = positions.reindex(kr)

    if idx.isna().any():
        raise ValueError(
            f"{name}: {int(idx.isna().sum())} samples could not be aligned."
        )

    return idx.values.astype(int)


# Implement load pickle.
def load_pickle(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} file not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(
            f"{name} pickle must contain a dictionary."
        )

    return obj


# Implement prediction vector.
def prediction_vector(values, n, name):
    a = np.asarray(values, dtype=float)

    if a.ndim == 2:
        if a.shape[1] == n:
            a = a.mean(axis=0)
        elif a.shape[0] == n:
            a = a.mean(axis=1)
        else:
            raise ValueError(
                f"{name}: incompatible shape {a.shape}."
            )
    elif a.ndim != 1:
        raise ValueError(f"{name}: expected 1D or 2D array.")

    if len(a) != n:
        raise ValueError(
            f"{name}: expected {n} values, got {len(a)}."
        )

    if (
        not np.all(np.isfinite(a))
        or np.any((a < 0) | (a > 1))
    ):
        raise ValueError(f"{name}: invalid probability values.")

    return a


# Implement load split.
def load_split(v, cfg):
    version = str(v).zfill(2)

    d1 = load_pickle(
        os.path.join(
            cfg["dnn_dir"],
            f"TestPerf_df{version}.pickle",
        ),
        "DNN",
    )

    d2 = load_pickle(
        os.path.join(
            cfg["gcn_dir"],
            f"TestPerf_df{version}.pickle",
        ),
        "GCN",
    )

    required = (
        "test_meta",
        "test_probs",
        "train_meta",
        "oof_probs",
        "fold_ids",
    )

    for name, obj in (("DNN", d1), ("GCN", d2)):
        for key in required:
            if obj.get(key) is None:
                raise KeyError(
                    f"{name} split {version}: missing key {key!r}."
                )

    meta_te = d1["test_meta"].reset_index(drop=True).copy()
    meta_tr = d1["train_meta"].reset_index(drop=True).copy()

    graph_te = d2["test_meta"].reset_index(drop=True).copy()
    graph_tr = d2["train_meta"].reset_index(drop=True).copy()

    for meta, name in (
        (meta_te, "DNN test"),
        (meta_tr, "DNN train"),
        (graph_te, "GCN test"),
        (graph_tr, "GCN train"),
    ):
        validate_meta(meta, name)

    # The official split is patient-level: no patient may occur in both sets.
    train_patients = set(meta_tr["Sid"].astype(str))
    test_patients = set(meta_te["Sid"].astype(str))
    patient_overlap = train_patients & test_patients
    if patient_overlap:
        preview = sorted(patient_overlap)[:10]
        raise ValueError(
            f"Split {version}: Train/Test patient leakage detected; "
            f"overlap={preview}"
        )

    # Verify the saved branch operating points before combining predictions.
    for branch_name, obj in (("DNN", d1), ("GCN", d2)):
        branch_thr = obj.get("threshold", 0.5)
        if not np.isfinite(float(branch_thr)) or not np.isclose(float(branch_thr), 0.5):
            raise ValueError(
                f"Split {version}: {branch_name} threshold is {branch_thr}; "
                "expected the fixed 0.5 operating point."
            )

    i_te = align_to_ref(
        meta_te, graph_te, f"split {version} test"
    )
    i_tr = align_to_ref(
        meta_tr, graph_tr, f"split {version} train"
    )

    if not np.array_equal(
        meta_te.Label.astype(int).values,
        graph_te.Label.astype(int).values[i_te],
    ):
        raise ValueError(
            f"Split {version}: test labels differ between lines."
        )

    if not np.array_equal(
        meta_tr.Label.astype(int).values,
        graph_tr.Label.astype(int).values[i_tr],
    ):
        raise ValueError(
            f"Split {version}: training labels differ between lines."
        )

    folds1 = np.asarray(d1["fold_ids"])
    folds2 = np.asarray(d2["fold_ids"])

    if folds1.ndim != 1 or len(folds1) != len(meta_tr):
        raise ValueError(
            f"Split {version}: invalid DNN fold_ids length."
        )

    if folds2.ndim != 1 or len(folds2) != len(graph_tr):
        raise ValueError(
            f"Split {version}: invalid GCN fold_ids length."
        )

    if not np.array_equal(folds1, folds2[i_tr]):
        raise ValueError(
            f"Split {version}: fold assignments differ between lines."
        )

    fold_df = pd.DataFrame({
        "Sid": meta_tr.Sid.astype(str).values,
        "fold": folds1,
    })

    if (fold_df.groupby("Sid").fold.nunique() > 1).any():
        raise ValueError(
            f"Split {version}: a patient occurs in multiple folds."
        )

    p1_te = np.asarray(d1["test_probs"], dtype=float)
    p2_te = np.asarray(d2["test_probs"], dtype=float)

    if p1_te.ndim == 1:
        p1_te = p1_te[None, :]

    if p2_te.ndim == 1:
        p2_te = p2_te[None, :]

    if p1_te.ndim != 2 or p1_te.shape[1] != len(meta_te):
        raise ValueError(
            f"Split {version}: invalid DNN test_probs shape {p1_te.shape}."
        )

    if p2_te.ndim != 2 or p2_te.shape[1] != len(graph_te):
        raise ValueError(
            f"Split {version}: invalid GCN test_probs shape {p2_te.shape}."
        )

    p1_te = p1_te.mean(axis=0)
    p2_te = p2_te.mean(axis=0)[i_te]

    for p in (p1_te, p2_te):
        if (
            not np.all(np.isfinite(p))
            or np.any((p < 0) | (p > 1))
        ):
            raise ValueError(
                f"Split {version}: invalid test probabilities."
            )

    p1_oof = prediction_vector(
        d1["oof_probs"],
        len(meta_tr),
        f"DNN split {version} oof_probs",
    )

    p2_oof = prediction_vector(
        d2["oof_probs"],
        len(graph_tr),
        f"GCN split {version} oof_probs",
    )[i_tr]

    return {
        "meta_te": meta_te,
        "meta_tr": meta_tr,
        "z1_te": logit(p1_te),
        "z2_te": logit(p2_te),
        "z1_oof": logit(p1_oof),
        "z2_oof": logit(p2_oof),
        "y_tr": meta_tr.Label.astype(int).values,
        "folds": folds1.astype(int),
    }


# Implement weighted logloss.
def weighted_logloss(z, y, weights):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=int)
    weights = np.asarray(weights, dtype=float)

    if not np.all(np.isfinite(z)):
        return np.inf

    return float(
        np.sum(
            weights * np.logaddexp(0.0, -(2 * y - 1) * z)
        ) / np.sum(weights)
    )


# Implement checked minimize.
def checked_minimize(objective, x0, name, bounds=None):
    result = minimize(
        objective,
        x0=np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if (
        not np.all(np.isfinite(result.x))
        or not np.isfinite(result.fun)
    ):
        raise RuntimeError(
            f"{name} optimisation failed: {result.message}; "
            f"objective={result.fun}"
        )

    if not result.success:
        warnings.warn(
            f"{name} optimiser stopped with message: {result.message}"
        )

    return result.x


class Fixed:
# Implement   init  .
    def __init__(self, fn):
        self.fn = fn

# Implement fit.
    def fit(self, z1, z2, y, pid):
        return self

# Implement predict.
    def predict(self, z1, z2, pid):
        return self.fn(z1, z2)



# Implement gate inputs.
def gate_inputs(z1, z2, pid, level):
    abs1 = np.abs(z1)
    abs2 = np.abs(z2)
    agree = (np.sign(z1) == np.sign(z2)).astype(float)

    if level == "patient":
        frame = pd.DataFrame({
            "pid": pid,
            "a1": abs1,
            "a2": abs2,
            "agree": agree,
        })

        means = frame.groupby("pid", sort=False)[
            ["a1", "a2", "agree"]
        ].transform("mean")

        abs1 = means.a1.values
        abs2 = means.a2.values
        agree = means.agree.values

    return abs1 - abs2, agree



class AdaptiveResidual:
    """DNN-anchored gate: z = z_DNN + alpha * (z_GCN - z_DNN)."""

# Implement   init  .
    def __init__(self, level, l2, max_weight):
        self.level = level
        self.l2 = float(l2)
        self.max_weight = float(max_weight)

# Implement  alpha.
    def _alpha(self, params, z1, z2, pid):
        confidence, agree = gate_inputs(
            z1, z2, pid, self.level
        )

        return self.max_weight * sigmoid(
            params[0]
            + params[1] * confidence / self.confidence_scale
            + params[2] * agree
        )

# Implement  forward.
    def _forward(self, params, z1, z2, pid):
        alpha = self._alpha(params, z1, z2, pid)
        return z1 + alpha * (z2 - z1)

# Implement fit.
    def fit(self, z1, z2, y, pid):
        weights = balanced_weights(y)
        confidence, _ = gate_inputs(
            z1, z2, pid, self.level
        )

        self.confidence_scale = float(np.std(confidence)) + 1e-6

# Compute the optimization objective.
        def objective(params):
            alpha = self._alpha(params, z1, z2, pid)
            fused = z1 + alpha * (z2 - z1)

            correction_penalty = self.l2 * float(
                np.mean(alpha ** 2)
            )
            parameter_penalty = 0.01 * self.l2 * float(
                np.sum(params ** 2)
            )

            return (
                weighted_logloss(fused, y, weights)
                + correction_penalty
                + parameter_penalty
            )

        self.params = checked_minimize(
            objective,
            [-2.0, 0.0, 0.0],
            f"adaptive_residual_{self.level}",
        )

        return self

# Implement predict.
    def predict(self, z1, z2, pid):
        result = self._forward(self.params, z1, z2, pid)

        if not np.all(np.isfinite(result)):
            raise ValueError(
                f"adaptive_residual_{self.level} returned "
                "non-finite logits."
            )

        return result

# Implement describe.
    def describe(self):
        return {
            "gate_bias": float(self.params[0]),
            "gate_conf": float(self.params[1]),
            "gate_agree": float(self.params[2]),
            "gate_max_weight": self.max_weight,
            "confidence_scale": self.confidence_scale,
        }


class GuardedResidual(AdaptiveResidual):
    """
    Confidence-guarded residual fusion.
    The guard reduces GCN influence if its absolute logit is lower than
    DNN's. This is a hypothesis to evaluate, not a guaranteed improvement.
    Disagreement increases residual weight when DNN and GCN logits differ and attenuates it when they agree.
    """
# Implement   init  .
    def __init__(
        self,
        level,
        l2,
        max_weight,
        temperature=1.0,
    ):
        super().__init__(level, l2, max_weight)

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.temperature = float(temperature)

# Implement fit.
    def fit(self, z1, z2, y, pid):
        weights = balanced_weights(y)

        confidence, _ = gate_inputs(
            z1, z2, pid, self.level
        )

        self.confidence_scale = float(
            np.std(confidence)
        ) + 1e-6

        # Calculate disagreement scale using TRAIN OOF only
        disagreement = self._get_disagreement(z1, z2, pid)

        self.disagree_scale = float(
            np.median(disagreement)
        ) + 1e-6

# Compute the optimization objective.
        def objective(params):
            alpha = self._alpha(params, z1, z2, pid)

            fused = z1 + alpha * (z2 - z1)

            correction_penalty = self.l2 * float(
                np.mean(alpha ** 2)
            )

            parameter_penalty = 0.01 * self.l2 * float(
                np.sum(params ** 2)
            )

            return (
                weighted_logloss(fused, y, weights)
                + correction_penalty
                + parameter_penalty
            )

        self.params = checked_minimize(
            objective,
            [-2.0, 0.0, 0.0],
            f"guarded_residual_{self.level}",
        )

        return self

# Implement  get disagreement.
    def _get_disagreement(self, z1, z2, pid):
        if self.level == "patient":
            frame = pd.DataFrame({
                "pid": np.asarray(pid),
                "diff": np.abs(z1 - z2),
            })

            return frame.groupby("pid", sort=False)["diff"].transform(
                "mean"
            ).values
        return np.abs(z1 - z2)

# Implement  alpha.
    def _alpha(self, params, z1, z2, pid):
        confidence, agree = gate_inputs(
            z1, z2, pid, self.level
        )

        learned_gate = sigmoid(
            params[0]
            + params[1] * confidence / self.confidence_scale
            + params[2] * agree
        )

        if self.level == "patient":
            frame = pd.DataFrame({
                "pid": np.asarray(pid),
                "abs_dnn": np.abs(z1),
                "abs_gcn": np.abs(z2),
            })

            means = frame.groupby("pid", sort=False)[
                ["abs_dnn", "abs_gcn"]
            ].transform("mean")

            abs_dnn = means["abs_dnn"].values
            abs_gcn = means["abs_gcn"].values
        else:
            abs_dnn = np.abs(z1)
            abs_gcn = np.abs(z2)

        confidence_guard = sigmoid(
            (abs_gcn - abs_dnn) / self.temperature
        )

        if self.level == "patient":
            frame_disagree = pd.DataFrame({
                "pid": np.asarray(pid),
                "diff": np.abs(z1 - z2),
            })
            disagree = frame_disagree.groupby("pid", sort=False)["diff"].transform("mean").values
        else:
            disagree = np.abs(z1 - z2)

        disagree_gate = sigmoid(
            (disagree - self.disagree_scale) / self.temperature
        )

        disagreement_factor = 0.55 + 0.45 * disagree_gate

        alpha = (
            self.max_weight
            * learned_gate
            * confidence_guard
            * disagreement_factor
        )

        return np.clip(alpha, 0.0, self.max_weight)

# Implement describe.
    def describe(self):
        result = super().describe()
        result.update({
            "fusion_type": "confidence_guarded_residual",
            "confidence_temperature": self.temperature,
            "disagreement_factor": "0.55+0.45*sigmoid((|zdnn-zgcn|-median)/temperature)",
        })
        return result


# Implement make fuser.
def make_fuser(name, cfg):
    if name == "dnn":
        return Fixed(lambda z1, z2: z1)

    if name == "gcn":
        return Fixed(lambda z1, z2: z2)

    if name == "avg_prob":
        return Fixed(
            lambda z1, z2: logit(
                0.5 * (sigmoid(z1) + sigmoid(z2))
            )
        )

    if name == "avg_logit":
        return Fixed(lambda z1, z2: 0.5 * (z1 + z2))


    if name == "guarded_residual_sample":
        return GuardedResidual(
            "sample",
            cfg["gate_l2"],
            cfg["gate_max_weight"],
            cfg["confidence_temperature"],
        )

    if name == "guarded_residual_patient":
        return GuardedResidual(
            "patient",
            cfg["gate_l2"],
            cfg["gate_max_weight"],
            cfg["confidence_temperature"],
        )

    raise ValueError(f"Unknown fusion method: {name}")
# Implement oof cv logits.
def oof_cv_logits(name, cfg, z1, z2, y, pid, folds):
    """
    Diagnostic CV only.

    These diagnostics are not an unbiased estimate of fusion selection
    when base-model OOF predictions were generated before this CV.
    """
    output = np.full(len(y), np.nan, dtype=float)

    for fold in np.unique(folds):
        train = folds != fold
        valid = folds == fold

        if not train.any() or not valid.any():
            continue

        if len(np.unique(y[train])) < 2:
            warnings.warn(
                f"Fold {fold}: meta-training data contains one class; "
                "skipping diagnostic fold."
            )
            continue

        model = make_fuser(name, cfg).fit(
            z1[train],
            z2[train],
            y[train],
            pid[train],
        )

        output[valid] = model.predict(
            z1[valid],
            z2[valid],
            pid[valid],
        )

    return output


# Implement run split.
def run_split(version, cfg, result_dir):
    D = load_split(version, cfg)

    sid_train = D["meta_tr"].Sid.astype(str).values
    sid_test = D["meta_te"].Sid.astype(str).values

    rows = []
    test_predictions = {}
    gate_params = {}

    for name in METHODS:
        # Diagnostic OOF-CV: not used to fit the final test predictions.
        zcv = oof_cv_logits(
            name,
            cfg,
            D["z1_oof"],
            D["z2_oof"],
            D["y_tr"],
            sid_train,
            D["folds"],
        )

        valid = np.isfinite(zcv)

        if valid.any():
            diagnostic_auc_sub = roc_auc(
                D["y_tr"][valid],
                zcv[valid],
            )

            diagnostic_auc_patient = patient_auc(
                sid_train[valid],
                D["y_tr"][valid],
                sigmoid(zcv[valid]),
            )
        else:
            diagnostic_auc_sub = np.nan
            diagnostic_auc_patient = np.nan

        # Fit the fusion method using training OOF predictions only.
        model = make_fuser(name, cfg).fit(
            D["z1_oof"],
            D["z2_oof"],
            D["y_tr"],
            sid_train,
        )

        # Evaluate on the held-out test split.
        test_logits = model.predict(
            D["z1_te"],
            D["z2_te"],
            sid_test,
        )

        test_p = sigmoid(test_logits)

        if not np.all(np.isfinite(test_p)):
            raise ValueError(
                f"Split {version}, {name}: invalid test probabilities."
            )

        test_predictions[name] = test_p

        if hasattr(model, "describe"):
            gate_params[name] = model.describe()

        row = {
            "split": version,
            "method": name,
            "OofAucSub_DIAGNOSTIC": diagnostic_auc_sub,
            "OofAucAvg_DIAGNOSTIC": diagnostic_auc_patient,
        }

        row.update(
            all_metrics(
                D["meta_te"],
                test_p,
                cfg["thr"],
            )
        )

        rows.append(row)

    # Create patient-level probability table.
    base = pd.DataFrame({
        "Sid": sid_test,
        "Label": D["meta_te"].Label.astype(int).values,
    })

    for name in METHODS:
        base["p_" + name] = test_predictions[name]

    patient_table = (
        base.groupby("Sid", sort=False)
        .agg(
            Label=("Label", "first"),
            **{
                "p_" + name: ("p_" + name, "mean")
                for name in METHODS
            },
        )
        .reset_index()
    )

    patient_table.insert(0, "split", version)

    output_path = os.path.join(
        result_dir,
        f"TestPerf_df{str(version).zfill(2)}.pickle",
    )

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "test_meta": D["meta_te"],
                "test_probs_by_method": test_predictions,
                "gate_params": gate_params,
                "cfg": cfg,
                "note": (
                    ""
                ),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    by_method = {r["method"]: r for r in rows}

    diagnostic_scores = {
        r["method"]: r["OofAucAvg_DIAGNOSTIC"]
        for r in rows
        if np.isfinite(r["OofAucAvg_DIAGNOSTIC"])
    }

    diagnostic_best = (
        max(diagnostic_scores, key=diagnostic_scores.get)
        if diagnostic_scores
        else "unavailable"
    )

    print(
        f"Split {version}: diagnostic OOF-CV best={diagnostic_best} "
        f"(not for confirmatory selection) | "
        f"test patient AUC: "
        f"DNN={by_method['dnn']['AucAvg']:.4f}, "
        f"GCN={by_method['gcn']['AucAvg']:.4f}, "
        f"{cfg['main']}={by_method[cfg['main']]['AucAvg']:.4f}"
    )

    return rows, patient_table


# Implement wilcoxon table.
def wilcoxon_table(results, main_method, metrics):
    output = []

    pivots = {
        m: results.pivot(
            index="split",
            columns="method",
            values=m,
        )
        for m in metrics
    }

    for other in METHODS:
        if other == main_method:
            continue

        for metric in metrics:
            pivot = pivots[metric]

            if main_method not in pivot or other not in pivot:
                continue

            pair = pivot[[main_method, other]].dropna()

            diff = (
                pair[main_method] - pair[other]
            ).values

            if len(diff) == 0 or np.allclose(diff, 0):
                pval = np.nan
            else:
                try:
                    pval = wilcoxon(
                        diff,
                        zero_method="wilcox",
                    ).pvalue
                except ValueError:
                    pval = np.nan

            output.append({
                "main": main_method,
                "other": other,
                "metric": metric,
                "n_paired_splits": len(diff),
                "median_main": (
                    np.median(pair[main_method])
                    if len(diff) else np.nan
                ),
                "median_other": (
                    np.median(pair[other])
                    if len(diff) else np.nan
                ),
                "median_diff": (
                    np.median(diff)
                    if len(diff) else np.nan
                ),
                "main_better_splits": int((diff > 0).sum()),
                "other_better_splits": int((diff < 0).sum()),
                "p_wilcoxon": pval,
            })

    return pd.DataFrame(output)


# Implement error reports.
def error_reports(patient_table, main_method, threshold):
    pt = patient_table.copy()

    for name in METHODS:
        pt["wrong_" + name] = (
            (pt["p_" + name] > threshold).astype(int)
            != pt.Label
        ).astype(int)

    pt["err_dnn"] = (pt.Label - pt.p_dnn).abs()
    pt["err_gcn"] = (pt.Label - pt.p_gcn).abs()

    agg = pt.groupby("Sid").agg(
        Label=("Label", "first"),
        times_in_test=("split", "size"),
        **{
            "wrong_" + name: ("wrong_" + name, "sum")
            for name in METHODS
        },
    )

    agg = agg.sort_values(
        ["wrong_dnn", "wrong_gcn"],
        ascending=False,
    )

    dnn_wrong = pt.wrong_dnn.values.astype(bool)
    gcn_wrong = pt.wrong_gcn.values.astype(bool)
    main_wrong = pt["wrong_" + main_method].values.astype(bool)

    both = int((dnn_wrong & gcn_wrong).sum())
    either = int((dnn_wrong | gcn_wrong).sum())

    corr = np.corrcoef(pt.err_dnn, pt.err_gcn)[0, 1]

    if not np.isfinite(corr):
        corr = np.nan

    report = "\n".join([
        f"patient-instances evaluated      : {len(pt)}",
        (
            "DNN wrong / GCN wrong / both      : "
            f"{dnn_wrong.sum()} / {gcn_wrong.sum()} / {both}"
        ),
        (
            "GCN right where DNN is wrong      : "
            f"{(dnn_wrong & ~gcn_wrong).sum()} of {dnn_wrong.sum()}"
        ),
        (
            "DNN right where GCN is wrong      : "
            f"{(gcn_wrong & ~dnn_wrong).sum()} of {gcn_wrong.sum()}"
        ),
        (
            "Jaccard overlap of error sets     : "
            f"{both / max(either, 1):.3f}"
        ),
        (
            "corr(|label - p|) DNN vs GCN      : "
            f"{corr:.3f}"
        ),
        (
            f"{main_method} wrong                : "
            f"{main_wrong.sum()} "
            f"(fixes {(dnn_wrong & ~main_wrong).sum()} DNN errors, "
            f"introduces {(~dnn_wrong & main_wrong).sum()} new errors)"
        ),
    ])

    return agg, report


# Parse arguments and run the configured fusion workflow.
def main():
    run, cfg = parse_args(sys.argv)

    result_dir = os.path.join(cfg["out_base"], run)
    os.makedirs(result_dir, exist_ok=True)

    print("Run:", run)
    print("Config:", cfg)

    for key in ("dnn_dir", "gcn_dir"):
        if not os.path.isdir(cfg[key]):
            raise FileNotFoundError(
                f"{key} directory not found: {cfg[key]}"
            )

# Discover available input splits.
    def available_splits(directory):
        found = set()

        for filename in os.listdir(directory):
            if (
                filename.startswith("TestPerf_df")
                and filename.endswith(".pickle")
            ):
                number = filename[
                    len("TestPerf_df"):-len(".pickle")
                ]

                if number.isdigit():
                    found.add(int(number))

        return found

    expected_splits = set(range(1, 21))
    dnn_splits = available_splits(cfg["dnn_dir"])
    gcn_splits = available_splits(cfg["gcn_dir"])

    if dnn_splits != expected_splits:
        raise FileNotFoundError(
            "DNN folder must contain exactly the 20 official splits 01-20; "
            f"missing={sorted(expected_splits - dnn_splits)}, "
            f"unexpected={sorted(dnn_splits - expected_splits)}"
        )
    if gcn_splits != expected_splits:
        raise FileNotFoundError(
            "GCN folder must contain exactly the 20 official splits 01-20; "
            f"missing={sorted(expected_splits - gcn_splits)}, "
            f"unexpected={sorted(gcn_splits - expected_splits)}"
        )
    if dnn_splits != gcn_splits:
        raise RuntimeError("DNN and GCN split identifiers do not match.")

    splits = sorted(expected_splits)
    print("Official splits validated in both lines:", len(splits))

    all_rows = []
    all_patient_tables = []

    for version in splits:
        rows, pt = run_split(version, cfg, result_dir)
        all_rows.extend(rows)
        all_patient_tables.append(pt)

    results = pd.DataFrame(all_rows)

    patient_table = pd.concat(
        all_patient_tables,
        ignore_index=True,
    )

    results.to_csv(
        os.path.join(result_dir, "PerSplit_all_methods.csv"),
        index=False,
    )

    patient_table.to_csv(
        os.path.join(result_dir, "PatientTable.csv"),
        index=False,
    )

    value_columns = (
        ["OofAucSub_DIAGNOSTIC", "OofAucAvg_DIAGNOSTIC"]
        + METRIC_COLS
    )

    median_summary = (
        results.groupby("method")[value_columns]
        .median()
        .reindex(METHODS)
    )

    mean_summary = (
        results.groupby("method")[value_columns]
        .mean()
        .reindex(METHODS)
    )

    median_summary.to_csv(
        os.path.join(result_dir, "Summary_median.csv")
    )

    mean_summary.to_csv(
        os.path.join(result_dir, "Summary_mean.csv")
    )

    mean_summary[
        ["OofAucSub_DIAGNOSTIC", "OofAucAvg_DIAGNOSTIC"]
    ].to_csv(
        os.path.join(result_dir, "OOF_cv_diagnostic_summary.csv")
    )

    key_metrics = [
        "AccSub",
        "SpcSub",
        "SnsSub",
        "F1Sub",
        "AucSub",
        "AccAvg",
        "SpcAvg",
        "SnsAvg",
        "F1Avg",
        "AucAvg",
        "AucPRAvg",
        "AccMaj",
        "F1Maj",
        "AucMaj",
    ]

    pd.set_option("display.width", 240)

    print("\n=== Median across test splits ===")
    print(median_summary[key_metrics].round(3).to_string())

    print("\n=== Mean across test splits ===")
    print(mean_summary[key_metrics].round(3).to_string())

    print("\n=== OOF-CV diagnostic only ===")
    print(
        mean_summary[
            ["OofAucSub_DIAGNOSTIC", "OofAucAvg_DIAGNOSTIC"]
        ].round(4).to_string()
    )

    print("\nPre-specified main method:", cfg["main"])

    wr = wilcoxon_table(
        results,
        cfg["main"],
        [
            "AccSub",
            "F1Sub",
            "AucSub",
            "AccAvg",
            "F1Avg",
            "AucAvg",
            "SnsAvg",
            "SpcAvg",
        ],
    )

    wr.to_csv(
        os.path.join(result_dir, "Wilcoxon_main_vs_others.csv"),
        index=False,
    )

    print("\n=== Paired Wilcoxon comparisons ===")

    if not wr.empty:
        print(
            wr[wr.metric.isin(["AucAvg", "AccAvg", "F1Avg"])][
                [
                    "other",
                    "metric",
                    "median_main",
                    "median_other",
                    "median_diff",
                    "main_better_splits",
                    "other_better_splits",
                    "p_wilcoxon",
                ]
            ].round(4).to_string(index=False)
        )

    error_table, report = error_reports(
        patient_table,
        cfg["main"],
        cfg["thr"],
    )

    error_table.to_csv(
        os.path.join(result_dir, "PatientErrors.csv")
    )

    with open(
        os.path.join(result_dir, "ErrorComplementarity.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report + "\n")

    print("\n=== Complementarity of DNN and GCN ===")
    print(report)

    print("\nMost frequently misclassified patients:")
    print(error_table.head(12).to_string())

    print("\nResults written to:", result_dir)


if __name__ == "__main__":
    main()
