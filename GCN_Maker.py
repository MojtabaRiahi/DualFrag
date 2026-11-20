#!/usr/bin/env python3
"""Prepare leakage-safe split data for the fold-specific GCN model.

This module loads raw PMF profiles and split metadata only. Fold-specific
 feature processing is delegated to GCN_Model.
"""
from __future__ import annotations

import re

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

CV_FOLDS = 5
PROFILE_SIZE = 200


# Handle extract lengths 51 to 250.
def extract_lengths_51_to_250(values: Sequence[float]) -> np.ndarray:
    """Return the PMF values for fragment lengths 51..250."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == PROFILE_SIZE:
        selected = arr
    elif arr.size >= 250:
        selected = arr[50:250]
    else:
        raise ValueError(f"Cannot extract lengths 51..250 from profile length {arr.size}.")
    if not np.isfinite(selected).all():
        raise ValueError("Profile contains NaN or infinite values.")
    return selected.astype(np.float32, copy=False)


# Handle resolve data path.
def resolve_data_path(value: str, data_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    candidate = data_root / candidate.name
    if candidate.is_file():
        return candidate
    matches = list(data_root.rglob(candidate.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate profile {value!r} under {data_root}.")


# Handle load profile.
def load_profile(value: str, data_root: Path) -> np.ndarray:
    with resolve_data_path(value, data_root).open("rb") as handle:
        return extract_lengths_51_to_250(pickle.load(handle))


# Handle parse path.
def parse_path(value: str) -> tuple[str, str, str, str]:
    """Extract FC id, patient id, subsample id, and experiment suffix."""
    parts = str(value).replace("\\", "/").split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse sample identifiers from {value!r}.")
    patient = parts[1].split("/")[-1]
    fc = parts[1].split("/")[0]
    if len(parts) == 3:
        experiment, subsample = "", parts[2].split(".")[0]
    else:
        experiment = "_" + "_".join(parts[2:-1])
        subsample = parts[-1].split(".")[0]
    return fc, patient, subsample, experiment


# Handle sample table.
def sample_table(paths: Sequence[str], labels: Sequence[int]) -> pd.DataFrame:
    rows = []
    for path, label in zip(paths, labels):
        fc, patient, sub, experiment = parse_path(path)
        rows.append((fc, patient, sub, experiment, int(label)))
    return pd.DataFrame(rows, columns=["FCid", "Sid", "Subid", "ExSid", "Label"])


# Handle patient folds.
def patient_folds(train_meta: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    """Assign whole patients to stratified folds; return one fold per sample."""
    rng = np.random.RandomState(seed)
    patient_labels = train_meta.groupby("Sid")["Label"].max()
    assignment = {}
    for label in (0, 1):
        ids = np.asarray(sorted(patient_labels.index[patient_labels.values == label]))
        rng.shuffle(ids)
        for i, patient in enumerate(ids):
            assignment[patient] = i % folds
    return train_meta["Sid"].map(assignment).to_numpy(dtype=np.int64)


# Handle validate split.
def validate_split(train_meta: pd.DataFrame, test_meta: pd.DataFrame, folds: int) -> None:
    for name, frame in (("TRAIN", train_meta), ("TEST", test_meta)):
        mixed = frame.groupby("Sid")["Label"].nunique()
        if (mixed > 1).any():
            raise ValueError(f"Patients have mixed labels in {name}: {mixed[mixed > 1].index[:10].tolist()}")
    overlap = set(train_meta.Sid) & set(test_meta.Sid)
    if overlap:
        raise ValueError(f"TRAIN/TEST patient overlap: {sorted(overlap)[:10]}")
    counts = train_meta.groupby("Sid")["Label"].max().value_counts()
    if any(int(counts.get(c, 0)) < folds for c in (0, 1)):
        raise ValueError(f"Each class needs at least {folds} training patients for CV.")


# Handle process split.
def process_split(split_path: Path, data_root: Path, output_dir: Path,
                  cv_folds: int = CV_FOLDS) -> Path:
    with split_path.open("rb") as handle:
        train_paths, train_labels, test_paths, test_labels = pickle.load(handle)
    train_paths, test_paths = list(train_paths), list(test_paths)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    if len(train_paths) != len(train_labels) or len(test_paths) != len(test_labels):
        raise ValueError(f"Paths and labels are misaligned in {split_path}.")
    if not np.isin(np.r_[train_labels, test_labels], [0, 1]).all():
        raise ValueError("Labels must be binary (0=healthy, 1=cancer).")

    train_meta = sample_table(train_paths, train_labels)
    test_meta = sample_table(test_paths, test_labels)
    validate_split(train_meta, test_meta, cv_folds)
    fold_train = patient_folds(train_meta, cv_folds, seed=int(re.search(r"(\d+)$", split_path.stem).group(1)))
    all_paths = train_paths + test_paths
    profiles = np.stack([load_profile(p, data_root) for p in all_paths]).astype(np.float32)
    n_train, n_all = len(train_paths), len(all_paths)
    train_mask = np.arange(n_all) < n_train
    fold_ids = np.full(n_all, -1, dtype=np.int64)
    fold_ids[:n_train] = fold_train
    metadata = pd.concat([train_meta, test_meta], ignore_index=True)
    result = {
        "version": "RawProfiles_SplitData_v1",
        "split_name": split_path.stem,
        "split_number": int(re.search(r"(\d+)$", split_path.stem).group(1)),
        "train_mask": train_mask,
        "test_mask": ~train_mask,
        "labels": np.r_[train_labels, test_labels],
        "file_names": [Path(p).name for p in all_paths],
        "patient_ids": metadata["Sid"].to_numpy(dtype=object),
        "fc_ids": metadata["FCid"].to_numpy(dtype=object),
        "subsample_ids": metadata["Subid"].to_numpy(dtype=object),
        "ex_ids": metadata["ExSid"].to_numpy(dtype=object),
        "fold_ids": fold_ids,
        "cv_folds": int(cv_folds),
        "profiles": profiles,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"graph_{split_path.stem}.pickle"
    with destination.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved raw split data: {destination} (train={n_train}, test={n_all-n_train})")
    return destination


# Handle run all splits.
def run_all_splits(splits_dir="./data/TrainTestPaths", data_dir="./data",
                   output_dir="./output_graph", cv_folds=CV_FOLDS) -> None:
    files = sorted(Path(splits_dir).glob("*.pickle"))
    if not files:
        raise FileNotFoundError(f"No split pickle files found in {splits_dir!r}.")
    for split in files:
        process_split(split, Path(data_dir), Path(output_dir), cv_folds)


if __name__ == "__main__":
    run_all_splits()
