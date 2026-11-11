"""Load the fixed course snapshots offline, with integrity and schema checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def manifest(root: Path = ROOT) -> dict:
    return json.loads((Path(root) / "data_manifest.json").read_text())


def load_dataset(name: str, root: Path = ROOT) -> pd.DataFrame:
    """Return one dataset; missing observations remain missing, never imputed.

    Paths are relative to this module, not the working directory. Unknown IDs,
    absent/corrupt files, and unexpected schemas are errors, not empty results.
    """
    root = Path(root)
    spec = manifest(root)["datasets"][name]
    path = root / spec["file"]
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != spec["sha256"]:
        raise ValueError(f"{name}: checksum mismatch; extract a fresh course packet")
    frame = pd.read_csv(path, index_col=0)
    if spec["index_kind"] == "datetime":
        frame.index = pd.to_datetime(frame.index, errors="raise")
    else:
        frame.index = pd.Index(frame.index, dtype="int64", name="sample")
    if frame.empty or len(frame) != spec["rows"]:
        raise ValueError(f"{name}: unexpected row count")
    if list(frame.columns) != spec["columns"]:
        raise ValueError(f"{name}: unexpected columns")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name}: timestamps/sample indices must be unique and sorted")
    if str(frame.index[0]) != spec["start"] or str(frame.index[-1]) != spec["end"]:
        raise ValueError(f"{name}: unexpected coverage")
    if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
        raise ValueError(f"{name}: nonnumeric data")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{name}: infinite values")
    if frame.isna().sum().to_dict() != spec["missing"]:
        raise ValueError(f"{name}: unexpected missing-value counts")
    if spec.get("frequency"):
        expected = pd.date_range(frame.index[0], frame.index[-1], freq=spec["frequency"])
        if not frame.index.equals(expected):
            raise ValueError(f"{name}: incomplete time grid")
    frame.attrs["capstone"] = spec
    return frame


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fixed 60/20/20 time blocks. Split before imputing/scaling/windowing.

    For horizon h, training examples must have their entire target inside the
    training block; a validation/test origin is scored only when its entire
    target is inside that block. Missing targets are not imputed for scoring.
    """
    if len(frame) < 10 or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Supply at least ten ordered, unique time points")
    a, b = int(0.6 * len(frame)), int(0.8 * len(frame))
    return frame.iloc[:a].copy(), frame.iloc[a:b].copy(), frame.iloc[b:].copy()


def load_pems_graph(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return locations in snapshot column order and edges for those sensors."""
    frame = load_dataset("pems_bay", root)
    for spec in frame.attrs["capstone"]["auxiliary"].values():
        path = Path(root) / spec["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("PEMS auxiliary file checksum mismatch")
    locations = pd.read_csv(Path(root) / "snapshots/sensor_locations_bay.csv", header=None,
                            names=["sensor", "latitude", "longitude"], dtype={"sensor": str}).set_index("sensor")
    edges = pd.read_csv(Path(root) / "snapshots/distances_bay.csv", dtype={"from": str, "to": str})
    nodes = list(frame.columns)
    return locations.loc[nodes], edges.loc[edges["from"].isin(nodes) & edges["to"].isin(nodes)].copy()


def validate_all(root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for name, spec in manifest(root)["datasets"].items():
        frame = load_dataset(name, root)
        train, validation, test = chronological_split(frame)
        target = spec["target"]
        for label, block in [("train", train), ("validation", validation), ("test", test)]:
            if block[target].notna().sum() < 10:
                raise ValueError(f"{name}: insufficient observed targets in {label}")
        rows.append({"dataset": name, "rows": len(frame), "columns": len(frame.columns),
                     "missing_cells": int(frame.isna().sum().sum()),
                     "train_end": str(train.index[-1]),
                     "validation_end": str(validation.index[-1]), "status": "PASS"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(validate_all().to_string(index=False))
