"""Prepare the Waddington-OT expression matrix for Figure 3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd

from .data import select_count_matrix

WOT_PERIOD_DAYS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_id_mapping(path: Path, value_name: str) -> pd.Series:
    """Read a two-column whitespace-delimited mapping with unique IDs."""
    table = pd.read_csv(path, sep=r"\s+")
    if table.shape[1] != 2:
        raise ValueError(f"Expected two columns in {path}, found {table.shape[1]}.")
    table.columns = ["id", value_name]
    table["id"] = table["id"].astype(str)
    if table["id"].duplicated().any():
        duplicated = table.loc[table["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"Duplicate cell ID {duplicated!r} in {path}.")
    return table.set_index("id")[value_name]


def format_period(day: float) -> str:
    """Format a numeric WOT day as the paper-facing period label."""
    return f"D{day:g}"


def prepare_wot_adata(
    adata: ad.AnnData,
    cell_days: pd.Series,
    batches: pd.Series,
    *,
    allowed_days: Iterable[float] = WOT_PERIOD_DAYS,
) -> ad.AnnData:
    """Join WOT metadata, filter periods, and preserve validated counts."""
    if not adata.obs_names.is_unique:
        raise ValueError("WOT expression matrix contains duplicate observation IDs.")
    if not cell_days.index.is_unique or not batches.index.is_unique:
        raise ValueError("WOT day and batch mappings must use unique cell IDs.")

    select_count_matrix(adata, "counts")
    obs_ids = adata.obs_names.astype(str)
    day_values = pd.to_numeric(cell_days.reindex(obs_ids), errors="coerce")
    allowed = np.asarray(tuple(float(day) for day in allowed_days))
    keep = day_values.notna().to_numpy() & np.isclose(
        day_values.fillna(np.inf).to_numpy()[:, None],
        allowed[None, :],
        rtol=0.0,
        atol=1e-8,
    ).any(axis=1)
    if not keep.any():
        raise ValueError("No WOT cells matched the configured periods.")

    selected_ids = obs_ids[keep]
    selected_batches = batches.reindex(selected_ids)
    if selected_batches.isna().any():
        missing = selected_ids[selected_batches.isna().to_numpy()][0]
        raise ValueError(f"Selected WOT cell {missing!r} has no batch annotation.")

    result = adata[keep].copy()
    selected_days = day_values.loc[selected_ids].to_numpy(dtype=float)
    result.obs["period"] = [format_period(day) for day in selected_days]
    result.obs["batch"] = selected_batches.astype(str).to_numpy()
    result.layers["counts"] = result.X.copy()
    result.uns["wot_preprocessing"] = {
        "allowed_periods": [format_period(day) for day in allowed],
        "n_source_cells": int(adata.n_obs),
        "n_selected_cells": int(result.n_obs),
    }
    return result


def prepare_wot_files(
    expression_path: Path,
    cell_days_path: Path,
    batches_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Prepare and write the CHTC-ready WOT H5AD file."""
    adata = ad.read_h5ad(expression_path)
    cell_days = read_id_mapping(cell_days_path, "day")
    batches = read_id_mapping(batches_path, "batch")
    result = prepare_wot_adata(adata, cell_days, batches)
    result.uns["wot_preprocessing"].update(
        {
            "expression_sha256": sha256_file(expression_path),
            "cell_days_sha256": sha256_file(cell_days_path),
            "batches_sha256": sha256_file(batches_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="gzip")
    return {
        **dict(result.uns["wot_preprocessing"]),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
