"""Materialize simulation-fidelity selection provenance without fitting a method."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import socket

from hydra import compose, initialize_config_dir
import pyrootutils

root = Path(
    pyrootutils.setup_root(
        __file__,
        indicator=(".git", ".project-root"),
        pythonpath=True,
        dotenv=True,
    )
)

from experiments.src.simulation_fidelity.data import (  # noqa: E402
    load_and_preprocess,
    train_test_split_adata,
)


def parse_args() -> argparse.Namespace:
    """Parse a minimal, explicit selection-probe command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--data-checksum", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the formal preprocessing and save its exact cells, genes, and split."""
    args = parse_args()
    config_dir = root / "experiments" / "configs"
    overrides = [
        f"paths.root_dir={root}",
        f"paths.data_path={args.data_path.resolve()}",
        f"data.checksum={args.data_checksum}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    adata_norm, adata_raw = load_and_preprocess(cfg)
    train_norm, eval_norm, _, _, split = train_test_split_adata(
        adata_norm,
        adata_raw,
        cfg,
    )
    payload = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "software": {
            package: version(package)
            for package in ("anndata", "numpy", "scanpy", "scipy", "scikit-learn")
        },
        "seed": int(cfg.seed),
        "data_selection": dict(adata_raw.uns["simulation_fidelity_input"]),
        "data_shape": {
            "selected_n_cells": int(adata_raw.n_obs),
            "n_genes": int(adata_raw.n_vars),
            "train_n_cells": int(train_norm.n_obs),
            "eval_n_cells": int(eval_norm.n_obs),
        },
        "split": split,
        "selected_var_names": adata_raw.var_names.astype(str).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
