"""Prepare the frozen full VAE+diffusion artifact bundle for TI benchmarking.

Example
-------
python experiments/scripts/ti_benchmarking/prepare_ti_artifacts.py \
    --config-path ../../configs --config-name prepare_ti_artifacts
"""

import os
import tempfile

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.src.ti_artifacts import prepare_ti_artifacts


@hydra.main(
    config_path="../../configs",
    config_name="prepare_ti_artifacts",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    bundle = prepare_ti_artifacts(cfg, cfg.paths.artifact_dir)
    print(f"Validated TI artifact bundle: {bundle.root}")
    print(f"artifact_hash={bundle.artifact_hash}")


if __name__ == "__main__":
    main()

