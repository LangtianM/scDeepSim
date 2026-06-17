import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_refactored_experiment_scripts_import():
    modules = [
        "experiments.scripts.branch_direction_knob",
        "experiments.scripts.check_batch_latent_gaussianity",
        "experiments.scripts.eval_batch_disentanglement",
        "experiments.scripts.eval_batch_dose_response",
        "experiments.scripts.eval_branch_point_tau",
        "experiments.scripts.eval_heldout_batch_validation",
        "experiments.scripts.eval_pt_dose_response",
        "experiments.scripts.eval_simulation_quality_scdesign3",
        "experiments.scripts.figure2_latent_disentanglement",
        "experiments.scripts.interpolate_batch_effect",
        "experiments.scripts.interpolate_trajectory",
        "experiments.scripts.ti_benchmarking.benchmark_ti",
        "experiments.scripts.train_vae_diffusion",
    ]

    for module in modules:
        importlib.import_module(module)


def test_experiment_utils_facade_exports_moved_helpers():
    utils = importlib.import_module("experiments.src.utils")

    for name in [
        "as_dense",
        "decode_latents",
        "encode_adata",
        "load_and_preprocess",
        "load_pancreas",
        "save_git_info",
        "train_celltype_batch_vae",
    ]:
        assert hasattr(utils, name)
