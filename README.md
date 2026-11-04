# scDeepSim

This repository contains the scDeepSim implementation and the experiments for
evaluating controllable single-cell simulation. scDeepSim combines a structured
variational autoencoder with latent diffusion to generate expression profiles
and vary batch effects and trajectory structure for method benchmarking.

## Repository contents

| Directory | Contents |
|---|---|
| [`scdeepsim/src/scdeepsim/`](scdeepsim/src/scdeepsim/) | Python package: VAE and diffusion models, data handling, and simulation controls. |
| [`scdeepsim/tests/`](scdeepsim/tests/) | Package and experiment tests. |
| [`experiments/scripts/`](experiments/scripts/) | Training, simulation, evaluation, and plotting entry points. |
| [`experiments/configs/`](experiments/configs/) | Hydra configurations with data paths and experiment settings. |
| [`experiments/src/`](experiments/src/) | Shared preprocessing, comparator, metric, and benchmark utilities. |
| [`experiments/notebooks/`](experiments/notebooks/) | Exploratory analyses and figures from saved results. |
| [`experiments/environments/`](experiments/environments/) | Environment definitions for running the experiments. |

## Reproduce the paper experiments

Start with the **[experiment guide](experiments/README.md)**. It provides
environment setup, dataset sources, commands, and output locations for the
four main experiments and the [additional experiments](experiments/README.md#additional-experiments):

| Experiment | Workflow |
|---|---|
| Simulation fidelity (Figure 2) | `simulation_fidelity.py` with the lung, pancreas, or immune config |
| Structured latent representations and batch interventions (Figure 3) | `latent_predictability.py` → `eval_batch_dose_response.py` |
| Batch integration (Figure 4) | `benchmark_batch_integration.py` |
| Trajectory inference (Figure 5) | `prepare_ti_artifacts.py` → `prepare_ti_direction_artifacts.py` → `benchmark_ti.py` with the direction config |

Scripts are under `experiments/scripts/`; trajectory scripts are in its
`ti_benchmarking/` directory. Formal runs use the settings in
`experiments/configs/`. The scDiffusion comparator requires the
[public upstream source and experiment patch](experiments/README.md#scdiffusion-setup).

Datasets, trained scDeepSim weights, and published-run sample archives are not
bundled; follow the guide to obtain the inputs and generate the results.

## Package installation

With Python 3.10 or newer, install from the repository root:

```bash
python -m pip install -e ./scdeepsim
```

## License and attribution

Released under the [MIT License](LICENSE). Parts of the diffusion implementation
were inspired by or adapted from
[lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
(MIT); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
