"""Optional Monocle3 R adapter for trajectory-inference benchmarks."""

from __future__ import annotations

from ._r_adapter import run_r_adapter


def run_monocle3(adata, *, output_dir, random_state: int = 0, **kwargs):
    """Run Monocle3 through the experiment-local R adapter.

    Keyword options are forwarded as adapter controls: ``r_use_conda_run``,
    ``r_conda_env``, and ``keep_adapter_inputs``.
    """
    return run_r_adapter(
        adata,
        method="monocle3",
        script_name="run_monocle3.R",
        output_dir=output_dir,
        random_state=random_state,
        use_conda_run=bool(kwargs.get("r_use_conda_run", False)),
        conda_env=str(kwargs.get("r_conda_env", "lightning")),
        keep_inputs=bool(kwargs.get("keep_adapter_inputs", False)),
    )
