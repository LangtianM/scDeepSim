"""Optional Slingshot R adapter."""

from __future__ import annotations

from ._r_adapter import run_r_adapter


def run_slingshot(adata, *, output_dir, random_state: int = 0, **kwargs):
    """Run Slingshot through the experiment-local R adapter."""
    return run_r_adapter(
        adata,
        method="slingshot",
        script_name="run_slingshot.R",
        output_dir=output_dir,
        random_state=random_state,
        use_conda_run=bool(kwargs.get("r_use_conda_run", False)),
        conda_env=str(kwargs.get("r_conda_env", "lightning")),
        keep_inputs=bool(kwargs.get("keep_adapter_inputs", False)),
    )
