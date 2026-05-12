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
    )
