"""Optional Monocle3 R adapter."""

from __future__ import annotations

from ._r_adapter import run_r_adapter


def run_monocle3(adata, *, output_dir, random_state: int = 0, **kwargs):
    """Run Monocle3 through the experiment-local R adapter."""
    return run_r_adapter(
        adata,
        method="monocle3",
        script_name="run_monocle3.R",
        output_dir=output_dir,
        random_state=random_state,
    )
