"""Trajectory-inference method adapters for experiment-level benchmarks."""

from .monocle3_adapter import run_monocle3
from .scanpy_dpt_paga import run_scanpy_dpt_paga
from .slingshot_adapter import run_slingshot


ADAPTERS = {
    "scanpy_dpt_paga": run_scanpy_dpt_paga,
    "slingshot": run_slingshot,
    "monocle3": run_monocle3,
}


__all__ = ["ADAPTERS", "run_scanpy_dpt_paga", "run_slingshot", "run_monocle3"]
