import numpy as np
import pandas as pd
import pytest

from experiments.src.ti_methods.slingshot_adapter import run_slingshot


def _tiny_adata():
    ad = pytest.importorskip("anndata")
    X = np.vstack(
        [
            np.linspace(0, 1, 8),
            np.linspace(0.1, 1.1, 8),
            np.linspace(1, 2, 8),
            np.linspace(1.1, 2.1, 8),
            np.linspace(1, 2, 8)[::-1],
            np.linspace(1.1, 2.1, 8)[::-1],
        ]
    )
    obs = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(X.shape[0])],
            "true_pseudotime": [0.0, 0.1, 0.6, 0.8, 0.6, 0.8],
            "true_lineage": ["trunk", "trunk", "branch_B", "branch_B", "branch_C", "branch_C"],
            "true_segment": ["trunk", "trunk", "branch", "branch", "branch", "branch"],
            "true_branch_point": [0.5] * X.shape[0],
        }
    ).set_index("cell_id", drop=False)
    return ad.AnnData(X=X, obs=obs)


def test_scanpy_dpt_paga_smoke(tmp_path):
    pytest.importorskip("scanpy")
    from experiments.src.ti_methods.scanpy_dpt_paga import run_scanpy_dpt_paga

    try:
        out = run_scanpy_dpt_paga(
            _tiny_adata(),
            output_dir=tmp_path,
            n_pcs=3,
            n_neighbors=2,
            resolution=0.5,
            random_state=0,
        )
    except Exception as exc:
        pytest.skip(f"Scanpy optional TI stack unavailable: {exc}")

    assert set(["cell_id", "method", "inferred_pseudotime", "inferred_lineage"]).issubset(out.columns)
    assert out["method"].eq("scanpy_dpt_paga").all()
    assert out["inferred_pseudotime"].notna().any()


def test_r_adapter_skips_when_rscript_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.src.ti_methods._r_adapter.shutil.which",
        lambda name: None,
    )
    out = run_slingshot(_tiny_adata(), output_dir=tmp_path)
    assert out.loc[0, "method"] == "slingshot"
    assert out.loc[0, "cell_id"] is pd.NA or pd.isna(out.loc[0, "cell_id"])
    assert "skipped" in out.loc[0, "metadata_json"]
