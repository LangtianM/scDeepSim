import numpy as np

import experiments.src.batch_integration as adapters
import experiments.scripts.benchmark_batch_integration as benchmark
from experiments.src.batch_integration import (
    IntegrationResult,
    run_combat,
    run_harmony,
    run_scanorama,
    run_unintegrated_pca,
)


def _matrix():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 6)).astype(np.float32)
    batches = np.asarray(["A", "B"] * 6)
    X[batches == "B"] += 0.5
    return X, batches


def test_unintegrated_and_combat_return_finite_order_aligned_embeddings():
    X, batches = _matrix()

    unintegrated = run_unintegrated_pca(
        X, batches, n_components=3, seed=7
    )
    combat = run_combat(X, batches, n_components=3, seed=7)

    for result in [unintegrated, combat]:
        assert result.status == "success", result.error
        assert result.embedding.shape == (X.shape[0], 3)
        assert np.isfinite(result.embedding).all()
        assert result.error is None
    assert np.array_equal(X, _matrix()[0])


def test_missing_optional_dependencies_return_clear_failures(monkeypatch):
    X, batches = _matrix()
    monkeypatch.setattr(adapters, "find_spec", lambda name: None)

    harmony = run_harmony(X, batches, n_components=3)
    scanorama = run_scanorama(X, batches, n_components=3)

    assert harmony.status == "failed"
    assert harmony.embedding is None
    assert "harmonypy==0.0.10" in harmony.error
    assert scanorama.status == "failed"
    assert scanorama.embedding is None
    assert "scanorama==1.7.4" in scanorama.error


def test_scanorama_stable_sorts_and_restores_original_order(monkeypatch):
    import scanpy.external as sce

    X = np.arange(30, dtype=np.float32).reshape(6, 5)
    batches = np.asarray(["B", "A", "B", "A", "B", "A"])
    shared_pca = np.column_stack(
        [np.arange(6, dtype=np.float32), np.arange(6, dtype=np.float32) + 20]
    )

    monkeypatch.setattr(adapters, "find_spec", lambda name: object())

    def fake_scanorama(work, key, basis, adjusted_basis):
        observed = work.obs[key].astype(str).tolist()
        assert observed == ["A", "A", "A", "B", "B", "B"]
        work.obsm[adjusted_basis] = work.obsm[basis] + 100.0

    monkeypatch.setattr(sce.pp, "scanorama_integrate", fake_scanorama)

    result = run_scanorama(
        X,
        batches,
        n_components=2,
        seed=4,
        pca_embedding=shared_pca,
    )

    assert result.status == "success", result.error
    assert np.array_equal(result.embedding, shared_pca + 100.0)
    assert result.metadata["order_restored"] is True


def test_adapter_failure_does_not_stop_remaining_methods(monkeypatch):
    X, batches = _matrix()
    failed = IntegrationResult(
        "unintegrated", None, "failed", 0.1, {}, "RuntimeError: expected"
    )
    succeeded = IntegrationResult(
        "combat", np.ones((12, 2)), "success", 0.2, {}, None
    )
    monkeypatch.setattr(benchmark, "run_unintegrated_pca", lambda *a, **k: failed)
    monkeypatch.setattr(benchmark, "run_combat", lambda *a, **k: succeeded)

    results = benchmark.run_integration_methods(
        X,
        batches,
        ["unintegrated", "combat"],
        n_components=2,
        seed=3,
    )

    assert [result.status for result in results] == ["failed", "success"]
    assert results[1].embedding.shape == (12, 2)
