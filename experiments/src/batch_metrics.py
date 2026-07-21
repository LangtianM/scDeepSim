"""Batch integration and biological preservation metrics.

All functions accept NumPy-like matrices with cells as rows and return scalar
values. Batch metrics are intended to decrease when batch effects are removed,
whereas biological preservation metrics should stay stable as control strength
changes.
"""

import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Batch separation metrics (higher = stronger batch effect)
# ---------------------------------------------------------------------------

def batch_asw(X, batch_labels):
    """Average Silhouette Width with batch as the grouping label.

    Returns a value in [-1, 1].  Higher values indicate stronger batch
    separation (cells cluster by batch rather than mixing).
    """
    labels = LabelEncoder().fit_transform(np.asarray(batch_labels))
    if len(np.unique(labels)) < 2:
        return 0.0
    scores = silhouette_samples(X, labels)
    return float(np.mean(scores))


def lisi(X, labels, k=30):
    """Local Inverse Simpson Index (LISI) averaged over all cells.

    For each cell the proportion of each label in its k-nearest neighbourhood
    is computed, then the inverse Simpson index ``1 / sum(p_c^2)`` gives the
    effective number of label categories locally.

    Returns the mean LISI across all cells.  A value of 1 means perfect
    segregation; a value equal to the number of unique labels means perfect
    mixing.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([label_map[l] for l in labels])

    if X.ndim != 2 or X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must contain the same number of cells.")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k >= X.shape[0]:
        raise ValueError("k must be smaller than the number of cells.")

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X)
    # Querying with X=None makes sklearn exclude each training row from its
    # own neighbors, including when duplicated coordinates make distance ties.
    neighbours = nn.kneighbors(X=None, return_distance=False)

    lisi_values = np.empty(len(X))
    for i in range(len(X)):
        neighbour_labels = int_labels[neighbours[i]]
        counts = np.bincount(neighbour_labels, minlength=n_labels)
        proportions = counts / counts.sum()
        simpson = (proportions ** 2).sum()
        lisi_values[i] = 1.0 / simpson if simpson > 0 else float(n_labels)

    return float(np.mean(lisi_values))


def ilisi(X, batch_labels, k=30):
    """Integration LISI (batch mixing).

    Higher iLISI = more batch mixing = weaker batch effect.
    """
    return lisi(X, batch_labels, k=k)


# ---------------------------------------------------------------------------
# Biological preservation metrics (should stay stable across alpha)
# ---------------------------------------------------------------------------

def celltype_asw(X, celltype_labels):
    """Average Silhouette Width with cell type as the grouping label.

    Higher values indicate that cell types remain well separated.
    """
    labels = LabelEncoder().fit_transform(np.asarray(celltype_labels))
    if len(np.unique(labels)) < 2:
        return 0.0
    scores = silhouette_samples(X, labels)
    return float(np.mean(scores))


def clisi(X, celltype_labels, k=30):
    """Cell-type LISI (biological preservation).

    Lower cLISI = cells of the same type stay together = better preservation.
    """
    return lisi(X, celltype_labels, k=k)


def celltype_rf_accuracy(X, celltype_labels, test_size=0.2, seed=42):
    """Random-forest cell-type classification accuracy.

    Returns
    -------
    tuple[float, float]
        ``(accuracy, balanced_accuracy)`` on a held-out split.
    """
    labels = LabelEncoder().fit_transform(np.asarray(celltype_labels))
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=seed,
    )
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return float(accuracy_score(y_test, y_pred)), float(balanced_accuracy_score(y_test, y_pred))


# ---------------------------------------------------------------------------
# Controlled batch-integration benchmark metrics
# ---------------------------------------------------------------------------

def batch_asw_within_celltype(X, batch_labels, celltype_labels):
    """Compute absolute batch ASW stratified equally across cell types.

    Silhouette values are computed separately inside each cell type so the
    score measures technical separation without allowing biological clusters
    to dominate. Each observed cell type contributes equal weight. Lower is
    better and the result lies in ``[0, 1]``.
    """
    X = np.asarray(X, dtype=np.float64)
    batch_labels = np.asarray(batch_labels)
    celltype_labels = np.asarray(celltype_labels)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional cell-by-feature matrix.")
    if X.shape[0] == 0:
        raise ValueError("At least one cell is required.")
    if batch_labels.ndim != 1 or celltype_labels.ndim != 1:
        raise ValueError("batch_labels and celltype_labels must be one-dimensional.")
    if not (X.shape[0] == batch_labels.shape[0] == celltype_labels.shape[0]):
        raise ValueError("X and both label arrays must have matching lengths.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    per_celltype = []
    for celltype in np.unique(celltype_labels):
        mask = celltype_labels == celltype
        X_group = X[mask]
        group_batches = LabelEncoder().fit_transform(batch_labels[mask])
        n_batches = np.unique(group_batches).size
        if n_batches < 2 or X_group.shape[0] <= n_batches:
            raise ValueError(
                f"Cell type {celltype!r} must contain at least two batches "
                "and more cells than batch labels."
            )
        scores = silhouette_samples(X_group, group_batches)
        per_celltype.append(float(np.mean(np.abs(scores))))
    return float(np.mean(per_celltype))


def compute_batch_integration_metrics(
    X,
    batch_labels,
    celltype_labels,
    lisi_k=30,
):
    """Return the four raw metrics used by the integration benchmark."""
    metrics = {
        "batch_asw": batch_asw_within_celltype(
            X,
            batch_labels,
            celltype_labels,
        ),
        "ilisi": ilisi(X, batch_labels, k=lisi_k),
        "celltype_asw": celltype_asw(X, celltype_labels),
        "clisi": clisi(X, celltype_labels, k=lisi_k),
    }
    return {name: float(value) for name, value in metrics.items()}
