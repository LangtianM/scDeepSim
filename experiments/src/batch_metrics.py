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

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    neighbours = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self

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
