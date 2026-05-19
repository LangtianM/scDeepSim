"""
Utility functions for scDeepSim project.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, List, Sequence
import umap


def compare_umap(
    data_list: Sequence[Union[np.ndarray, list]],
    labels_list: Optional[Sequence[Optional[Union[np.ndarray, list]]]] = None,
    title_list: Optional[Sequence[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "tab10",
    alpha: float = 0.6,
    s: int = 10,
    save_path: Optional[str] = None,
    dpi: int = 300,
    share_axes: bool = True,
    equal_aspect: bool = False,
    axis_padding: float = 0.05,
) -> Tuple[List[np.ndarray], plt.Figure]:
    """
    Compare multiple datasets using UMAP visualization with consistent dimensionality reduction.

    This function concatenates all datasets, applies UMAP transformation once (so all
    datasets share the same embedding space), then creates a 1×N panel visualization.

    Parameters
    ----------
    data_list : sequence of array-like
        List of datasets, each with shape (n_samples_i, n_features).
    labels_list : sequence of array-like or None, optional
        List of label arrays aligned with `data_list`. Each element can be None.
        If None, all datasets are plotted without labels.
    title_list : sequence of str, optional
        Titles aligned with `data_list`. If None, uses "Dataset 1..N".
    n_neighbors : int, default=15
        Number of neighbors for UMAP.
    min_dist : float, default=0.1
        Minimum distance parameter for UMAP.
    metric : str, default="euclidean"
        Distance metric for UMAP.
    random_state : int, default=42
        Random state for reproducibility.
    figsize : tuple, default=(14, 6)
        Figure size (width, height) in inches.
    cmap : str, default="tab10"
        Colormap for visualization.
    alpha : float, default=0.6
        Transparency of points.
    s : int, default=10
        Size of points.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        DPI for saving the figure.
    share_axes : bool, default=True
        If True, all panels use the same UMAP x/y limits.
    equal_aspect : bool, default=False
        If True, draw each panel with equal x/y data-unit scaling.
    axis_padding : float, default=0.05
        Fractional padding added around the shared embedding extent.

    Returns
    -------
    embeddings : list of np.ndarray
        List of UMAP embeddings, each with shape (n_samples_i, 2).
    fig : matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> datasets = [np.random.randn(100, 50), np.random.randn(150, 50)]
    >>> labels = [np.random.randint(0, 3, size=100), np.random.randint(0, 3, size=150)]
    >>> titles = ["A", "B"]
    >>> embs, fig = compare_umap(datasets, labels, titles)
    >>> plt.show()
    """
    if data_list is None or len(data_list) == 0:
        raise ValueError("data_list must be a non-empty sequence of datasets")

    datasets: List[np.ndarray] = [np.asarray(d) for d in data_list]

    # Validate input shapes and feature dims
    if any(ds.ndim != 2 for ds in datasets):
        raise ValueError("All datasets in data_list must be 2D arrays")

    n_features = datasets[0].shape[1]
    for i, ds in enumerate(datasets):
        if ds.shape[1] != n_features:
            raise ValueError(
                f"Feature dimensions must match across datasets: dataset 0 has {n_features} "
                f"features, dataset {i} has {ds.shape[1]} features"
            )

    # Normalize labels_list
    if labels_list is None:
        norm_labels: List[Optional[np.ndarray]] = [None for _ in datasets]
    else:
        if len(labels_list) != len(datasets):
            raise ValueError("labels_list must have the same length as data_list")
        norm_labels = []
        for i, (ds, lab) in enumerate(zip(datasets, labels_list)):
            if lab is None:
                norm_labels.append(None)
                continue
            lab_arr = np.asarray(lab)
            if lab_arr.shape[0] != ds.shape[0]:
                raise ValueError(
                    f"Labels length mismatch for dataset {i}: got {lab_arr.shape[0]} labels, "
                    f"but dataset has {ds.shape[0]} samples"
                )
            norm_labels.append(lab_arr)

    # Normalize title_list
    if title_list is None:
        titles: List[str] = [f"Dataset {i+1}" for i in range(len(datasets))]
    else:
        if len(title_list) != len(datasets):
            raise ValueError("title_list must have the same length as data_list")
        titles = [str(t) for t in title_list]

    # Build a shared label→color mapping so the same cell type gets the same
    # color in every panel, regardless of how many panels there are.
    all_labels = np.concatenate(
        [np.asarray(lab) for lab in norm_labels if lab is not None]
    ) if any(lab is not None for lab in norm_labels) else np.array([])
    global_unique = np.unique(all_labels) if len(all_labels) > 0 else np.array([])
    colormap = plt.get_cmap(cmap)
    n_global = max(len(global_unique), 1)
    shared_color_dict = {
        label: colormap(i / n_global) for i, label in enumerate(global_unique)
    }

    # Concatenate datasets for consistent UMAP transformation
    sizes = [ds.shape[0] for ds in datasets]
    data_combined = np.vstack(datasets)

    # Perform UMAP on combined data
    print(
        f"Performing UMAP on combined dataset ({data_combined.shape[0]} samples, "
        f"{data_combined.shape[1]} features) across {len(datasets)} datasets..."
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
    )

    embedding_combined = reducer.fit_transform(data_combined)

    # Split embeddings back into original datasets
    embeddings: List[np.ndarray] = []
    start = 0
    for n in sizes:
        embeddings.append(embedding_combined[start : start + n])
        start += n

    print("UMAP completed. Generating visualizations...")

    # Create 1×N panel visualization
    per_panel_w = figsize[0] / 2.0
    multi_figsize = (max(per_panel_w * len(datasets), 6.0), figsize[1])
    fig, axes = plt.subplots(1, len(datasets), figsize=multi_figsize)
    if len(datasets) == 1:
        axes = [axes]

    for i, (emb, lab, title) in enumerate(zip(embeddings, norm_labels, titles)):
        plot_umap(
            embedding=emb,
            labels=lab,
            title=title,
            ax=axes[i],
            cmap=cmap,
            alpha=alpha,
            s=s,
            color_dict=shared_color_dict if shared_color_dict else None,
        )

    if share_axes:
        x_min, y_min = embedding_combined.min(axis=0)
        x_max, y_max = embedding_combined.max(axis=0)
        x_pad = (x_max - x_min) * axis_padding
        y_pad = (y_max - y_min) * axis_padding
        for ax in axes:
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    if equal_aspect:
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    # Save figure if path is provided
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return embeddings, fig


def _create_comparison_plot(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    labels1: Optional[Union[np.ndarray, list]] = None,
    labels2: Optional[Union[np.ndarray, list]] = None,
    title1: str = "Dataset 1",
    title2: str = "Dataset 2",
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "tab10",
    alpha: float = 0.6,
    s: int = 10,
) -> plt.Figure:
    """
    Create side-by-side UMAP comparison plots.

    Parameters
    ----------
    embedding1 : np.ndarray, shape (n_samples1, 2)
        UMAP embedding for the first dataset.
    embedding2 : np.ndarray, shape (n_samples2, 2)
        UMAP embedding for the second dataset.
    labels1 : array-like, optional
        Labels for coloring points in the first dataset.
    labels2 : array-like, optional
        Labels for coloring points in the second dataset.
    title1 : str
        Title for the first subplot.
    title2 : str
        Title for the second subplot.
    figsize : tuple
        Figure size (width, height) in inches.
    cmap : str
        Colormap for visualization.
    alpha : float
        Transparency of points.
    s : int
        Size of points.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot first dataset
    plot_umap(
        embedding=embedding1,
        labels=labels1,
        title=title1,
        ax=axes[0],
        cmap=cmap,
        alpha=alpha,
        s=s,
    )

    # Plot second dataset
    plot_umap(
        embedding=embedding2,
        labels=labels2,
        title=title2,
        ax=axes[1],
        cmap=cmap,
        alpha=alpha,
        s=s,
    )

    plt.tight_layout()

    return fig


def plot_umap(
    embedding: np.ndarray,
    labels: Optional[Union[np.ndarray, list]] = None,
    title: str = "UMAP Projection",
    ax: Optional[plt.Axes] = None,
    cmap: str = "tab10",
    alpha: float = 0.6,
    s: int = 10,
    show_legend: bool = True,
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
    color_dict: Optional[dict] = None,
) -> plt.Axes:
    """
    Plot a single UMAP embedding.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_samples, 2)
        2D UMAP embedding to visualize.
    labels : array-like, optional
        Labels for coloring points. If None, all points are colored the same.
    title : str, default="UMAP Projection"
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes.
    cmap : str, default="tab10"
        Colormap for visualization.
    alpha : float, default=0.6
        Transparency of points.
    s : int, default=10
        Size of points.
    show_legend : bool, default=True
        Whether to show legend when labels are provided.
    xlabel : str, default="UMAP 1"
        Label for x-axis.
    ylabel : str, default="UMAP 2"
        Label for y-axis.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    if ax is None:
        ax = plt.gca()

    # Validate embedding shape
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("Embedding must be a 2D array with shape (n_samples, 2)")

    # Plot with or without labels
    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)

        # Prefer a globally-consistent color dict; fall back to per-panel colormap.
        if color_dict is None:
            colormap = plt.get_cmap(cmap)
            n = max(len(unique_labels), 1)
            color_dict = {label: colormap(i / n) for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color_dict.get(label, (0.5, 0.5, 0.5, 1.0))],
                label=str(label),
                alpha=alpha,
                s=s,
                edgecolors="none",
            )

        if show_legend:
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, fontsize=8
            )
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1], alpha=alpha, s=s, edgecolors="none"
        )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    return ax

def umap_plot(
    data: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    title: str = "UMAP Projection",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    cmap: str = "tab10",
    alpha: float = 0.6,
    s: int = 10,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    Plot UMAP embedding of a single dataset with labels.
    """
    # INSERT_YOUR_CODE
    data = np.asarray(data)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(data)

    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        colors = plt.get_cmap(cmap)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors(i / len(unique_labels))],
                label=str(label),
                alpha=alpha,
                s=s,
                edgecolors="none",
            )
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, fontsize=8
        )
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            alpha=alpha, s=s, edgecolors="none"
        )

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return embedding, fig
