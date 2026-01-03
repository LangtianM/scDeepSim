"""
Utility functions for scDeepSim project.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
import umap


def compare_umap(
    data1: Union[np.ndarray, list],
    data2: Union[np.ndarray, list],
    labels1: Optional[Union[np.ndarray, list]] = None,
    labels2: Optional[Union[np.ndarray, list]] = None,
    title1: str = "Dataset 1",
    title2: str = "Dataset 2",
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
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Compare two datasets using UMAP visualization with consistent dimensionality reduction.

    This function concatenates two datasets, applies UMAP transformation to ensure
    consistent dimensionality reduction, then creates side-by-side visualizations.

    Parameters
    ----------
    data1 : array-like, shape (n_samples1, n_features)
        First dataset to visualize.
    data2 : array-like, shape (n_samples2, n_features)
        Second dataset to visualize.
    labels1 : array-like, shape (n_samples1,), optional
        Labels for coloring points in the first dataset.
    labels2 : array-like, shape (n_samples2,), optional
        Labels for coloring points in the second dataset.
    title1 : str, default="Dataset 1"
        Title for the first subplot.
    title2 : str, default="Dataset 2"
        Title for the second subplot.
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

    Returns
    -------
    embedding1 : np.ndarray, shape (n_samples1, 2)
        UMAP embedding for the first dataset.
    embedding2 : np.ndarray, shape (n_samples2, 2)
        UMAP embedding for the second dataset.
    fig : matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> data1 = np.random.randn(100, 50)
    >>> data2 = np.random.randn(150, 50)
    >>> emb1, emb2, fig = compare_umap(data1, data2)
    >>> plt.show()
    """
    # Convert to numpy arrays
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # Validate input shapes
    if data1.ndim != 2 or data2.ndim != 2:
        raise ValueError("Both data1 and data2 must be 2D arrays")

    if data1.shape[1] != data2.shape[1]:
        raise ValueError(
            f"Feature dimensions must match: data1 has {data1.shape[1]} features, "
            f"data2 has {data2.shape[1]} features"
        )

    n_samples1 = data1.shape[0]

    # Concatenate datasets for consistent UMAP transformation
    data_combined = np.vstack([data1, data2])

    # Perform UMAP on combined data
    print(
        f"Performing UMAP on combined dataset ({data_combined.shape[0]} samples, "
        f"{data_combined.shape[1]} features)..."
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
    embedding1 = embedding_combined[:n_samples1]
    embedding2 = embedding_combined[n_samples1:]

    print("UMAP completed. Generating visualizations...")

    # Create side-by-side visualizations
    fig = _create_comparison_plot(
        embedding1=embedding1,
        embedding2=embedding2,
        labels1=labels1,
        labels2=labels2,
        title1=title1,
        title2=title2,
        figsize=figsize,
        cmap=cmap,
        alpha=alpha,
        s=s,
    )

    # Save figure if path is provided
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return embedding1, embedding2, fig


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

        # Use colormap
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