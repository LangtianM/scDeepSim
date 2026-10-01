"""Plot one held-out transfer from saved artifacts without running a VAE.

Usage:
    python experiments/scripts/plot_scgen_style_batch_transfer.py /absolute/run --celltype alpha --seed 42
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP


def illustration_coordinates(directory, config):
    """Fit on this repetition's VAE training cells and cache projections."""
    path = directory / "embedding.csv"
    if path.exists():
        return pd.read_csv(path)
    permitted = ad.read_h5ad(directory.parent / "permitted.h5ad")
    target = ad.read_h5ad(directory.parent / "target.h5ad")
    prediction = ad.read_h5ad(directory / "predictions.h5ad")
    source = permitted[prediction.obs.source_cell_id.to_numpy()].copy()
    with np.load(directory / "splits.npz") as splits:
        fitting = permitted[splits["train"]].copy()
    pca = PCA(n_components=min(30, fitting.n_vars, fitting.n_obs - 1),
              svd_solver="randomized", random_state=42)
    fitting_pca = pca.fit_transform(fitting.X)
    embedding = UMAP(
        n_neighbors=int(config["evaluation"]["umap_n_neighbors"]),
        min_dist=float(config["evaluation"]["umap_min_dist"]),
        random_state=42, transform_seed=42, n_jobs=1,
    )
    fitting_umap = embedding.fit_transform(fitting_pca)
    frames = []
    for group, data in (("fitting", fitting), ("source", source),
                        ("target", target), ("predicted", prediction)):
        coords = fitting_umap if group == "fitting" else embedding.transform(pca.transform(data.X))
        frames.append(pd.DataFrame({
            "cell_id": data.obs_names, "group": group,
            "umap_1": coords[:, 0], "umap_2": coords[:, 1],
        }))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(path, index=False)
    return result


def plot_umap(ax, coordinates, celltype, source_batch, target_batch):
    """Show fitting cells, real source/target cells, and simulated targets."""
    name = celltype.replace("_", " ")
    for group, color, label, size in (
        ("fitting", "#cccccc", "VAE fitting cells", 2),
        ("source", "#4c78a8", f"{source_batch} {name}", 10),
        ("target", "#54a24b", f"Real {target_batch} {name}", 10),
        ("predicted", "#e68135", f"Simulated {target_batch} {name}", 10),
    ):
        subset = coordinates.loc[coordinates.group == group]
        ax.scatter(subset.umap_1, subset.umap_2, s=size, color=color,
                   alpha=0.45 if group == "fitting" else 0.65, linewidths=0,
                   label=label, rasterized=True)
    ax.set_title("A  UMAP visualization", loc="left", fontweight="bold")
    ax.set(xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(fontsize=8, frameon=False, markerscale=1.5, loc="best")


def plot_gene_statistic(ax, per_gene, metrics, statistic, label, letter):
    """Compare real and simulated per-gene moments with an identity line."""
    observed = per_gene[f"target_{statistic}"].to_numpy()
    simulated = per_gene[f"predicted_{statistic}"].to_numpy()
    ax.scatter(observed, simulated, s=9, alpha=0.45, color="#4c78a8", rasterized=True)
    limits = (0, max(observed.max(), simulated.max()) * 1.05)
    ax.plot(limits, limits, "--", color="#777777", linewidth=1)
    ax.set(xlabel=f"Real target {label.lower()}", ylabel=f"Simulated {label.lower()}",
           xlim=limits, ylim=limits, aspect="equal")
    ax.set_title(f"{letter}  Gene {label.lower()} expression", loc="left", fontweight="bold")
    ax.text(0.04, 0.96,
            f"r = {metrics[f'{statistic}_pearson']:.3f}\n"
            f"$R^2$ = {metrics[f'{statistic}_r2']:.3f}\n"
            f"RMSE = {metrics[f'{statistic}_rmse']:.3f}",
            va="top", transform=ax.transAxes)
    ax.spines[["top", "right"]].set_visible(False)


def plot_task(run_dir, celltype, seed):
    """Save one UMAP/mean/standard-deviation figure as PNG and PDF."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "experiment_config.json").read_text())
    directory = run_dir / "holdouts" / celltype / f"seed_{seed}"
    coordinates = illustration_coordinates(directory, config)
    per_gene = pd.read_csv(directory / "per_gene.csv")
    metrics = pd.read_csv(directory / "metrics.csv").set_index("gene_subset").loc["all_genes"]
    source = config["split"]["reference_batch"]
    target = config["split"]["target_batch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained")
    plot_umap(axes[0], coordinates, celltype, source, target)
    plot_gene_statistic(axes[1], per_gene, metrics, "mean", "Mean", "B")
    plot_gene_statistic(axes[2], per_gene, metrics, "std", "Standard deviation", "C")
    # fig.suptitle(
    #     f"{celltype.replace('_', ' ')} · seed {seed} · {source} → {target} · α = 1\n"
    #     f"Log-normalized expression · {len(per_gene):,} genes",
    #     fontsize=14,
    # )
    output = run_dir / "figures" / celltype
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"seed_{seed}.{extension}", dpi=220)
    plt.close(fig)
    return output / f"seed_{seed}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    print(plot_task(args.run_dir, args.celltype, args.seed))


if __name__ == "__main__":
    main()
