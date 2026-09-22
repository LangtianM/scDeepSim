"""Render completed unified dataset runs with mean and sample SD across seeds."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"structured": "#286FAD", "plain": "#D97A32"}
METRICS = {"batch_asw": "Signed within-cell-type batch ASW", "ilisi": "iLISI",
           "celltype_asw": "Cell-type ASW", "clisi": "cLISI",
           "celltype_rf_bal_acc": "Cell-type RF balanced accuracy",
           "celltype_rf_acc": "Cell-type RF accuracy"}


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def curve(ax, summary, metric):
    for model, color in COLORS.items():
        rows = summary[summary.model == model].sort_values("alpha")
        x, y, sd = (rows[column].to_numpy() for column in
                    ("alpha", f"{metric}_mean", f"{metric}_sd"))
        ax.plot(x, y, marker="o", markersize=3, label=model, color=color)
        ax.fill_between(x, y - sd, y + sd, alpha=0.16, color=color)
    ax.set(xlabel="Intervention strength α", ylabel=METRICS[metric])
    ax.spines[["top", "right"]].set_visible(False)


def render_dataset(directory):
    figures = directory / "figures"
    latent = pd.read_csv(directory / "latent_predictability/results/latent_predictability_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    order = pd.MultiIndex.from_product([COLORS, ["z_celltype", "z_batch", "z_residual"]],
                                       names=["model", "subspace"])
    for ax, target in zip(axes, ["celltype", "batch"]):
        rows = latent[latent.label == target].set_index(["model", "subspace"]).reindex(order)
        means = rows.balanced_accuracy_mean.to_numpy().reshape(2, 3)
        sd = rows.balanced_accuracy_sd.to_numpy().reshape(2, 3)
        im = ax.imshow(means, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i, j in np.ndindex(means.shape):
            ax.text(j, i, f"{means[i, j]:.3f}\n±{sd[i, j]:.3f}", ha="center", va="center",
                    color="white" if means[i, j] > 0.6 else "black", fontsize=9)
        ax.set(xticks=range(3), xticklabels=["Cell type", "Batch", "Residual"],
               yticks=range(2), yticklabels=list(COLORS), title=f"{target} predictability",
               xlabel="Latent subspace")
    fig.colorbar(im, ax=axes, label="RF balanced accuracy")
    fig.suptitle(f"{directory.name}: within-dataset latent predictability (mean ± SD, n=3)")
    save(fig, figures, "latent_predictability")

    summary = pd.read_csv(directory / "dose_response/results/dose_response_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        curve(ax, summary, metric)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"{directory.name}: batch dose response (mean ± SD, n=3)")
    save(fig, figures, "dose_response")

    per_type = pd.read_csv(directory / "dose_response/results/dose_response_celltype_summary.csv")
    support = pd.read_csv(directory / "dose_response/results/direction_support.csv")
    counts = support.groupby("celltype").matched_count.sum()
    types = sorted(per_type.celltype.unique())
    metrics = ["batch_asw", "ilisi_within_celltype", "celltype_asw"]
    fig, axes = plt.subplots(1, 6, figsize=(20, max(5, len(types) * 0.32)), constrained_layout=True)
    for ax, (model, metric) in zip(axes, [(m, k) for m in COLORS for k in metrics]):
        rows = per_type[per_type.model == model]
        matrix = rows.pivot(index="celltype", columns="alpha", values=f"{metric}_mean").reindex(types)
        limits = (1, 2) if metric == "ilisi_within_celltype" else (-1, 1)
        im = ax.imshow(matrix, aspect="auto", vmin=limits[0], vmax=limits[1],
                       cmap="viridis" if metric == "ilisi_within_celltype" else "RdBu_r")
        ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
               title=f"{model}\n{metric}", xlabel="α", yticks=range(len(types)),
               yticklabels=[f"{t} (n={counts[t]})" for t in types] if ax is axes[0] else [])
        ax.tick_params(axis="x", labelrotation=90)
        fig.colorbar(im, ax=ax, shrink=0.5)
    fig.suptitle(f"{directory.name}: per-type means; n is matched real cells per batch")
    save(fig, figures, "dose_response_celltypes")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output.resolve()
    completed = {}
    for name in ["immune", "lung", "embryo"]:
        directory = output / name
        if (directory / "dose_response/results/dose_response_summary.csv").exists():
            completed[name] = render_dataset(directory)
    if completed:
        fig, axes = plt.subplots(len(completed), 4, figsize=(16, 3.4 * len(completed)),
                                 squeeze=False, constrained_layout=True)
        for row, (name, summary) in zip(axes, completed.items()):
            for ax, metric in zip(row, list(METRICS)[:4]):
                curve(ax, summary, metric)
                ax.set_title(name)
        axes[0, 0].legend(frameon=False)
        save(fig, output / "figures", "dose_response_across_datasets")


if __name__ == "__main__":
    main()
