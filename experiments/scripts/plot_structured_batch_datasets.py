"""Render predictability and dose response for one dataset in the notebook format."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HEATMAP_SIZE = (10, 5.5)
DOSE_SIZE = (13, 5.5)
FONT_SIZE = 10
plt.rcParams.update({"font.size": FONT_SIZE, "axes.spines.top": False})


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


BLOCKS = ['z_celltype', 'z_batch', 'z_residual']
BLOCK_NAMES = ['Cell-type', 'Batch', 'Residual block']
TARGETS = ['celltype', 'batch']
BLOCK_COLORS = ['#8dd3c7', '#bebada', '#fb8072']


def mean_sd(mean, sd):
    return f'{mean:.3f}' if pd.isna(sd) else f'{mean:.3f} ± {sd:.3f}'


def draw_predictability(fig, predictability_summary, batch_label='Batch'):
    heat_stats = predictability_summary.set_index(['model', 'label', 'subspace'])
    target_names = ['Cell type', batch_label]
    grid = fig.add_gridspec(2, 2, height_ratios=[0.65, 2], width_ratios=[1, 0.04],
                           left=0.29, right=0.91, bottom=0.13, top=0.85,
                           hspace=0.10, wspace=0.07)
    schematic = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    for column, block in enumerate(BLOCKS):
        schematic.add_patch(plt.Rectangle((column, 0), 1, 1, color=BLOCK_COLORS[column]))
        schematic.text(column + 0.5, 0.5, f'{BLOCK_NAMES[column]}',
                       ha='center', va='center', fontsize=FONT_SIZE)
    schematic.set(xlim=(0, 3), ylim=(0, 1))
    schematic.axis('off')
    matrix = np.array([[heat_stats.loc[('structured', target, block), 'balanced_accuracy_mean']
                        for block in BLOCKS] for target in TARGETS])
    mesh = ax.pcolormesh(np.arange(4), np.arange(3), matrix, vmin=0, vmax=1,
                         cmap='viridis', edgecolors='white', linewidth=1)
    row_labels = []
    for row, target in enumerate(TARGETS):
        reference = heat_stats.loc[('structured', target, BLOCKS[0])]
        row_labels.append(f"{target_names[row]}\nRandom: {reference['random_ba_mean']:.3f}\n"
                          )
        for column, block in enumerate(BLOCKS):
            structured = heat_stats.loc[('structured', target, block)]
            plain = heat_stats.loc[('plain', target, block)]
            annotation = mean_sd(structured['balanced_accuracy_mean'], structured['balanced_accuracy_sd'])
            annotation += '\nPlain: ' + mean_sd(plain['balanced_accuracy_mean'], plain['balanced_accuracy_sd'])
            color = 'white' if matrix[row, column] < 0.55 else 'black'
            ax.text(column + 0.5, row + 0.5, annotation, ha='center', va='center',
                    color=color, fontsize=FONT_SIZE - 1)
    ax.set(xticks=np.arange(3) + 0.5, xticklabels=['Cell type', 'Batch', 'Residual'],
           yticks=np.arange(2) + 0.5, yticklabels=row_labels, xlabel='Latent coordinate block')
    ax.invert_yaxis()
    ax.tick_params(length=0)
    cbar = fig.colorbar(mesh, cax=fig.add_subplot(grid[1, 1]))
    cbar.set_label('Structured mean balanced accuracy')


METRIC_STYLES = {
    'batch_asw': ('Batch ASW', '#009E73', 'o'),
    'ilisi': ('iLISI', '#D55E00', 's'),
    'celltype_asw': ('CT ASW', '#8E44AD', '^'),
    'celltype_rf_bal_acc': ('CT RF balanced accuracy', '#0072B2', 'D'),
    'clisi': ('cLISI', '#E69F00', 'v'),
}
MODEL_STYLES = {'structured': ('Structured VAE + diffusion', '-'),
                'plain': ('Plain VAE + diffusion', '--')}


def draw_curve(ax, metric, dose, dose_summary):
    label, color, marker = METRIC_STYLES[metric]
    for model, (_, line_style) in MODEL_STYLES.items():
        summary = dose_summary[dose_summary['model'] == model].sort_values('alpha')
        x = summary['alpha'].to_numpy()
        mean = summary[f'{metric}_mean'].to_numpy()
        sd = summary[f'{metric}_sd'].to_numpy()
        ax.plot(x, mean, color=color, linestyle=line_style, marker=marker,
                markersize=4, linewidth=1.7,
                markerfacecolor=color if model == 'structured' else 'white')
        if np.isfinite(sd).all():
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.09)
        raw = dose[dose['model'] == model]
        ax.scatter(raw['alpha'], raw[metric], s=13, marker=marker,
                   facecolors=color if model == 'structured' else 'none',
                   edgecolors=color, alpha=0.45, linewidths=0.7)


def draw_dose_response(fig, dose, dose_summary, dose_metadata):
    grid = fig.add_gridspec(1, 2, left=0.09, right=0.91, bottom=0.25, top=0.82, wspace=0.65)
    batch_ax, bio_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ilisi_ax, clisi_ax = batch_ax.twinx(), bio_ax.twinx()
    for axis, metric in [(batch_ax, 'batch_asw'), (ilisi_ax, 'ilisi'),
                         (bio_ax, 'celltype_asw'), (bio_ax, 'celltype_rf_bal_acc'),
                         (clisi_ax, 'clisi')]:
        draw_curve(axis, metric, dose, dose_summary)
    batch_ax.set(title='Batch separation', ylabel='Batch ASW')
    ilisi_ax.set_ylabel('iLISI', color=METRIC_STYLES['ilisi'][1])
    bio_ax.set(title='Biological preservation', ylabel='CT ASW / RF balanced accuracy')
    clisi_ax.set_ylabel('cLISI', color=METRIC_STYLES['clisi'][1])
    for axis in (batch_ax, bio_ax):
        axis.set_xlabel(r'Intervention strength $\alpha$')
        axis.set_xticks(dose_metadata['alpha_values'])
        axis.axvline(1, color='0.5', linewidth=0.8, linestyle=':')
        axis.grid(alpha=0.15)
    for axis, metrics in [(batch_ax, ['batch_asw', 'ilisi']),
                          (bio_ax, ['celltype_asw', 'celltype_rf_bal_acc', 'clisi'])]:
        handles = [Line2D([], [], color=METRIC_STYLES[m][1], marker=METRIC_STYLES[m][2],
                          linestyle='', label=METRIC_STYLES[m][0]) for m in metrics]
        axis.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.23),
                    ncol=1, frameon=False, fontsize=FONT_SIZE - 1, handletextpad=0.4)
    handles = [Line2D([], [], color='0.2', linestyle=style, label=label)
               for label, style in MODEL_STYLES.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False)


def load_results(directory):
    vae_run = directory / "latent_predictability"
    dose_run = directory / "dose_response"
    vae_metadata = json.loads((vae_run / "results/metadata.json").read_text())
    dose_metadata = json.loads((dose_run / "results/metadata.json").read_text())
    latent = pd.read_csv(vae_run / "results/latent_predictability_summary.csv")
    dose = pd.read_csv(dose_run / "results/dose_response_metrics.csv")
    summary = pd.read_csv(dose_run / "results/dose_response_summary.csv")
    assert vae_metadata["complete"] and dose_metadata["complete"]
    assert Path(dose_metadata["source_vae_run"]).resolve() == vae_run.resolve()
    assert vae_metadata["seeds"] == dose_metadata["seeds"]
    assert latent["n_replicates"].eq(len(vae_metadata["seeds"])).all()
    assert summary["n_replicates"].eq(len(vae_metadata["seeds"])).all()
    return latent, vae_metadata, dose, summary, dose_metadata


def render_dataset(directory, results, batch_label="Batch"):
    latent, vae_metadata, dose, summary, dose_metadata = results
    figures = directory / "figures"
    fig = plt.figure(figsize=HEATMAP_SIZE)
    draw_predictability(fig, latent, batch_label)
    fig.suptitle("Covariate predictability from latent blocks", y=0.97)
    save(fig, figures, "latent_predictability")

    fig = plt.figure(figsize=DOSE_SIZE)
    draw_dose_response(fig, dose, summary, dose_metadata)
    save(fig, figures, "dose_response")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_results", type=Path,
                        help="Dataset directory containing latent_predictability/ and dose_response/.")
    parser.add_argument("--batch-label", default="Batch", help="Batch label for the predictability heatmap.")
    args = parser.parse_args()
    directory = args.dataset_results.resolve()
    render_dataset(directory, load_results(directory), args.batch_label)


if __name__ == "__main__":
    main()
