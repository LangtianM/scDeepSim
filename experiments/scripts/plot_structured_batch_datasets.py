"""Plot saved results for one dataset in the manuscript's structured-intervention style."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HEATMAP_SIZE = (13, 4.5)
DOSE_SIZE = (13, 5.5)
COMPOSITE_SIZE = (13, 8.8)
ROW_GAP = 0.07
FONT_SIZE = 10
MARKER_SIZE = 4
SEED_SIZE = 13
LINE_WIDTH = 1.7
SEED_ALPHA = 0.45
BAND_ALPHA = 0.09
HEATMAP_CMAP = 'viridis'
DOSE_LAYOUT = dict(left=0.09, right=0.91, bottom=0.25, top=0.82, wspace=0.45)
BLOCKS = ['z_celltype', 'z_batch', 'z_residual']
BLOCK_NAMES = ['Cell-type', 'Batch', 'Residual']
MODEL_STYLES = {'structured': ('Structured', '-'), 'plain': ('Plain', '--')}
METRIC_STYLES = {
    'batch_asw': ('Batch ASW', '#009E73', 'o'),
    'ilisi': ('iLISI', '#D55E00', 's'),
    'celltype_asw': ('Cell-type ASW', '#8E44AD', '^'),
    'clisi': ('cLISI', '#E69F00', 'v'),
}
plt.rcParams.update({'font.size': FONT_SIZE, 'axes.titlesize': FONT_SIZE + 2,
                     'axes.labelsize': FONT_SIZE, 'axes.spines.top': False,
                     'pdf.fonttype': 42})


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=300, facecolor='white')
    fig.savefig(directory / f"{name}.pdf", facecolor='white')
    plt.close(fig)


def draw_predictability(fig, predictability_summary, batch_label='Batch'):
    latent_stats = predictability_summary.set_index(['model', 'label', 'subspace'])
    grid = fig.add_gridspec(
        1, 2, left=0.09, right=0.85,
        bottom=0.15, top=0.85, wspace=0.2,
    )
    for col, (model, (label, _)) in enumerate(MODEL_STYLES.items()):
        ax = fig.add_subplot(grid[0, col])
        values = np.array([
            [latent_stats.loc[(model, target, block), 'balanced_accuracy_mean']
             for block in BLOCKS] for target in ['celltype', 'batch']
        ])
        mesh = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect='auto')
        for (row, column), value in np.ndenumerate(values):
            ax.text(column, row, f'{value:.2f}', ha='center', va='center',
                    color='white' if value < 0.55 else 'black', fontsize=FONT_SIZE + 1)
        ax.set(
            title=label + ' VAE',
            xlabel='Latent coordinate block',
            ylabel='Predicted label' if col == 0 else '',
            xticks=range(3),
            xticklabels=BLOCK_NAMES,
            yticks=range(2),
            yticklabels=['Cell\ntype', batch_label] if col == 0 else ['', ''],
        )
        ax.set_xticks(np.arange(0.5, 2.5), minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='both', length=0)
        ax.spines[:].set_visible(False)

    cbar = fig.colorbar(
        mesh,
        cax=fig.add_axes([0.9, 0.15, 0.02, 0.70]),
        orientation='vertical',
        ticks=np.linspace(0, 1, 6),
    )
    cbar.ax.set_xlabel('Mean balanced\naccuracy', labelpad=10)
    cbar.outline.set_visible(False)


def draw_curve(ax, metric, dose, dose_summary):
    label, color, marker = METRIC_STYLES[metric]
    for model, (_, line_style) in MODEL_STYLES.items():
        summary = dose_summary[dose_summary['model'] == model].sort_values('alpha')
        x = summary['alpha'].to_numpy()
        mean = summary[f'{metric}_mean'].to_numpy()
        sd = summary[f'{metric}_sd'].to_numpy()
        ax.plot(x, mean, color=color, linestyle=line_style, marker=marker,
                markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                markerfacecolor=color if model == 'structured' else 'white')
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=BAND_ALPHA)
        raw = dose[dose['model'] == model]
        ax.scatter(raw['alpha'], raw[metric], s=SEED_SIZE, marker=marker,
                   facecolors=color if model == 'structured' else 'none',
                   edgecolors=color, alpha=SEED_ALPHA, linewidths=0.7)


def draw_dose_response(fig, dose, dose_summary, dose_metadata):
    grid = fig.add_gridspec(1, 2, **DOSE_LAYOUT)
    batch_ax, bio_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ilisi_ax, clisi_ax = batch_ax.twinx(), bio_ax.twinx()
    for axis, metric in [(batch_ax, 'batch_asw'), (ilisi_ax, 'ilisi'),
                         (bio_ax, 'celltype_asw'),
                         (clisi_ax, 'clisi')]:
        draw_curve(axis, metric, dose, dose_summary)
    batch_ax.set(title='Batch separation', ylabel='Batch ASW')
    ilisi_ax.set_ylabel('iLISI')
    bio_ax.set(title='Cell type structure preservation', ylabel='Cell-type ASW')
    clisi_ax.set_ylabel('cLISI')
    for axis in (batch_ax, bio_ax):
        axis.set_xlabel(r'Intervention strength $\alpha$')
        axis.set_xticks(dose_metadata['alpha_values'])
        axis.axvline(1, color='0.5', linewidth=0.8, linestyle=':')
        axis.grid(alpha=0.15)
    for axis, metrics in [(batch_ax, ['batch_asw', 'ilisi']),
                          (bio_ax, ['celltype_asw', 'clisi'])]:
        handles = [Line2D([], [], color=METRIC_STYLES[m][1], marker=METRIC_STYLES[m][2],
                          linestyle='', label=METRIC_STYLES[m][0]) for m in metrics]
        axis.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.23),
                    ncol=2, frameon=False, fontsize=FONT_SIZE - 1, handletextpad=0.4)
    handles = [Line2D([], [], color='0.2', linestyle=style, label=label + ' VAE + diffusion')
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
    save(fig, figures, "latent_predictability")

    fig = plt.figure(figsize=DOSE_SIZE)
    draw_dose_response(fig, dose, summary, dose_metadata)
    save(fig, figures, "dose_response")

    fig = plt.figure(figsize=COMPOSITE_SIZE)
    latent_row, dose_row = fig.subfigures(2, 1, height_ratios=[4, 5.5], hspace=ROW_GAP)
    draw_predictability(latent_row, latent, batch_label)
    draw_dose_response(dose_row, dose, summary, dose_metadata)
    latent_row.text(0.5, -0.035, '(a) Latent-subspace predictability',
                    ha='center', va='bottom', fontsize=FONT_SIZE + 1)
    dose_row.text(0.5, 0.015, '(b) Batch-intervention dose response',
                  ha='center', va='bottom', fontsize=FONT_SIZE + 1)
    save(fig, figures, "structured_batch_intervention")


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
