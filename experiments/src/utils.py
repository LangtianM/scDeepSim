import os
import numpy as np
import scanpy as sc
import torch
import pytorch_lightning as pl
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.dataset import ScDataModule


def load_and_preprocess(path, n_cells, n_genes, min_genes=10, min_cells=2, seed=42):
    """Load anndata object from h5ad file, subsample n_cells, select n_genes HVGs, 
    normalize + log1p, and return the processed anndata object."""
    np.random.seed(seed)
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    idx = np.random.choice(adata.n_obs, n_cells, replace=False)
    adata = adata[idx]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata
