import numpy as np
import pandas as pd


def batch_directions(X_encoded, batch_labels, ref_batch = None, cell_types = None):
    """Calculate the mean shift across different batches comparing to a reference batch. For each batch, it first finds the mean shift of each cell type to the reference batch, then averages the mean shifts across cell types.

    Parameters
    ----------
    X_encoded : np.ndarray
        The encoded data. (n_cells, latent_dim)
    batch_labels : array-like
        The batch label. (n_cells,)
    ref_batch : str, optional
        The reference batch. The default is None, in which case the first batch is used as the reference.
    cell_types : array-like, optional
        The cell types array with shape (n_cells,). The default is None, in which case the function find the grand mean shift.
        
    Returns
    -------
    pd.DataFrame: A dataframe with the mean shift for each batch, column names are the batch labels.
    
    Notes
    -----
    If there's no matched cell type in both batches, the mean shift is defined as the grand mean shift.
    """
    # Convert inputs to numpy arrays
    X_encoded = np.asarray(X_encoded)
    batch_labels = np.asarray(batch_labels)
    
    # Get unique batches
    unique_batches = np.unique(batch_labels)
    
    # Set reference batch
    if ref_batch is None:
        ref_batch = unique_batches[0]
    
    if ref_batch not in unique_batches:
        raise ValueError(f"Reference batch '{ref_batch}' not found in batch_labels")
    
    # Get reference batch mask
    ref_mask = batch_labels == ref_batch
    X_ref = X_encoded[ref_mask]
    
    # Initialize result dictionary
    batch_shifts = {}
    
    # If cell types are provided, compute cell-type-specific shifts
    if cell_types is not None:
        cell_types = np.asarray(cell_types)
        
        # Get cell types in reference batch
        ref_cell_types = cell_types[ref_mask]
        unique_cell_types = np.unique(ref_cell_types)
        
        # For each batch (excluding reference)
        for batch in unique_batches:
            if batch == ref_batch:
                # Reference batch has zero shift
                batch_shifts[batch] = np.zeros(X_encoded.shape[1])
                continue
            
            batch_mask = batch_labels == batch
            X_batch = X_encoded[batch_mask]
            batch_cell_types = cell_types[batch_mask]
            
            # Collect shifts for each cell type
            cell_type_shifts = []
            
            for ct in unique_cell_types:
                # Check if this cell type exists in both batches
                ref_ct_mask = ref_cell_types == ct
                batch_ct_mask = batch_cell_types == ct
                
                if ref_ct_mask.sum() > 0 and batch_ct_mask.sum() > 0:
                    # Compute mean shift for this cell type
                    ref_ct_mean = X_ref[ref_ct_mask].mean(axis=0)
                    batch_ct_mean = X_batch[batch_ct_mask].mean(axis=0)
                    shift = batch_ct_mean - ref_ct_mean
                    cell_type_shifts.append(shift)
            
            # Average shifts across cell types
            if len(cell_type_shifts) > 0:
                batch_shifts[batch] = np.mean(cell_type_shifts, axis=0)
            else:
                # No matched cell types, use grand mean shift
                ref_mean = X_ref.mean(axis=0)
                batch_mean = X_batch.mean(axis=0)
                batch_shifts[batch] = batch_mean - ref_mean
    else:
        # No cell types provided, compute grand mean shifts
        ref_mean = X_ref.mean(axis=0)
        
        for batch in unique_batches:
            if batch == ref_batch:
                batch_shifts[batch] = np.zeros(X_encoded.shape[1])
            else:
                batch_mask = batch_labels == batch
                X_batch = X_encoded[batch_mask]
                batch_mean = X_batch.mean(axis=0)
                batch_shifts[batch] = batch_mean - ref_mean
    
    # Convert to DataFrame
    return pd.DataFrame(batch_shifts)
    
    
    

    