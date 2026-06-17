"""Compatibility re-exports for shared experiment utilities.

Older scripts import helpers from ``experiments.src.utils``. New code should
prefer importing from the narrower modules directly, but this file remains as a
stable aggregation point for existing entry points.
"""

import logging

from experiments.src.batch_control import (
    apply_direction,
    compute_batch_direction,
    compute_global_direction,
)
from experiments.src.common import (
    as_dense,
    decode_latents,
    encode_adata,
    encode_matrix,
    save_git_info,
    set_random_seed,
)
from experiments.src.data import (
    fit_label_encoder,
    load_and_preprocess,
    load_pancreas,
    prepare_celltype_batch_data,
)
from experiments.src.training import (
    batch_supervised_config,
    build_truncated_normal_vae,
    celltype_batch_supervised_config,
    celltype_supervised_config,
    selected_adversarial_config,
    train_batch_vae,
    train_celltype_batch_vae,
    train_celltype_vae,
    train_supervised_vae,
)

log = logging.getLogger(__name__)
