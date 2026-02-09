from torch.utils.data import Dataset, random_split, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # pyright: ignore[reportMissingImports]
import scipy.sparse as sp
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl  # pyright: ignore[reportMissingImports]


def to_tensor(X):
    t = torch.tensor(X, dtype=torch.float32)
    if t.dim() == 2 and t.size(1) == 1:
        if t.size(0) == 1:
            return t.view(1)
        return t
    return t.squeeze()


class ScDataset(Dataset):
    def __init__(self, adata, label_key, encoder="LabelEncoder"):
        super().__init__()
        self.adata = adata
        self.sparse = sp.issparse(adata.X)
        self.encoder = encoder
        raw_labels = adata.obs[label_key].values
        if self.encoder == "LabelEncoder":
            self.classes = LabelEncoder().fit_transform(raw_labels)
        elif self.encoder == "OneHotEncoder":
            self.classes = OneHotEncoder(sparse=False).fit_transform(
                raw_labels.reshape(-1, 1)
            )
        else:
            raise ValueError(f"Invalid encoder: {encoder}")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        """Get the idx-th row of cellxgene data and its label.

        Args:
            idx (_type_): _description_
        """
        x = self.adata.X[idx]
        if self.sparse:
            x = to_tensor(x.toarray())
        else:
            x = to_tensor(x)
        if self.encoder == "LabelEncoder":
            classes = torch.tensor(self.classes[idx], dtype=torch.long)
        else:
            classes = torch.tensor(self.classes[idx], dtype=torch.float32)
        return x, classes


class ScDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for single-cell data.
    
    Args:
        adata: AnnData object containing the data
        label_key: Key in adata.obs for labels
        encoder: "LabelEncoder" or "OneHotEncoder"
        batch_size: Batch size for DataLoader
        val_split: Fraction of data for validation
        balanced_sampling: If True, use weighted sampling to balance classes.
            This ensures each class is sampled with equal probability,
            which helps with imbalanced datasets (e.g., rare cell types).
    """
    
    def __init__(
        self, 
        adata, 
        label_key, 
        encoder="LabelEncoder", 
        batch_size=128, 
        val_split=0.2,
        balanced_sampling=False,
    ):
        super().__init__()
        self.adata = adata
        self.label_key = label_key
        self.batch_size = batch_size
        self.val_split = val_split
        self.encoder = encoder
        self.balanced_sampling = balanced_sampling

    def setup(self, stage=None):
        full = ScDataset(self.adata, self.label_key, self.encoder)
        val_size = int(len(full) * self.val_split)
        train_size = len(full) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full, [train_size, val_size]
        )
        
        # Store full dataset reference for balanced sampling
        self._full_dataset = full
        
        # Compute sample weights for balanced sampling
        if self.balanced_sampling:
            self._train_sampler = self._create_balanced_sampler(self.train_dataset)
        else:
            self._train_sampler = None
    
    def _create_balanced_sampler(self, subset):
        """
        Create a WeightedRandomSampler for balanced class sampling.
        
        Each sample's weight = 1 / (number of samples in its class)
        This ensures each class has equal probability of being sampled.
        """
        # Get the original dataset's class labels
        full_classes = self._full_dataset.classes
        
        # Get indices of samples in this subset
        indices = subset.indices
        
        # Get class labels for samples in this subset
        if isinstance(full_classes, np.ndarray) and full_classes.ndim > 1:
            # OneHotEncoder case: convert to class indices
            subset_classes = np.argmax(full_classes[indices], axis=1)
        else:
            # LabelEncoder case: already class indices
            subset_classes = np.array(full_classes)[indices]
        
        # Count samples per class in the subset
        unique_classes, class_counts = np.unique(subset_classes, return_counts=True)
        class_to_count = dict(zip(unique_classes, class_counts))
        
        # Compute weight for each sample: 1 / class_count
        sample_weights = np.array([
            1.0 / class_to_count[c] for c in subset_classes
        ])
        
        # Normalize weights (optional, but good practice)
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(subset),
            replacement=True  # Allow replacement for balanced sampling
        )
        
        return sampler

    def train_dataloader(self):
        if self.balanced_sampling and self._train_sampler is not None:
            # Use sampler for balanced sampling (cannot use shuffle with sampler)
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                sampler=self._train_sampler
            )
        else:
            # Default: random shuffle
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
