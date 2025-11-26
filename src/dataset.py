from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # pyright: ignore[reportMissingImports]
import scipy.sparse as sp
import torch
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
    def __init__(self, adata, label_key, encoder = "LabelEncoder"):
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
    def __init__(self, adata, label_key, encoder = "LabelEncoder", batch_size=128, val_split=0.1):
        super().__init__()
        self.adata = adata
        self.label_key = label_key
        self.batch_size = batch_size
        self.val_split = val_split
        self.encoder = encoder

    def setup(self, stage=None):
        full = ScDataset(self.adata, self.label_key, self.encoder)
        val_size = int(len(full) * self.val_split)
        train_size = len(full) - val_size
        self.train_dataset, self.val_dataset = random_split(full, [train_size, val_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)