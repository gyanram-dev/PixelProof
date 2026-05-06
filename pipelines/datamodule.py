import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class TensorDataset(Dataset):
    def __init__(self, data_dir: str, label: str, transform: Optional[Callable] = None):
        self.data_dir = Path(data_dir) / label
        self.transform = transform
        self.files = sorted([f for f in self.data_dir.iterdir() if f.suffix == ".pt"])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        tensor = torch.load(self.files[idx])
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor


class PixelProofDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        
        self.files = []
        self.labels = []
        
        for label_idx, label in enumerate(["ai_generated", "real"]):
            label_dir = self.data_dir / label
            if label_dir.exists():
                files = sorted([f for f in label_dir.iterdir() if f.suffix == ".pt"])
                self.files.extend(files)
                self.labels.extend([label_idx] * len(files))
        
        logger.info(f"Loaded {len(self.files)} images for {split}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        tensor = torch.load(self.files[idx])
        label = self.labels[idx]
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, label


def create_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
):
    """Create train, val, and test dataloaders."""
    
    train_dataset = PixelProofDataset(data_dir, "train", train_transform)
    val_dataset = PixelProofDataset(data_dir, "val", val_transform)
    test_dataset = PixelProofDataset(data_dir, "test", val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_counts(data_dir: str = "data"):
    """Get class distribution counts."""
    counts = {}
    
    for split in ["train", "val", "test"]:
        counts[split] = {}
        
        for label in ["ai_generated", "real"]:
            label_dir = Path(data_dir) / split / label
            
            if label_dir.exists():
                files = list(label_dir.iterdir())
                counts[split][label] = len([f for f in files if f.suffix == ".pt"])
            else:
                counts[split][label] = 0
    
    return counts