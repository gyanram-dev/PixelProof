import os
import sys

# Windows OpenMP fix MUST happen before torch import
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from pipelines.env_setup import setup_environment
from models.cnn_model import get_model

# Initialize environment for Windows stability
setup_environment()

# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/training_pipeline.log", mode="a")
    ]
)
logger = logging.getLogger("PixelProof.Train")

class PixelProofTensorDataset(Dataset):
    """
    Custom Dataset for loading .pt tensor files from split directories.
    Expected structure: <root>/<split>/<class_name>/*.pt
    """
    def __init__(self, root_dir: Path, split: str):
        self.root_dir = Path(root_dir) / split
        self.classes = ["ai_generated", "real"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            files = list(class_dir.glob("*.pt"))
            for f in files:
                self.samples.append((f, self.class_to_idx[class_name]))
                
        logger.info(f"Loaded {len(self.samples)} samples for split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        tensor = torch.load(path, weights_only=True)
        # Ensure tensor is in correct format (3, H, W) and float32
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        return tensor, torch.tensor([label], dtype=torch.float32)

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, output_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.best_val_loss = float('inf')
        self.metrics_history = {
            "train_loss": [], "val_loss": [],
            "accuracy": [], "precision": [], "recall": []
        }

    def _calculate_metrics(self, y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> Dict[str, float]:
        """Calculate binary classification metrics."""
        y_pred = torch.sigmoid(y_pred_logits) > 0.5
        y_true = y_true.bool()
        
        tp = (y_pred & y_true).sum().item()
        tn = (~y_pred & ~y_true).sum().item()
        fp = (y_pred & ~y_true).sum().item()
        fn = (~y_pred & y_true).sum().item()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return running_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_outputs = []
        
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())
            
        avg_loss = running_loss / len(dataloader)
        metrics = self._calculate_metrics(torch.cat(all_targets), torch.cat(all_outputs))
        metrics["val_loss"] = avg_loss
        
        return metrics

    def run(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int = 5):
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        epochs_no_improve = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.metrics_history["train_loss"].append(train_loss)
            self.metrics_history["val_loss"].append(val_metrics["val_loss"])
            self.metrics_history["accuracy"].append(val_metrics["accuracy"])
            self.metrics_history["precision"].append(val_metrics["precision"])
            self.metrics_history["recall"].append(val_metrics["recall"])
            
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
                logger.info(f"New best model saved at epoch {epoch}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            # Early stopping
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
                
        # Save metrics
        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=4)
        logger.info(f"Training history saved to {self.output_dir / 'training_metrics.json'}")

def main():
    parser = argparse.ArgumentParser(description="PixelProof CNN Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/splits", help="Path to splits directory")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Where to save results")
    parser.add_argument("--epochs", type=int, default=20, help="Max number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    train_ds = PixelProofTensorDataset(Path(args.data_dir), "train")
    val_ds = PixelProofTensorDataset(Path(args.data_dir), "val")
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Empty datasets. Please run 04_split_dataset.py first.")
        sys.exit(1)
        
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers=0 for Windows stability
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model and trainer
    model = get_model()
    trainer = Trainer(model, device, Path(args.output_dir))
    
    # Run training
    trainer.run(train_loader, val_loader, args.epochs, args.patience)

if __name__ == "__main__":
    main()
