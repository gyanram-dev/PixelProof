import os
import sys

# Windows OpenMP fix MUST happen before torch import
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from pipelines.env_setup import setup_environment
from models.cnn_model import get_model
import importlib.util

# Helper to import from scripts starting with numbers
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import PixelProofTensorDataset dynamically
train_module = import_from_path("train_model", Path(__file__).parent / "05_train_model.py")
PixelProofTensorDataset = train_module.PixelProofTensorDataset

# Initialize environment
setup_environment()

# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/evaluation.log", mode="a")
    ]
)
logger = logging.getLogger("PixelProof.Eval")

class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device, results_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.classes = ["ai_generated", "real"]

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        all_targets = []
        all_outputs = []
        all_probs = []

        logger.info("Starting evaluation on test set...")
        for inputs, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_targets = np.array(all_targets).flatten()
        all_outputs = np.array(all_outputs).flatten()
        all_probs = np.array(all_probs).flatten()

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(all_targets, all_outputs)),
            "precision": float(precision_score(all_targets, all_outputs)),
            "recall": float(recall_score(all_targets, all_outputs)),
            "f1_score": float(f1_score(all_targets, all_outputs)),
            "report": classification_report(all_targets, all_outputs, target_names=self.classes, output_dict=True)
        }

        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_outputs)
        self._plot_confusion_matrix(cm)
        
        # Save metrics
        metrics_path = self.results_dir / "evaluation_results.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation complete. Metrics saved to {metrics_path}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics

    def _plot_confusion_matrix(self, cm: np.ndarray):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix - PixelProof')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = self.results_dir / "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Confusion matrix plot saved to {plot_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PixelProof Model Evaluation")
    parser.add_argument("--test-dir", type=str, default="data/splits", help="Path to splits directory")
    parser.add_argument("--model-path", type=str, default="models/checkpoints/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--results-dir", type=str, default="models", help="Where to save results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = get_model()
    if not Path(args.model_path).exists():
        logger.error(f"Model not found at {args.model_path}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    
    # Load test dataset
    test_ds = PixelProofTensorDataset(Path(args.test_dir), "test")
    if len(test_ds) == 0:
        logger.error("Test dataset is empty.")
        sys.exit(1)
        
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    evaluator = Evaluator(model, device, Path(args.results_dir))
    evaluator.evaluate(test_loader)

if __name__ == "__main__":
    main()
