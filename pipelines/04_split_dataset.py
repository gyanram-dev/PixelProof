import os
import sys
import logging
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from pipelines.env_setup import setup_environment

# Initialize environment
setup_environment()

# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/split_pipeline.log", mode="a")
    ]
)
logger = logging.getLogger("PixelProof.Split")

class DatasetSplitter:
    """
    Production-ready dataset splitter for PixelProof.
    Handles class-balanced splitting, leakage detection, and metadata generation.
    """
    CLASSES = ["ai_generated", "real"]
    
    def __init__(
        self,
        source_dir: Path,
        output_base_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        self.source_dir = Path(source_dir)
        self.output_base_dir = Path(output_base_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Define split paths
        self.split_paths = {
            "train": self.output_base_dir / "train",
            "val": self.output_base_dir / "val",
            "test": self.output_base_dir / "test"
        }
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0 (currently {total_ratio})")

    def _get_files(self, class_name: str) -> List[Path]:
        """Get all relevant files for a specific class."""
        class_path = self.source_dir / class_name
        if not class_path.exists():
            logger.warning(f"Class directory not found: {class_path}")
            return []
        
        # Include common image formats and .pt files (tensors)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pt"}
        files = [
            f for f in class_path.iterdir() 
            if f.is_file() and f.suffix.lower() in extensions
        ]
        return sorted(files)

    def _detect_leakage(self, splits: Dict[str, List[Path]]) -> bool:
        """Verify that no file exists in more than one split."""
        logger.info("Verifying split integrity (leakage detection)...")
        
        seen_files: Dict[str, str] = {} # filename -> split_name
        has_leakage = False
        
        for split_name, files in splits.items():
            for f in files:
                if f.name in seen_files:
                    logger.error(f"LEAKAGE DETECTED: File '{f.name}' found in both '{seen_files[f.name]}' and '{split_name}'")
                    has_leakage = True
                seen_files[f.name] = split_name
                
        return not has_leakage

    def split_dataset(self):
        """Execute the splitting process."""
        logger.info(f"Starting split: Train={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={self.test_ratio:.0%}")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_base_dir}")
        
        random.seed(self.seed)
        
        split_metadata = {
            "config": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "seed": self.seed
            },
            "classes": {},
            "total": {"train": 0, "val": 0, "test": 0}
        }
        
        all_split_files: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

        for class_name in self.CLASSES:
            files = self._get_files(class_name)
            if not files:
                continue
                
            logger.info(f"Processing class '{class_name}': {len(files)} files found")
            
            # Shuffle files deterministically
            random.shuffle(files)
            
            # Calculate split indices
            n_total = len(files)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            
            # Distribute files
            class_splits = {
                "train": files[:n_train],
                "val": files[n_train:n_train + n_val],
                "test": files[n_train + n_val:]
            }
            
            # Update global splits for leakage detection
            for s_name, s_files in class_splits.items():
                all_split_files[s_name].extend(s_files)
                
            # Store metadata for this class
            split_metadata["classes"][class_name] = {
                "total": n_total,
                "train": len(class_splits["train"]),
                "val": len(class_splits["val"]),
                "test": len(class_splits["test"])
            }
            
            # Copy files with progress tracking
            for split_name, s_files in class_splits.items():
                dest_dir = self.split_paths[split_name] / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"  Copying {len(s_files)} files to {split_name}/{class_name}...")
                for f in tqdm(s_files, desc=f"{class_name} -> {split_name}", unit="file"):
                    shutil.copy2(f, dest_dir / f.name)
                    split_metadata["total"][split_name] += 1

        # Verify no leakage
        if not self._detect_leakage(all_split_files):
            logger.error("Split failed: Leakage detected between splits.")
            return False

        # Save metadata
        meta_path = self.output_base_dir / "split_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(split_metadata, f, indent=4)
        
        logger.info(f"Metadata saved to {meta_path}")
        logger.info("Dataset splitting completed successfully.")
        logger.info(f"Final counts: {split_metadata['total']}")
        return True

def main():
    parser = argparse.ArgumentParser(description="PixelProof Production Dataset Splitter")
    parser.add_argument("--source", type=str, default="data/processed", help="Source directory with class subfolders")
    parser.add_argument("--output", type=str, default="data/splits", help="Base output directory for splits")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio (0.0-1.0)")
    parser.add_argument("--val", type=float, default=0.15, help="Val ratio (0.0-1.0)")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing splits")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if output_dir.exists() and not args.force:
        logger.error(f"Output directory {output_dir} already exists. Use --force to overwrite.")
        sys.exit(1)
    elif output_dir.exists() and args.force:
        logger.warning(f"Overwriting existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    try:
        splitter = DatasetSplitter(
            source_dir=source_dir,
            output_base_dir=output_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed
        )
        if splitter.split_dataset():
            print("\nDataset Split Successful!")
            print(f"Results available in: {output_dir}")
        else:
            print("\nDataset Split Failed. Check logs.")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"An error occurred during splitting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
