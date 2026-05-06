import os
import sys
import logging
import json
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import random
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    def __init__(
        self, 
        source_dir: str, 
        train_dir: str, 
        val_dir: str, 
        test_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        self.source_dir = Path(source_dir)
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
    def _get_image_files(self, label: str) -> List[Path]:
        """Get all image files in a label directory."""
        source = self.source_dir / label
        if not source.exists():
            return []
        
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".pt"}
        return [f for f in source.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    
    def _split_files(
        self, 
        files: List[Path], 
        train_ratio: float, 
        val_ratio: float
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split files with deterministic hashing."""
        random.seed(self.seed)
        random.shuffle(files)
        
        total = len(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        return train_files, val_files, test_files
    
    def _copy_files(self, files: List[Path], dest_dir: Path) -> int:
        """Copy files to destination directory."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        for f in files:
            dest = dest_dir / f.name
            
            if dest.exists():
                file_hash = hashlib.md5(f.read_bytes()).hexdigest()[:8]
                dest = dest_dir / f"{f.stem}_{file_hash}{f.suffix}"
            
            shutil.copy2(f, dest)
            copied += 1
        
        return copied
    
    def split_label(self, label: str, min_samples: int = 10) -> dict:
        """Split a single label's files."""
        files = self._get_image_files(label)
        
        if len(files) < min_samples:
            logger.warning(f"{label}: only {len(files)} files (min: {min_samples}), skipping")
            return {"train": 0, "val": 0, "test": 0}
        
        train_files, val_files, test_files = self._split_files(
            files, self.train_ratio, self.val_ratio
        )
        
        train_dest = self.train_dir / label
        val_dest = self.val_dir / label
        test_dest = self.test_dir / label
        
        train_count = self._copy_files(train_files, train_dest)
        val_count = self._copy_files(val_files, val_dest)
        test_count = self._copy_files(test_files, test_dest)
        
        logger.info(
            f"Split {label}: train={train_count}, val={val_count}, test={test_count}"
        )
        
        return {
            "train": train_count,
            "val": val_count,
            "test": test_count
        }
    
    def run(self) -> dict:
        """Run split on both classes."""
        logger.info(
            f"Starting dataset split "
            f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
        )
        
        results = {}
        
        for label in ["ai_generated", "real"]:
            results[label] = self.split_label(label)
        
        total = {
            split: sum(results[label][split] for label in results)
            for split in ["train", "val", "test"]
        }
        
        meta = {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "splits": results,
            "total": total
        }
        
        with open(self.train_dir.parent / "split_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Split complete: {total}")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--source", type=str, default="data/processed", help="Source directory")
    parser.add_argument("--train-dir", type=str, default="data/train", help="Train directory")
    parser.add_argument("--val-dir", type=str, default="data/val", help="Validation directory")
    parser.add_argument("--test-dir", type=str, default="data/test", help="Test directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    splitter = DatasetSplitter(
        args.source,
        args.train_dir,
        args.val_dir,
        args.test_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    results = splitter.run()
    
    print(f"\nSplit Summary:")
    for label, counts in results.items():
        print(f"  {label}: train={counts['train']}, val={counts['val']}, test={counts['test']}")


if __name__ == "__main__":
    main()