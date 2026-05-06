import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torch
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    def __init__(self, source_dir: str, dest_dir: str, img_size: int = 224):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
        
    def process_image(self, file_path: Path, dest_path: Path) -> bool:
        """Process a single image and save as tensor."""
        try:
            with Image.open(file_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                tensor = self.transform(img)
                
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                tensor_path = dest_path.with_suffix(".pt")
                torch.save(tensor, tensor_path)
                
                return True
                
        except Exception as e:
            logger.debug(f"Failed to process {file_path.name}: {e}")
            return False
    
    def process_directory(self, label: str) -> Tuple[int, int]:
        """Process all images in a directory."""
        source = self.source_dir / label
        dest = self.dest_dir / label
        
        if not source.exists():
            logger.warning(f"Source directory not found: {source}")
            return 0, 0
        
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        files = [f for f in source.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        
        processed = 0
        failed = 0
        
        for f in files:
            if self.process_image(f, dest / f.name):
                processed += 1
            else:
                failed += 1
        
        return processed, failed
    
    def run(self) -> dict:
        """Run preprocessing on both classes."""
        logger.info(f"Starting preprocessing (size={self.img_size}x{self.img_size})")
        
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for label in ["ai_generated", "real"]:
            processed, failed = self.process_directory(label)
            results[label] = {"processed": processed, "failed": failed}
            logger.info(f"Processed {label}: {processed} success, {failed} failed")
        
        meta = {
            "img_size": self.img_size,
            "mean": self.MEAN,
            "std": self.STD,
            "num_classes": 2,
            "class_to_idx": {"ai_generated": 0, "real": 1}
        }
        
        with open(self.dest_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        total_processed = sum(r["processed"] for r in results.values())
        logger.info(f"Preprocessing complete: {total_processed} images")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess dataset images")
    parser.add_argument("--source", type=str, default="data/raw", help="Source directory")
    parser.add_argument("--dest", type=str, default="data/processed", help="Destination directory")
    parser.add_argument("--size", type=int, default=224, help="Target image size")
    args = parser.parse_args()
    
    preprocessor = ImagePreprocessor(args.source, args.dest, args.size)
    results = preprocessor.run()
    
    print(f"\nPreprocessing Summary:")
    for label, stats in results.items():
        print(f"  {label}: {stats['processed']} processed, {stats['failed']} failed")


if __name__ == "__main__":
    main()