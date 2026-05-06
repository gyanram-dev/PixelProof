import os
import sys
import logging
import json
import signal
from pathlib import Path
from typing import Tuple, Optional

# Setup environment before any ML imports
from env_setup import setup_environment
setup_environment()

# Windows-compatible output

def log(message: str):
    """Immediate output with flush."""
    print(message)
    sys.stdout.flush()


log("=" * 50)
log("PixelProof Preprocessing Pipeline")
log("=" * 50)
log(f"Python: {sys.version.split()[0]}")
log(f"Platform: {sys.platform}")
log(f"Working dir: {os.getcwd()}")
log("=" * 50)

# Setup logging BEFORE imports
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "pipeline.log", mode="a")
    ]
)

log("Importing libraries...")

try:
    from PIL import Image
    log("PIL imported")
except Exception as e:
    log(f"ERROR importing PIL: {e}")
    sys.exit(1)

try:
    import torch
    log(f"Torch imported: {torch.__version__}")
except Exception as e:
    log(f"ERROR importing torch: {e}")
    sys.exit(1)

try:
    from torchvision import transforms
    log("Torchvision imported")
except Exception as e:
    log(f"ERROR importing torchvision: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    log("tqdm imported")
except Exception as e:
    log(f"ERROR importing tqdm: {e}")
    sys.exit(1)

log("All imports successful")
log("-" * 50)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Image processing timed out")


class ImagePreprocessor:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    TIMEOUT_SECONDS = 30

    def __init__(
        self, 
        source_dir: str, 
        dest_dir: str, 
        img_size: int = 224,
        timeout: int = 30
    ):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.img_size = img_size
        self.timeout = timeout
        
        # Build transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
        
        self.stats = {
            "total_scanned": 0,
            "processed": 0,
            "failed": 0,
            "skipped": 0
        }
        
        log(f"Preprocessor initialized: {img_size}x{img_size}, timeout={timeout}s")

    def is_valid_image(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if image is valid before processing."""
        if not file_path.exists():
            return False, "not_found"
        
        if file_path.stat().st_size == 0:
            return False, "empty_file"
        
        try:
            with Image.open(file_path) as img:
                img.verify()
            
            with Image.open(file_path) as img:
                width, height = img.size
                if width < 32 or height < 32:
                    return False, "too_small"
                if img.mode not in ["RGB", "L", "RGBA", "P"]:
                    return False, "unsupported_mode"
            
            return True, None
            
        except Exception as e:
            return False, str(e)

    def process_image(self, file_path: Path, dest_path: Path) -> bool:
        """Process single image with timeout."""
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
            logging.debug(f"Failed {file_path.name}: {e}")
            return False

    def process_directory(self, label: str) -> Tuple[int, int]:
        """Process all images in a directory."""
        source = self.source_dir / label
        dest = self.dest_dir / label
        
        if not source.exists():
            log(f"WARNING: Source directory not found: {source}")
            return 0, 0
        
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        
        files = [
            f for f in source.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        total = len(files)
        self.stats["total_scanned"] += total
        
        log(f"Found {total} images in {label}/")
        
        if total == 0:
            log(f"No images to process in {label}/")
            return 0, 0
        
        processed = 0
        failed = 0
        
        # Process with progress bar
        with tqdm(files, desc=f"{label}", unit="img", ncols=80) as pbar:
            for f in pbar:
                is_valid, error = self.is_valid_image(f)
                
                if not is_valid:
                    self.stats["skipped"] += 1
                    failed += 1
                    continue
                
                if self.process_image(f, dest / f.name):
                    processed += 1
                    self.stats["processed"] += 1
                else:
                    failed += 1
                    self.stats["failed"] += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "ok": processed,
                    "fail": failed
                })
        
        log(f"Completed {label}: {processed} ok, {failed} failed")
        
        return processed, failed

    def run(self) -> dict:
        """Run preprocessing on both classes."""
        log("=" * 50)
        log("Starting preprocessing")
        log(f"Source: {self.source_dir}")
        log(f"Dest: {self.dest_dir}")
        log("=" * 50)
        
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for label in ["ai_generated", "real"]:
            log(f"Processing {label}...")
            
            processed, failed = self.process_directory(label)
            
            results[label] = {
                "processed": processed,
                "failed": failed
            }
        
        # Save metadata
        meta = {
            "img_size": self.img_size,
            "mean": self.MEAN,
            "std": self.STD,
            "num_classes": 2,
            "class_to_idx": {
                "ai_generated": 0,
                "real": 1
            },
            "stats": self.stats
        }
        
        with open(self.dest_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        log("=" * 50)
        log("PREPROCESSING COMPLETE")
        log("=" * 50)
        
        total = self.stats["processed"]
        failed = self.stats["failed"]
        
        log(f"Total processed: {total}")
        log(f"Total failed: {failed}")
        
        if total > 0:
            rate = (total / (total + failed)) * 100
            log(f"Success rate: {rate:.1f}%")
        
        log("=" * 50)
        
        return results


def main():
    import argparse
    
    log("Parsing arguments...")
    
    parser = argparse.ArgumentParser(
        description="Preprocess dataset images"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw",
        help="Source directory"
    )
    
    parser.add_argument(
        "--dest",
        type=str,
        default="data/processed",
        help="Destination directory"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target image size"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout per image (seconds)"
    )
    
    args = parser.parse_args()
    
    log(f"Arguments: source={args.source}, dest={args.dest}, size={args.size}")
    
    log("Creating preprocessor...")
    
    preprocessor = ImagePreprocessor(
        args.source,
        args.dest,
        args.size,
        args.timeout
    )
    
    log("Running preprocessing...")
    
    results = preprocessor.run()
    
    log("=" * 50)
    log("PREPROCESSING SUMMARY")
    log("=" * 50)
    
    for label, stats in results.items():
        log(f"  {label}: {stats['processed']} processed, {stats['failed']} failed")
    
    log("=" * 50)
    log("Pipeline finished successfully")
    log("=" * 50)
    
    sys.exit(0)


if __name__ == "__main__":
    main()