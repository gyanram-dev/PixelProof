import os
import sys
import logging
import shutil
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def download_cifake(
    target_dir: str = "data/tmp_cifake",
    max_ai_samples: int = 500,
    max_real_samples: int = 500
) -> Optional[Path]:
    """Download CIFAKE dataset using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        return None
    
    logger.info("Downloading CIFAKE dataset...")
    logger.info(f"  Target: {max_ai_samples} AI + {max_real_samples} real images")
    
    try:
        dataset_path = kagglehub.dataset_download(
            "birdy654/cifake-real-and-ai-generated-synthetic-images",
            path=target_dir
        )
        
        dataset_path = Path(dataset_path)
        logger.info(f"Downloaded to: {dataset_path}")
        
        return dataset_path
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def find_folders(dataset_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate AI-generated and real image folders in dataset."""
    ai_folder = None
    real_folder = None
    
    search_patterns = [
        ("**/fake*", "**/FAKE*"),
        ("**/ai*", "**/AI*"),
        ("**/synthetic*", "**/SYNTHETIC*"),
    ]
    
    for pattern in search_patterns[0]:
        matches = list(dataset_path.glob(pattern))
        if matches and matches[0].is_dir():
            ai_folder = matches[0]
            break
    
    real_matches = list(dataset_path.glob("**/real*")) + list(dataset_path.glob("**/REAL*"))
    if real_matches and real_matches[0].is_dir():
        real_folder = real_matches[0]
    
    if not ai_folder or not real_folder:
        logger.warning("Could not auto-detect folders. Checking subdirectories...")
        
        for item in dataset_path.iterdir():
            if item.is_dir():
                name_lower = item.name.lower()
                if "fake" in name_lower or "ai" in name_lower or "synthetic" in name_lower:
                    if not ai_folder:
                        ai_folder = item
                elif "real" in name_lower or "original" in name_lower:
                    if not real_folder:
                        real_folder = item
    
    return ai_folder, real_folder


class DatasetSetup:
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    def __init__(self, raw_dir: str = "data/raw", skip_duplicates: bool = True):
        self.raw_dir = Path(raw_dir)
        self.skip_duplicates = skip_duplicates
        self.existing_hashes: set = set()
        
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Load existing hashes to prevent duplicates."""
        metadata_file = self.raw_dir / "ingestion_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                    self.existing_hashes = set(meta.get("hashes", []))
                    logger.info(f"Loaded {len(self.existing_hashes)} existing hashes")
            except Exception:
                pass
    
    def _compute_hash(self, file_path: Path) -> Optional[str]:
        """Compute SHA256 hash."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if image is valid."""
        try:
            with Image.open(file_path) as img:
                img.verify()
                img.load()
                if img.size[0] < 32 or img.size[1] < 32:
                    return False
            return True
        except Exception:
            return False
    
    def _copy_with_progress(
        self, 
        source_files: list, 
        dest_dir: Path, 
        label: str,
        max_samples: int
    ) -> Tuple[int, int]:
        """Copy files with progress logging."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        total = min(len(source_files), max_samples)
        copied = 0
        skipped = 0
        
        logger.info(f"  Copying {total} {label} images to {dest_dir.name}/")
        
        for i, f in enumerate(source_files[:max_samples]):
            if not self._is_valid_image(f):
                logger.debug(f"Invalid: {f.name}")
                skipped += 1
                continue
            
            file_hash = self._compute_hash(f)
            
            if self.skip_duplicates and file_hash in self.existing_hashes:
                skipped += 1
                continue
            
            dest_file = dest_dir / f.name
            
            if dest_file.exists():
                stem = dest_file.stem
                suffix = dest_file.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(f, dest_file)
            
            if file_hash:
                self.existing_hashes.add(file_hash)
            
            copied += 1
            
            if copied % 100 == 0 or copied == total:
                logger.info(f"    [{copied}/{total}] {label} images copied")
        
        logger.info(f"  Copied {copied} {label} images")
        
        return copied, skipped
    
    def run(
        self, 
        dataset_path: Path, 
        max_ai_samples: int = 500, 
        max_real_samples: int = 500
    ) -> Dict[str, int]:
        """Copy dataset to raw folder."""
        logger.info("=" * 60)
        logger.info("Setting up dataset from CIFAKE")
        logger.info("=" * 60)
        
        ai_folder, real_folder = find_folders(dataset_path)
        
        if not ai_folder:
            logger.error("Could not find AI-generated/fake/synthetic folder")
            return {}
        if not real_folder:
            logger.error("Could not find real/original folder")
            return {}
        
        logger.info(f"Found folders:")
        logger.info(f"  AI-generated: {ai_folder}")
        logger.info(f"  Real: {real_folder}")
        
        ai_files = [
            f for f in ai_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS
        ]
        real_files = [
            f for f in real_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS
        ]
        
        ai_files = sorted(ai_files)[:max_ai_samples]
        real_files = sorted(real_files)[:max_real_samples]
        
        logger.info(f"Source files: {len(ai_files)} AI, {len(real_files)} real")
        
        ai_dest = self.raw_dir / "ai_generated"
        real_dest = self.raw_dir / "real"
        
        ai_copied, ai_skipped = self._copy_with_progress(
            ai_files, ai_dest, "AI-generated (fake)", max_ai_samples
        )
        
        real_copied, real_skipped = self._copy_with_progress(
            real_files, real_dest, "real", max_real_samples
        )
        
        self._save_metadata()
        
        results = {
            "ai_copied": ai_copied,
            "ai_skipped": ai_skipped,
            "real_copied": real_copied,
            "real_skipped": real_skipped
        }
        
        self._log_summary(results)
        
        return results
    
    def _save_metadata(self):
        """Save metadata for future runs."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "CIFAKE",
            "hashes": list(self.existing_hashes),
            "source": "kagglehub"
        }
        
        with open(self.raw_dir / "ingestion_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _log_summary(self, results: Dict[str, int]):
        """Log final summary."""
        logger.info("=" * 60)
        logger.info("DATASET SETUP COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  AI-generated copied: {results.get('ai_copied', 0)}")
        logger.info(f"  AI-generated skipped: {results.get('ai_skipped', 0)}")
        logger.info(f"  Real copied: {results.get('real_copied', 0)}")
        logger.info(f"  Real skipped: {results.get('real_skipped', 0)}")
        
        total_copied = results.get('ai_copied', 0) + results.get('real_copied', 0)
        total_skipped = results.get('ai_skipped', 0) + results.get('real_skipped', 0)
        
        logger.info(f"  Total copied: {total_copied}")
        logger.info(f"  Total skipped: {total_skipped}")
        logger.info(f"  Dataset location: {self.raw_dir}")
        logger.info("=" * 60)


def cleanup_temp(dataset_path: Path, temp_dir: str = "data/tmp_cifake"):
    """Clean up temporary download folder."""
    temp_path = Path(temp_dir)
    
    if temp_path.exists() and dataset_path in temp_path.glob("*"):
        try:
            import shutil
            shutil.rmtree(temp_path)
            logger.info(f"Cleaned up temporary folder: {temp_path}")
        except Exception as e:
            logger.warning(f"Could not cleanup temp folder: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and setup CIFAKE dataset")
    parser.add_argument(
        "--max-ai-samples", 
        type=int, 
        default=500, 
        help="Maximum AI-generated images to download"
    )
    parser.add_argument(
        "--max-real-samples", 
        type=int, 
        default=500, 
        help="Maximum real images to download"
    )
    parser.add_argument(
        "--raw-dir", 
        type=str, 
        default="data/raw", 
        help="Raw data destination"
    )
    parser.add_argument(
        "--keep-temp", 
        action="store_true", 
        help="Keep temporary download folder"
    )
    args = parser.parse_args()
    
    logger.info("PixelProof Dataset Download & Setup")
    logger.info(f"Target: {args.max_ai_samples} AI + {args.max_real_samples} real images")
    
    dataset_path = download_cifake(
        max_ai_samples=args.max_ai_samples,
        max_real_samples=args.max_real_samples
    )
    
    if not dataset_path:
        logger.error("Failed to download dataset")
        sys.exit(1)
    
    setup = DatasetSetup(args.raw_dir)
    results = setup.run(
        dataset_path, 
        args.max_ai_samples, 
        args.max_real_samples
    )
    
    if not args.keep_temp:
        cleanup_temp(dataset_path)
    
    print("\n" + "=" * 40)
    print("DOWNLOAD COMPLETE")
    print("=" * 40)
    print(f"AI-generated: {results.get('ai_copied', 0)} images")
    print(f"Real: {results.get('real_copied', 0)} images")
    print(f"Location: {args.raw_dir}")
    print("=" * 40)


if __name__ == "__main__":
    main()