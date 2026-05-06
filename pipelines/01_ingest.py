import os
import sys
import logging
import shutil
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict
from datetime import datetime
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


class DatasetIngestion:
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    HASH_CHUNK_SIZE = 8192
    
    def __init__(self, source_dir: str, raw_dir: str, skip_duplicates: bool = True):
        self.source_dir = Path(source_dir)
        self.raw_dir = Path(raw_dir)
        self.skip_duplicates = skip_duplicates
        
        self.hash_cache: Set[str] = set()
        self.stats: Dict[str, int] = {
            "scanned": 0,
            "copied": 0,
            "duplicates": 0,
            "corrupted": 0,
            "skipped": 0,
            "errors": 0
        }
        
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Load existing file hashes to detect duplicates across runs."""
        metadata_file = self.raw_dir / "ingestion_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                    self.hash_cache = set(meta.get("hashes", []))
                    logger.info(f"Loaded {len(self.hash_cache)} existing file hashes")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
    
    def _compute_hash(self, file_path: Path) -> Optional[str]:
        """Compute SHA256 hash of file for duplicate detection."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(self.HASH_CHUNK_SIZE):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    def _is_valid_image(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if image is valid and not corrupted."""
        try:
            with Image.open(file_path) as img:
                img.verify()
            
            with Image.open(file_path) as img:
                img.load()
                width, height = img.size
                
                if width < 32 or height < 32:
                    return False, "too_small"
                
                if img.mode not in ["RGB", "L", "RGBA", "P"]:
                    return False, "unsupported_mode"
            
            return True, None
            
        except Exception as e:
            return False, str(type(e).__name__)
    
    def _copy_file(self, source: Path, dest: Path, file_hash: str) -> bool:
        """Copy file with duplicate handling."""
        if self.skip_duplicates and file_hash in self.hash_cache:
            return False
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        if dest.exists():
            dest = self._get_unique_filename(dest)
        
        shutil.copy2(source, dest)
        self.hash_cache.add(file_hash)
        
        return True
    
    def _get_unique_filename(self, dest: Path) -> Path:
        """Generate unique filename if file exists."""
        counter = 1
        stem = dest.stem
        suffix = dest.suffix
        
        while dest.exists():
            dest = dest.parent / f"{stem}_{counter}{suffix}"
            counter += 1
        
        return dest
    
    def ingest(self, class_name: str, label: str, batch_size: int = 1000) -> int:
        """Ingest images from source directory to raw folder."""
        dest = self.ai_dest if label == "ai" else self.real_dest
        source = self.source_dir / class_name
        
        if not source.exists():
            logger.warning(f"Source directory not found: {source}")
            return 0
        
        files = [
            f for f in source.iterdir() 
            if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS
        ]
        
        self.stats["scanned"] += len(files)
        
        logger.info(f"Processing {len(files)} {label} images from {class_name}")
        
        copied = 0
        pbar = tqdm(files, desc=f"  {label}", unit="img")
        
        for f in pbar:
            is_valid, error = self._is_valid_image(f)
            
            if not is_valid:
                if error == "too_small":
                    self.stats["skipped"] += 1
                elif error in ["corrupted", "UnidentifiedImageError"]:
                    self.stats["corrupted"] += 1
                    logger.debug(f"Corrupted: {f.name} - {error}")
                else:
                    self.stats["skipped"] += 1
                continue
            
            file_hash = self._compute_hash(f)
            
            if not file_hash:
                self.stats["errors"] += 1
                continue
            
            dest_file = dest / f.name
            
            if self._copy_file(f, dest_file, file_hash):
                copied += 1
                self.stats["copied"] += 1
            else:
                self.stats["duplicates"] += 1
            
            pbar.set_postfix({"copied": copied, "dupes": self.stats["duplicates"]})
        
        logger.info(f"Ingested {copied} {label} images from {class_name}")
        return copied
    
    def _save_metadata(self):
        """Save ingestion metadata for future runs."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "hashes": list(self.hash_cache),
            "stats": self.stats
        }
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.raw_dir / "ingestion_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata with {len(self.hash_cache)} unique hashes")
    
    def run(self, ai_source: str, real_source: str) -> dict:
        """Run full ingestion pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Dataset Ingestion")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Destination: {self.raw_dir}")
        logger.info(f"Skip duplicates: {self.skip_duplicates}")
        logger.info("=" * 60)
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        counts = {
            "ai_generated": self.ingest(ai_source, "ai"),
            "real": self.ingest(real_source, "ai")
        }
        
        counts["real"] = self.ingest(real_source, "real")
        
        self._save_metadata()
        
        self._log_summary()
        
        return counts
    
    def _log_summary(self):
        """Log final ingestion summary."""
        logger.info("=" * 60)
        logger.info("Ingestion Complete")
        logger.info("=" * 60)
        logger.info(f"  Total scanned: {self.stats['scanned']}")
        logger.info(f"  Successfully copied: {self.stats['copied']}")
        logger.info(f"  Duplicates skipped: {self.stats['duplicates']}")
        logger.info(f"  Corrupted images: {self.stats['corrupted']}")
        logger.info(f"  Other skipped: {self.stats['skipped']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        
        copied = self.stats["copied"]
        scanned = self.stats["scanned"]
        
        if scanned > 0:
            success_rate = (copied / scanned) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")
        
        ai_count = len(list(self.raw_dir.glob("ai_generated/*")))
        real_count = len(list(self.raw_dir.glob("real/*")))
        
        logger.info(f"  Files now in raw/ai_generated: {ai_count}")
        logger.info(f"  Files now in raw/real: {real_count}")
        
        logger.info("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest dataset images with validation")
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Source directory containing ai_generated/ and real/ folders"
    )
    parser.add_argument(
        "--raw-dir", 
        type=str, 
        default="data/raw", 
        help="Raw data directory"
    )
    parser.add_argument(
        "--no-skip-duplicates", 
        action="store_true", 
        help="Don't skip duplicate images"
    )
    args = parser.parse_args()
    
    ingestion = DatasetIngestion(
        args.source, 
        args.raw_dir, 
        skip_duplicates=not args.no_skip_duplicates
    )
    stats = ingestion.run("ai_generated", "real")
    
    print("\n" + "=" * 40)
    print("INGESTION SUMMARY")
    print("=" * 40)
    for key, val in stats.items():
        print(f"  {key}: {val} images")


if __name__ == "__main__":
    main()