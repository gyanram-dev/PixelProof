import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Optional, List

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
    def __init__(self, source_dir: str, raw_dir: str):
        self.source_dir = Path(source_dir)
        self.raw_dir = Path(raw_dir)
        self.ai_dest = self.raw_dir / "ai_generated"
        self.real_dest = self.raw_dir / "real"

    def ingest(self, class_name: str, label: str) -> int:
        """Ingest images from source directory to raw folder."""
        dest = self.ai_dest if label == "ai" else self.real_dest
        source = self.source_dir / class_name
        
        if not source.exists():
            logger.warning(f"Source directory not found: {source}")
            return 0
        
        dest.mkdir(parents=True, exist_ok=True)
        
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        files = [
            f for f in source.iterdir() 
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        copied = 0
        for f in files:
            dest_file = dest / f.name
            if dest_file.exists():
                dest_file = dest / f"{f.stem}_{f.stat().st_nlink}{f.suffix}"
            shutil.copy2(f, dest_file)
            copied += 1
        
        logger.info(f"Ingested {copied} {label} images from {class_name}")
        return copied

    def run(self, ai_source: str, real_source: str) -> dict:
        """Run full ingestion pipeline."""
        logger.info("Starting dataset ingestion")
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "ai_generated": self.ingest(ai_source, "ai"),
            "real": self.ingest(real_source, "real")
        }
        
        total = sum(stats.values())
        logger.info(f"Ingestion complete: {total} total images")
        
        return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest dataset images")
    parser.add_argument("--source", type=str, required=True, help="Source directory containing ai_generated/ and real/ folders")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw data directory")
    args = parser.parse_args()
    
    ingestion = DatasetIngestion(args.source, args.raw_dir)
    stats = ingestion.run("ai_generated", "real")
    
    print(f"\nIngestion Summary:")
    for key, val in stats.items():
        print(f"  {key}: {val} images")


if __name__ == "__main__":
    main()