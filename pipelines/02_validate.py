import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


class ImageValidator:
    def __init__(self, data_dir: str, min_size: int = 32):
        self.data_dir = Path(data_dir)
        self.min_size = min_size
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        
    def is_valid_image(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if an image file is valid and readable."""
        if file_path.suffix.lower() not in self.valid_extensions:
            return False, "invalid_extension"
        
        try:
            with Image.open(file_path) as img:
                img.verify()
            
            with Image.open(file_path) as img:
                width, height = img.size
                if width < self.min_size or height < self.min_size:
                    return False, "too_small"
                
                if img.mode not in ["RGB", "L", "RGBA"]:
                    return False, "unsupported_mode"
                    
                img.load()
                
        except Exception as e:
            return False, f"corrupted: {str(e)}"
        
        return True, None
    
    def validate_directory(self, label: str) -> List[Path]:
        """Validate all images in a directory."""
        source_dir = self.data_dir / label
        
        if not source_dir.exists():
            logger.warning(f"Directory not found: {source_dir}")
            return []
        
        files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in self.valid_extensions]
        
        valid_files = []
        invalid_count = 0
        invalid_reasons = {}
        
        for f in files:
            is_valid, reason = self.is_valid_image(f)
            if is_valid:
                valid_files.append(f)
            else:
                invalid_count += 1
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                logger.debug(f"Invalid: {f.name} - {reason}")
        
        logger.info(f"Validated {label}: {len(valid_files)}/{len(files)} valid")
        
        if invalid_reasons:
            logger.info(f"  Invalid reasons: {invalid_reasons}")
        
        return valid_files
    
    def run(self, quarantine: bool = True) -> dict:
        """Run validation on both classes."""
        logger.info("Starting image validation")
        
        results = {}
        
        for label in ["ai_generated", "real"]:
            valid_files = self.validate_directory(label)
            results[label] = valid_files
            
            if quarantine:
                quarantine_dir = self.data_dir / f"quarantine_{label}"
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                
                files = list((self.data_dir / label).iterdir())
                for f in files:
                    if f not in valid_files:
                        dest = quarantine_dir / f.name
                        f.rename(dest)
                        logger.debug(f"Quarantined: {f.name}")
        
        total_valid = sum(len(v) for v in results.values())
        logger.info(f"Validation complete: {total_valid} total valid images")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate dataset images")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory to validate")
    parser.add_argument("--min-size", type=int, default=32, help="Minimum image dimension")
    parser.add_argument("--no-quarantine", action="store_true", help="Don't move invalid images to quarantine")
    args = parser.parse_args()
    
    validator = ImageValidator(args.data_dir, args.min_size)
    results = validator.run(quarantine=not args.no_quarantine)
    
    print(f"\nValidation Summary:")
    for label, files in results.items():
        print(f"  {label}: {len(files)} valid images")


if __name__ == "__main__":
    main()