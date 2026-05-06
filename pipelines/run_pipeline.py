import logging
import sys
from pathlib import Path
import importlib.util

LOG_DIR = Path("data")
LOG_FILE = LOG_DIR / "pipeline.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline_step(script: Path, description: str, **kwargs):
    """Run a single pipeline step."""
    logger.info(f"=" * 60)
    logger.info(f"STEP: {description}")
    logger.info(f"Script: {script}")
    logger.info(f"=" * 60)
    
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    
    original_argv = sys.argv
    sys.argv = [str(script)]
    
    for key, value in kwargs.items():
        if value is not None:
            sys.argv.extend([f"--{key}", str(value)])
    
    try:
        spec.loader.exec_module(module)
        
        if hasattr(module, "main"):
            module.main()
        
        logger.info(f"Completed: {description}")
        return True
        
    except Exception as e:
        logger.error(f"Failed: {description} - {e}")
        return False
        
    finally:
        sys.argv = original_argv


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run full dataset pipeline")
    parser.add_argument("--source", type=str, help="Source directory for ingestion")
    parser.add_argument("--size", type=int, default=224, help="Image size for preprocessing")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion step")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation step")
    parser.add_argument("--skip-split", action="store_true", help="Skip split step")
    args = parser.parse_args()
    
    pipeline_dir = Path("pipelines")
    
    logger.info("Starting PixelProof Dataset Pipeline")
    
    steps = [
        (pipeline_dir / "01_ingest.py", "Ingestion", {"source": args.source or "data/raw"}),
        (pipeline_dir / "02_validate.py", "Validation", {"data_dir": "data/raw"}),
        (pipeline_dir / "03_preprocess.py", "Preprocessing", {"source": "data/raw", "dest": "data/processed", "size": args.size}),
        (pipeline_dir / "04_split.py", "Splitting", {
            "source": "data/processed",
            "train-dir": "data/train",
            "val-dir": "data/val", 
            "test-dir": "data/test",
            "train-ratio": args.train_ratio,
            "val-ratio": args.val_ratio,
            "test-ratio": args.test_ratio,
            "seed": args.seed
        }),
    ]
    
    results = {}
    
    for i, (script, description, kwargs) in enumerate(steps):
        step_num = i + 1
        
        if step_num == 1 and args.skip_ingest:
            logger.info(f"Skipping: {description}")
            continue
        if step_num == 2 and args.skip_validate:
            logger.info(f"Skipping: {description}")
            continue
        if step_num == 4 and args.skip_split:
            logger.info(f"Skipping: {description}")
            continue
            
        success = run_pipeline_step(script, description, **kwargs)
        results[description] = "PASS" if success else "FAIL"
        
        if not success:
            logger.error("Pipeline failed, stopping")
            break
    
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    
    for step, status in results.items():
        logger.info(f"  {step}: {status}")


if __name__ == "__main__":
    main()