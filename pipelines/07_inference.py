import os
import sys

# Windows OpenMP fix MUST happen before torch import
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from pipelines.env_setup import setup_environment
from models.cnn_model import get_model

# Initialize environment
setup_environment()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PixelProof.Inference")

class PixelProofInference:
    """
    Production inference class for PixelProof.
    Handles image loading, preprocessing, and model prediction.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    CLASSES = ["ai_generated", "real"]

    def __init__(self, model_path: str, img_size: int = 224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        # Build transform pipeline (identical to 03_preprocess.py)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
        
        # Load model
        self.model = get_model(input_size=img_size)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")

    def predict(self, image_path: str) -> Dict:
        """
        Predict if an image is real or AI generated.
        Returns: { 'class': str, 'confidence': float, 'raw_score': float }
        """
        try:
            path = Path(image_path)
            if not path.exists():
                return {"error": f"File not found: {image_path}"}

            # Load and preprocess
            with Image.open(path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                prob = torch.sigmoid(logits).item()

            # Class 0: ai_generated, Class 1: real
            # If prob > 0.5, it's 'real' (index 1)
            class_idx = 1 if prob > 0.5 else 0
            confidence = prob if class_idx == 1 else (1.0 - prob)

            return {
                "class": self.CLASSES[class_idx],
                "confidence": float(confidence),
                "raw_score": float(prob),
                "path": str(path)
            }

        except Exception as e:
            logger.error(f"Inference error for {image_path}: {e}")
            return {"error": str(e)}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PixelProof Single Image Inference")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    try:
        inferencer = PixelProofInference(args.model)
        result = inferencer.predict(args.image)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
            
        print("\n" + "="*30)
        print("PIXELPROOF INFERENCE RESULT")
        print("="*30)
        print(f"Image:      {result['path']}")
        print(f"Prediction: {result['class'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("="*30 + "\n")

    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
