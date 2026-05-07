import time
import torch
import torch.nn as nn
from typing import Dict, Union

def run_inference(model: nn.Module, input_tensor: torch.Tensor, device: torch.device) -> Dict[str, Union[str, float]]:
    """
    Runs inference on a preprocessed image tensor.
    Returns prediction results including class and confidence.
    """
    start_time = time.time()
    
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
        
    inference_time = time.time() - start_time
    
    # Class 0: ai_generated, Class 1: real
    if prob > 0.5:
        predicted_class = "REAL"
        confidence = prob
    else:
        predicted_class = "AI_GENERATED"
        confidence = 1.0 - prob
        
    return {
        "class": predicted_class,
        "confidence": confidence,
        "raw_score": prob,
        "inference_time_ms": inference_time * 1000
    }
