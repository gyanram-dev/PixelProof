import sys
import os
from pathlib import Path
import torch
import streamlit as st

# Add the project root to sys.path to allow importing from 'models'
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.cnn_model import get_model

@st.cache_resource
def load_pixelproof_model(model_path: str, device: torch.device):
    """
    Loads and caches the PixelProof model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
    model = get_model()
    # Use weights_only=True for security
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
