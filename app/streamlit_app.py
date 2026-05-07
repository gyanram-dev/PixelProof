import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import sys
import time
from PIL import Image
from pathlib import Path

# Add project root to sys.path for internal utility imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Fix Windows OpenMP issue
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Custom Utilities
from app.utils.preprocessing import preprocess_image
from app.utils.model_loader import load_pixelproof_model
from app.utils.inference import run_inference
from app.utils.styles import apply_custom_styles, render_hero, render_footer

# --- Page Config ---
st.set_page_config(
    page_title="PixelProof | AI Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize UI ---
apply_custom_styles()

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("<h2 style='color: #F0F6FC; margin-bottom: 0;'>🛡️ PixelProof</h2>", unsafe_allow_html=True)
        st.caption("Forensic Image Analysis")
        st.divider()
        
        st.markdown("### ⚙️ Engine Specs")
        st.markdown("""
        <div class="pixel-card">
            <p class="metric-label">ARCHITECTURE</p>
            <p class="metric-value">PixelProof CNN-v1</p>
            <p class="metric-label">INPUT DIMENSIONS</p>
            <p class="metric-value">224 × 224 × 3</p>
            <p class="metric-label">NORMALIZATION</p>
            <p class="metric-value">ImageNet Stats</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🧠 Supported Generators")
        st.markdown("""
        - Midjourney (v4, v5, v6)
        - Stable Diffusion (XL, 1.5)
        - DALL·E 3
        - Adobe Firefly
        """)
        
        st.divider()
        st.caption("Developed by AI Engineering | 2026")

    # --- Hero Section ---
    render_hero()
    
    # --- Model Loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/checkpoints/best_model.pth"
    
    try:
        model = load_pixelproof_model(model_path, device)
    except Exception as e:
        st.error(f"Engine initialization failed: {e}")
        return

    # --- Main Analysis Interface ---
    main_col1, main_col2 = st.columns([1, 1], gap="large")

    with main_col1:
        st.markdown("### 📥 Source Selection")
        with st.container():
            uploaded_file = st.file_uploader(
                "Drop image here for analysis", 
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.markdown('<div class="pixel-card">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                
                # Image Metadata Preview
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<p class="metric-label">FORMAT</p><p class="metric-value">{uploaded_file.type.split("/")[-1].upper()}</p>', unsafe_allow_html=True)
                m2.markdown(f'<p class="metric-label">DIMENSIONS</p><p class="metric-value">{image.size[0]}x{image.size[1]}</p>', unsafe_allow_html=True)
                m3.markdown(f'<p class="metric-label">SIZE</p><p class="metric-value">{uploaded_file.size/1024:.1f} KB</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pixel-card" style="text-align: center; padding: 4rem 2rem; border: 2px dashed #30363D;">
                    <p style="color: #8B949E; font-size: 1.1rem;">
                        No image selected.<br>
                        Upload a file to begin forensic analysis.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    with main_col2:
        st.markdown("### 📊 Forensic Analysis")
        
        if uploaded_file:
            with st.spinner("Decoding pixel artifacts..."):
                # Preprocess & Inference
                input_tensor = preprocess_image(image)
                results = run_inference(model, input_tensor, device)
                
                # Results UI
                is_real = results["class"] == "REAL"
                state_class = "state-real" if is_real else "state-ai"
                label_text = "AUTHENTIC" if is_real else "AI GENERATED"
                
                st.markdown(f"""
                    <div class="pixel-card result-card {state_class}">
                        <p class="result-label">CLASSIFICATION</p>
                        <h2 class="result-value">{label_text}</h2>
                        <p class="confidence-text">Confidence Level: {results['confidence']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence Meter
                st.markdown(f'<p class="metric-label">PROBABILITY CONFIDENCE</p>', unsafe_allow_html=True)
                st.progress(results['confidence'])
                
                # Detailed Metrics
                st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                d1.markdown(f'<p class="metric-label">LATENCY</p><p class="metric-value">{results["inference_time_ms"]:.1f}ms</p>', unsafe_allow_html=True)
                d2.markdown(f'<p class="metric-label">HARDWARE</p><p class="metric-value">{device.type.upper()}</p>', unsafe_allow_html=True)
                d3.markdown(f'<p class="metric-label">PRECISION</p><p class="metric-value">FP32</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if results['confidence'] < 0.75:
                    st.warning("🔬 **Ambiguity Detected**: The model suggests this image contains hybrid features or low-level artifacts that are difficult to categorize with high certainty.")
        else:
            st.markdown("""
            <div class="pixel-card" style="padding: 2rem; color: #8B949E;">
                Waiting for input data...
            </div>
            """, unsafe_allow_html=True)

    # --- Footer ---
    render_footer()

if __name__ == "__main__":
    main()
