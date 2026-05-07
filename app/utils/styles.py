import streamlit as st

def apply_custom_styles():
    """
    Applies professional CSS for a premium AI product aesthetic.
    Focuses on dark theme, subtle borders, and clean typography.
    """
    st.markdown("""
        <style>
        /* Base page styling */
        .stApp {
            background-color: #0D1117;
            color: #C9D1D9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Hero Section */
        .hero-container {
            padding: 3rem 0 2rem 0;
            text-align: center;
        }
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -0.05rem;
            margin-bottom: 0.5rem;
            color: #F0F6FC;
        }
        .hero-tagline {
            font-size: 1.25rem;
            color: #8B949E;
            max-width: 600px;
            margin: 0 auto 2rem auto;
        }
        
        /* Premium Card Styling */
        .pixel-card {
            background: #161B22;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        
        /* Prediction Result Cards */
        .result-card {
            text-align: center;
            padding: 2.5rem 1.5rem;
            border-radius: 16px;
            margin-top: 1rem;
        }
        .result-label {
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.1rem;
            color: #8B949E;
            margin-bottom: 0.5rem;
        }
        .result-value {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        /* State Colors */
        .state-real {
            border: 1px solid rgba(35, 134, 54, 0.4);
            background: rgba(35, 134, 54, 0.05);
        }
        .state-real .result-value {
            color: #3FB950;
        }
        
        .state-ai {
            border: 1px solid rgba(248, 81, 73, 0.4);
            background: rgba(248, 81, 73, 0.05);
        }
        .state-ai .result-value {
            color: #F85149;
        }
        
        /* Metrics & Metadata */
        .metric-label {
            font-size: 0.75rem;
            color: #8B949E;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1rem;
            font-weight: 600;
            color: #C9D1D9;
        }
        
        /* Sidebar Polish */
        [data-testid="stSidebar"] {
            background-color: #010409;
            border-right: 1px solid #30363D;
        }
        
        /* Hide default Streamlit elements for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Progress Bar Polish */
        .stProgress > div > div > div > div {
            background-color: #58A6FF;
        }
        </style>
    """, unsafe_allow_html=True)

def render_hero():
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">PixelProof</h1>
            <p class="hero-tagline">
                Advanced AI Authenticity Detection. Identify synthetic media with 
                forensic precision using deep learning analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #484F58; font-size: 0.8rem; padding: 1rem;">
            PixelProof Forensic Engine v1.0.0-baseline | Built for AI Authenticity Verification
        </div>
    """, unsafe_allow_html=True)
