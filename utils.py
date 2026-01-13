import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# --- CONFIGURATION ---
# Colors
COLOR_BG = "#0E1117"
COLOR_SURFACE = "#1E1E1E"
COLOR_ACCENT_GREEN = "#00FFA3"  # Cyber Green
COLOR_ACCENT_RED = "#FF0055"  # Cyber Red
COLOR_ACCENT_BLUE = "#00D4FF"  # Cyber Blue
COLOR_TEXT_MAIN = "#FAFAFA"
COLOR_TEXT_DIM = "#A0A0A0"


# --- MODEL LOADING ---
@st.cache_resource
def load_onnx_model(model_dir):
    """
    Loads the ONNX model and tokenizer from the specified directory.
    Cached to prevent reloading on every interaction.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = ORTModelForSequenceClassification.from_pretrained(model_dir)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model from {model_dir}: {e}")
        return None, None


# --- DESIGN SYSTEM ---
def apply_custom_css():
    """
    Injects the custom CSS for the 'Cyber/Hacker' aesthetic.
    """
    st.markdown(
        f"""
        <style>
            /* IMPORT FONTS */
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');

            /* GLOBAL SETTINGS */
            .stApp {{
                background-color: {COLOR_BG};
                font-family: 'Inter', sans-serif;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                font-family: 'Space Grotesk', sans-serif;
                color: {COLOR_TEXT_MAIN} !important;
                letter-spacing: -0.5px;
            }}
            
            p, label, span {{
                color: {COLOR_TEXT_DIM};
            }}

            /* CUSTOM BUTTONS */
            /* CUSTOM BUTTONS */
            /* Default - Green (Zoomer/Universal) */
            div.stButton > button {{
                background: #00FFA3 !important;
                color: #000000 !important; /* Black text for neon green */
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 700 !important;
                letter-spacing: 0.5px;
                border: none;
                padding: 0.7rem 1.2rem;
                border-radius: 8px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 255, 163, 0.3);
                text-shadow: none !important;
            }}
            
            div.stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 255, 163, 0.5);
                background: #00E692 !important;
            }}

            /* Primary - Red (Manager/Corporate) */
            div.stButton > button[kind="primary"] {{
                background: #FF0055 !important;
                color: #FFFFFF !important; /* White text for red */
                box-shadow: 0 4px 15px rgba(255, 0, 85, 0.3);
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }}

            div.stButton > button[kind="primary"]:hover {{
                background: #E6004C !important;
                box-shadow: 0 6px 20px rgba(255, 0, 85, 0.5);
            }}

            /* TEXT AREAS (Glassmorphism) */
            .stTextArea textarea {{
                background-color: rgba(30, 30, 30, 0.6);
                backdrop-filter: blur(10px);
                border: 1px solid #333;
                color: {COLOR_TEXT_MAIN};
                font-family: 'Inter', sans-serif;
                border-radius: 10px;
                transition: border-color 0.3s;
            }}
            
            .stTextArea textarea:focus {{
                border-color: {COLOR_ACCENT_BLUE};
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.1);
            }}

            /* RESULT CARDS */
            .result-card {{
                background: rgba(30, 30, 30, 0.4);
                backdrop-filter: blur(12px);
                border-radius: 16px;
                padding: 2rem;
                margin-top: 1.5rem;
                border: 1px solid rgba(255, 255, 255, 0.05);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                text-align: center;
                animation: fadeIn 0.5s ease-out;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .result-title {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.8rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }}
            
            .toxic-text {{ color: {COLOR_ACCENT_RED}; text-shadow: 0 0 15px rgba(255, 0, 85, 0.4); }}
            .safe-text {{ color: {COLOR_ACCENT_GREEN}; text-shadow: 0 0 15px rgba(0, 255, 163, 0.4); }}
            .neutral-text {{ color: {COLOR_ACCENT_BLUE}; }}

            .translation-box {{
                background: rgba(0, 0, 0, 0.3);
                border-left: 4px solid;
                padding: 1rem;
                margin-top: 1rem;
                text-align: left;
                font-family: 'Courier New', monospace;
                font-size: 1.1rem;
                color: #e0e0e0;
                border-radius: 4px;
            }}

            /* SIDEBAR */
            .css-1d391kg, [data-testid="stSidebar"] {{
                background-color: rgba(11, 13, 18, 0.95);
                backdrop-filter: blur(20px);
                border-right: 1px solid rgba(255, 255, 255, 0.05);
            }}

            /* SIDEBAR NAVIGATION LINKS */
            div[data-testid="stSidebar"] div[data-testid="stSidebarNav"] li {{
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.02));
                border-left: 2px solid transparent;            
                margin-bottom: 5px;
                transition: all 0.2s ease;
            }}

            div[data-testid="stSidebar"] div[data-testid="stSidebarNav"] li:hover {{
                border-left: 2px solid {COLOR_ACCENT_BLUE};
                background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), transparent);
            }}
            
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label, confidence, translation):
    """
    Renders the result card with appropriate styling and triggers animations.
    """

    # 1. Determine Style & Animation
    if label == "PASSIVE_AGGRESSIVE":
        theme_color = COLOR_ACCENT_RED
        text_class = "toxic-text"
        icon = "💀"
        title = "TOXIC DETECTED"
        # Trigger Snow (The Cold Shoulder)
        st.snow()

    elif label == "POSITIVE":
        theme_color = COLOR_ACCENT_GREEN
        text_class = "safe-text"
        icon = "🎉"
        title = "GENUINELY NICE"
        # Trigger Balloons (Party)
        st.balloons()

    else:  # NEUTRAL
        theme_color = COLOR_ACCENT_BLUE
        text_class = "neutral-text"
        icon = "🤖"
        title = "NEUTRAL / PROFESSIONAL"
        # No big animation, maybe just the card fade-in

    # 2. Render HTML Card
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title {text_class}">
                {icon} {title}
            </div>
            <p style="font-size: 1rem; opacity: 0.8; margin-bottom: 20px;">
                Confidence: <b>{confidence:.1%}</b>
            </p>
            <div class="translation-box" style="border-color: {theme_color};">
                <span style="display:block; font-size: 0.8rem; color: #888; margin-bottom: 5px;">
                    REAL MEANING:
                </span>
                {translation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
