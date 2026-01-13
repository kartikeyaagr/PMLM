import streamlit as st
from utils import apply_custom_css

st.set_page_config(page_title="PMLM", page_icon="💀", layout="centered")
apply_custom_css()

# --- HERO SECTION ---
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Spacer
st.markdown(
    "<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'>💀 PMLM</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='text-align: center; font-weight: 300; opacity: 0.8; margin-top: 0;'>per my last model</h3>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)  # Spacer

# --- INTRO TEXT ---
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 40px; font-size: 1.1rem;'>
        for when you want to know what that one bitchy coworker is thinking
    </div>
    """,
    unsafe_allow_html=True,
)

# --- NAVIGATION CARDS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="result-card" style="margin-top: 0; height: 250px; cursor: pointer;">
            <div style="font-size: 3rem;">💼</div>
            <h4>The Manager</h4>
            <p style="font-size: 0.9rem;">The suit who hates you. Reminds me of my last boss ngl.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Talk to The Manager", use_container_width=True, type="primary"):
        st.switch_page("pages/1_The_Manager.py")

with col2:
    st.markdown(
        """
        <div class="result-card" style="margin-top: 0; height: 250px; cursor: pointer;">
            <div style="font-size: 3rem;">🧢</div>
            <h4>The Zoomer</h4>
            <p style="font-size: 0.9rem;">The intern with no filter. Literally us.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Talk to The Zoomer", use_container_width=True):
        st.switch_page("pages/2_The_Zoomer.py")

# --- FOOTER ---
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; font-size: 0.8rem; opacity: 0.5;'>
        I'm just getting started - Kartikeya Agrawal 2026. 
    </div>
    """,
    unsafe_allow_html=True,
)
