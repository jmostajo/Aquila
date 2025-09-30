
from pathlib import Path
import base64
import streamlit as st

def _embed_font_css():
    here = Path(__file__).parent
    try:
        reg_b64 = base64.b64encode((here/"fonts"/"Inter-Variable.woff2").read_bytes()).decode("ascii")
        ita_b64 = base64.b64encode((here/"fonts"/"Inter-Variable-Italic.woff2").read_bytes()).decode("ascii")
    except FileNotFoundError:
        st.warning("⚠️ No se encontraron ./fonts/Inter-Variable*.woff2")
        reg_b64 = ita_b64 = ""

    st.markdown(f"""
    <style>
      @font-face {{
        font-family: 'Inter';
        src: url(data:font/woff2;base64,{reg_b64}) format('woff2');
        font-weight: 100 900; font-style: normal; font-display: swap;
      }}
      @font-face {{
        font-family: 'Inter';
        src: url(data:font/woff2;base64,{ita_b64}) format('woff2');
        font-weight: 100 900; font-style: italic; font-display: swap;
      }}
      html, body, [class*="st-"] * {{
        font-family: 'Inter', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
      }}
    </style>
    """, unsafe_allow_html=True)
