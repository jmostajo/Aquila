from __future__ import annotations
import streamlit as st
from fonts_embed import _embed_font_css
import numpy as np
import pandas as pd
from io import BytesIO
import json, re, hashlib, secrets
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import time
import uuid, html
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Dict, Tuple, Union
from scipy.linalg import eig, inv
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
from numpy.typing import ArrayLike
import os, shutil, tempfile, subprocess
from pathlib import Path
from PIL import Image
from string import Template
from contextlib import asynccontextmanager
from fastapi import FastAPI
import base64  # for logo encoding

_embed_font_css()

# ===========================
# FUTURISTIC DARK THEME WITH ADVANCED VISUALS
# ===========================
FUTURISTIC_THEME_CSS = """
<style>
:root { 
    color-scheme: dark !important; 
    --primary-glow: radial-gradient(circle at 50% 50%, rgba(96, 165, 250, 0.15) 0%, transparent 70%);
    --secondary-glow: radial-gradient(circle at 20% 80%, rgba(167, 139, 250, 0.1) 0%, transparent 50%);
    --accent-glow: radial-gradient(circle at 80% 20%, rgba(244, 114, 182, 0.1) 0%, transparent 50%);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {
    background-color: #0A0F1A !important; 
    color: #F8FAFC !important; 
    transition: none !important;
}

div, section, header, main, aside, nav { 
    background-color: transparent !important; 
}

/* Main app background with animated gradient mesh */
.stApp {
    background: 
        radial-gradient(ellipse at top left, rgba(30, 58, 138, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at top right, rgba(88, 28, 135, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at bottom left, rgba(190, 24, 93, 0.1) 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, rgba(14, 165, 233, 0.1) 0%, transparent 50%),
        linear-gradient(135deg, #0A0F1A 0%, #111827 50%, #0A0F1A 100%) !important;
    position: relative;
    overflow-x: hidden;
}

/* Animated background elements */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        var(--primary-glow),
        var(--secondary-glow),
        var(--accent-glow);
    background-size: 100% 100%;
    animation: gradient-pulse 20s ease infinite;
    pointer-events: none;
    z-index: -1;
}

@keyframes gradient-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

/* Floating particles effect */
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
}

.floating-particle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: rgba(96, 165, 250, 0.3);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}

/* Enhanced sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, 
        rgba(26, 35, 50, 0.95) 0%, 
        rgba(15, 23, 33, 0.98) 100%) !important;
    border-right: 1px solid rgba(96, 165, 250, 0.15) !important;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 60px rgba(96, 165, 250, 0.1);
}

/* Futuristic cards with glass morphism */
.futuristic-card {
    background: linear-gradient(135deg, 
        rgba(30, 41, 59, 0.7) 0%, 
        rgba(15, 23, 33, 0.9) 100%);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(20px);
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1),
        inset 0 -1px 0 rgba(0, 0, 0, 0.2);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.futuristic-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(96, 165, 250, 0.1), 
        transparent);
    transition: left 0.6s ease;
}

.futuristic-card:hover::before {
    left: 100%;
}

.futuristic-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: rgba(96, 165, 250, 0.4);
    box-shadow: 
        0 20px 60px rgba(96, 165, 250, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.2),
        inset 0 -1px 0 rgba(0, 0, 0, 0.3);
}

/* Enhanced metric cards */
.metric-card-futuristic {
    background: linear-gradient(135deg, 
        rgba(30, 41, 59, 0.8) 0%, 
        rgba(15, 23, 33, 0.95) 100%);
    border: 1px solid rgba(96, 165, 250, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(15px);
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.25),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card-futuristic::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.metric-card-futuristic:hover::after {
    opacity: 1;
}

.metric-card-futuristic:hover {
    transform: translateY(-4px);
    border-color: rgba(96, 165, 250, 0.3);
    box-shadow: 
        0 12px 48px rgba(96, 165, 250, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

/* Futuristic inputs */
.stNumberInput input, .stSelectbox select, .stTextInput input {
    background: rgba(15, 23, 33, 0.6) !important;
    border: 1px solid rgba(96, 165, 250, 0.2) !important;
    border-radius: 12px !important;
    color: #F8FAFC !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px);
}

.stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
    border-color: rgba(96, 165, 250, 0.6) !important;
    box-shadow: 
        0 0 0 3px rgba(96, 165, 250, 0.1),
        0 0 30px rgba(96, 165, 250, 0.2) !important;
    background: rgba(15, 23, 33, 0.8) !important;
}

/* Enhanced slider */
.stSlider [data-baseweb="slider"] {
    background: linear-gradient(90deg, 
        rgb(239, 68, 68), 
        rgb(251, 146, 60), 
        rgb(234, 179, 8), 
        rgb(132, 204, 22), 
        rgb(34, 197, 94)) !important;
    height: 8px !important;
    border-radius: 10px !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
}

.stSlider [data-baseweb="slider"]:hover {
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3), 0 0 10px rgba(96, 165, 250, 0.3);
}

/* Futuristic buttons */
.futuristic-button {
    background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 
        0 8px 32px rgba(96, 165, 250, 0.3),
        0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

.futuristic-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(255, 255, 255, 0.2), 
        transparent);
    transition: left 0.6s ease;
}

.futuristic-button:hover::before {
    left: 100%;
}

.futuristic-button:hover {
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 
        0 15px 45px rgba(96, 165, 250, 0.4),
        0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

/* Section headers with futuristic design */
.section-header-futuristic {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 3rem 0 2rem;
    padding: 1rem 0;
    position: relative;
    text-align: center;
}

.section-header-futuristic::before {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 4px;
    background: linear-gradient(90deg, transparent, #60A5FA, #A78BFA, #F472B6, transparent);
    border-radius: 2px;
}

/* Step indicators */
.step-indicator {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 2rem 0;
    gap: 2rem;
}

.step-circle {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1E293B 0%, #0F1721 100%);
    border: 2px solid rgba(96, 165, 250, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    color: #60A5FA;
    transition: all 0.3s ease;
    position: relative;
}

.step-circle.active {
    background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
    color: white;
    border-color: #60A5FA;
    box-shadow: 0 0 20px rgba(96, 165, 250, 0.4);
}

.step-circle.completed {
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    color: white;
    border-color: #22C55E;
}

.step-connector {
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, rgba(96, 165, 250, 0.3), rgba(96, 165, 250, 0.1));
    max-width: 80px;
}

/* Enhanced info boxes */
.info-box-futuristic {
    background: linear-gradient(135deg, 
        rgba(59, 130, 246, 0.1) 0%, 
        rgba(147, 51, 234, 0.1) 100%);
    border-left: 4px solid #60A5FA;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 2rem 0;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(96, 165, 250, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

/* Advanced animations */
@keyframes hologram-glow {
    0%, 100% { 
        box-shadow: 
            0 0 20px rgba(96, 165, 250, 0.3),
            inset 0 0 20px rgba(96, 165, 250, 0.1);
    }
    50% { 
        box-shadow: 
            0 0 40px rgba(96, 165, 250, 0.6),
            inset 0 0 30px rgba(96, 165, 250, 0.2);
    }
}

.hologram-glow {
    animation: hologram-glow 3s ease-in-out infinite;
}

@keyframes data-stream {
    0% { transform: translateY(-100%); opacity: 0; }
    50% { opacity: 1; }
    100% { transform: translateY(100%); opacity: 0; }
}

.data-stream {
    position: absolute;
    width: 2px;
    height: 100%;
    background: linear-gradient(to bottom, transparent, #60A5FA, transparent);
    animation: data-stream 2s linear infinite;
}

/* Decision panel with advanced visuals */
.decision-panel {
    background: linear-gradient(135deg, 
        rgba(30, 41, 59, 0.9) 0%, 
        rgba(15, 23, 33, 0.95) 100%);
    border-radius: 24px;
    padding: 3rem;
    text-align: center;
    backdrop-filter: blur(25px);
    border: 1px solid rgba(96, 165, 250, 0.3);
    box-shadow: 
        0 20px 80px rgba(96, 165, 250, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.decision-panel::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(
        from 0deg,
        transparent,
        rgba(96, 165, 250, 0.1),
        rgba(167, 139, 250, 0.1),
        rgba(244, 114, 182, 0.1),
        transparent
    );
    animation: rotate 10s linear infinite;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Risk visualization */
.risk-visualization {
    background: linear-gradient(135deg, 
        rgba(30, 41, 59, 0.8) 0%, 
        rgba(15, 23, 33, 0.9) 100%);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(96, 165, 250, 0.2);
    backdrop-filter: blur(15px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
}

/* Progress bars */
.progress-bar-futuristic {
    background: rgba(15, 23, 33, 0.6);
    border-radius: 10px;
    height: 12px;
    overflow: hidden;
    position: relative;
    border: 1px solid rgba(96, 165, 250, 0.2);
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #60A5FA, #A78BFA);
    border-radius: 10px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(255, 255, 255, 0.3), 
        transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}
</style>
"""

st.markdown(FUTURISTIC_THEME_CSS, unsafe_allow_html=True)

# Add floating particles
st.markdown("""
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: -1;">
    <div class="floating-particle" style="left: 10%; top: 20%; animation-delay: 0s;"></div>
    <div class="floating-particle" style="left: 20%; top: 60%; animation-delay: 1s;"></div>
    <div class="floating-particle" style="left: 50%; top: 30%; animation-delay: 2s;"></div>
    <div class="floating-particle" style="left: 80%; top: 70%; animation-delay: 3s;"></div>
    <div class="floating-particle" style="left: 90%; top: 40%; animation-delay: 4s;"></div>
    <div class="floating-particle" style="left: 30%; top: 80%; animation-delay: 5s;"></div>
</div>
""", unsafe_allow_html=True)

# === Logo (Hexa.png) support ===
BASE_DIR = Path(__file__).parent.resolve()
LOGO_CANDIDATES = [
    Path.home() / "Desktop" / "Hexa.png",   # Desktop
    BASE_DIR / "Hexa.png",                   # project root
    BASE_DIR / "assets" / "Hexa.png",
    BASE_DIR / "static" / "Hexa.png",
    BASE_DIR / "images" / "Hexa.png",
]

def _first_existing(paths):
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None

RESOLVED_LOGO = _first_existing(LOGO_CANDIDATES)

@st.cache_data(show_spinner=False)
def _load_logo(path: Path | None):
    if not path:
        return None
    try:
        return Image.open(path)
    except Exception:
        return None

def _img_to_base64(img) -> str:
    if img is None:
        return ""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

logo_img = _load_logo(RESOLVED_LOGO)
logo_b64 = _img_to_base64(logo_img)
# === end logo block ===

# === Constantes y helpers ===
APP_VERSION = "6.6-AQ"
CO_ANUAL_FIJO = 0.015
TM_ANUAL_FIJO = 0.0075
PD12_ANCLA_1, PD12_ANCLA_5 = 0.80, 0.05
EPS = 1e-12

def clamp(x, lo, hi): return max(lo, min(hi, x))

def fmt_pct(x, p=2):
    try:
        if x is None or not np.isfinite(x): return "—"
        return f"{round(x*100, p):.{p}f}%"
    except Exception: return "—"

def fmt_usd(x, p=0):
    try: return "—" if x is None or not np.isfinite(x) else f"${x:,.{p}f}"
    except Exception: return "—"

def to_monthly(r_ann: float) -> float:
    r = np.clip(float(r_ann), -0.99, 10.0)
    return (1.0 + r) ** (1.0/12.0) - 1.0

def to_annual_from_monthly(r_m: float) -> float:
    r = np.clip(float(r_m), -0.99, 10.0)
    return (1.0 + r) ** 12 - 1.0

def lambda_from_pd12(pd12: float) -> float:
    pd12 = clamp(float(pd12), 0.0 + EPS, 1.0 - EPS)
    return -np.log(1.0 - pd12)

def lambda_anchors(pd1_12=PD12_ANCLA_1, pd5_12=PD12_ANCLA_5):
    lam1 = lambda_from_pd12(pd1_12); lam5 = lambda_from_pd12(pd5_12)
    return lam1, lam5

def lambda_from_score(score: float, lam1: float, lam5: float) -> float:
    score = clamp(float(score), 1.0, 5.0)
    ln1, ln5 = np.log(max(lam1, EPS)), np.log(max(lam5, EPS))
    alpha = (score - 1.0) / 4.0
    return float(np.exp(ln1 + alpha*(ln5 - ln1)))

def pd_hazard_months(lam: float, m: int) -> float:
    m = int(max(0, m)); t_years = m / 12.0
    return 1.0 - np.exp(-max(lam, 0.0) * t_years)

def lgd_politica(garantias_usd: float, EAD: float) -> float:
    EAD = max(float(EAD), 0.0)
    if EAD <= 0: return 0.0
    Vmax = min(float(garantias_usd), EAD)
    LGD = 1.0 - (Vmax / EAD)
    return max(0.0, min(1.0, LGD))

def calcular_resultados_ejecutivo(
    score: float, EAD: float, tc_ann: float, garantias_usd: float, gastos_usd: float = 0.0
):
    lam1, lam5 = lambda_anchors()
    lam = lambda_from_score(score, lam1, lam5)
    PD_12m = 1.0 - np.exp(-lam)
    PD1 = pd_hazard_months(lam, 12)
    PD2_cond = pd_hazard_months(lam, 3)
    LGD = lgd_politica(garantias_usd, EAD)
    PDD = LGD * EAD
    tc_m = to_monthly(tc_ann)
    co_m = to_monthly(CO_ANUAL_FIJO)
    tm_m = to_monthly(TM_ANUAL_FIJO)
    CF_t1 = EAD * (1 + tc_m) ** 12
    CF_t2_cura = EAD * (1 + tc_m) ** 15 * (1 + tm_m) ** 3
    PV_t2_def = garantias_usd
    PV_t1 = CF_t1 / ((1 + co_m) ** 12)
    PV_t2_cura = CF_t2_cura / ((1 + co_m) ** 15)
    EV_VP = (1 - PD1) * PV_t1 + PD1 * ((1 - PD2_cond) * PV_t2_cura + PD2_cond * PV_t2_def)
    RE_anual_simple = tc_ann * (1 - PD_12m) - LGD * PD_12m
    Texp = 12.0 + PD1 * 3.0
    multiplo_vp = EV_VP / EAD if EAD > 0 else np.nan
    return {"PD_12m": PD_12m, "LGD": LGD, "PDD": PDD, "RE_anual_simple": RE_anual_simple,
            "EV_VP": EV_VP, "multiplo_vp": multiplo_vp, "Texp": Texp}

# === Page config ===
st.set_page_config(
    page_title="Aquila — Análisis de Riesgo Crediticio",
    page_icon=logo_img if logo_img else "📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Sidebar ===
with st.sidebar:
    # Show your logo in the sidebar
    if logo_img:
        st.image(logo_img, use_container_width=True)
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='margin: 0; background: linear-gradient(135deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🦅 AQUILA</h2>
        <p style='margin: 0; color: rgba(248, 250, 252, 0.7); font-size: 0.9rem;'>Análisis de Riesgo Crediticio Inteligente</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    RET_THRESHOLD = st.number_input(
        "🎯 Umbral de Retorno Mínimo",
        min_value=0.0, max_value=1.0,
        value=0.12, step=0.01, format="%.2f",
        help="Retorno esperado mínimo para aprobar (12% default)"
    )
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: rgba(248, 250, 252, 0.5); font-size: 0.8rem;'>
        <p>© {} · Juan José Mostajo León</p>
        <p>Version {}</p>
    </div>
    """.format(datetime.now().year, APP_VERSION), unsafe_allow_html=True)

# Optional centered logo above header
if logo_b64:
    st.markdown(
        f"<div style='text-align:center; padding: 2rem 0 1rem;'>"
        f"<img src='data:image/png;base64,{logo_b64}' alt='AQUILA logo' "
        f"style='height:80px; margin-bottom:10px; filter: drop-shadow(0 0 20px rgba(96,165,250,.4));'/>"
        f"</div>",
        unsafe_allow_html=True
    )

# Header with enhanced design
st.markdown("""
<div style='text-align: center; padding: 2rem 0 3rem;'>
    <h1 style='font-size: 4rem; font-weight: 900; background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 30%, #F472B6 70%, #60A5FA 100%); 
    background-size: 200% auto; -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; 
    margin-bottom: 0.5rem; animation: gradient-shift 3s ease-in-out infinite;'>
        AQUILA
    </h1>
    <p style='font-size: 1.3rem; color: rgba(248, 250, 252, 0.8); letter-spacing: 0.05em; text-transform: uppercase;'>
        Sistema de Decisión de Riesgo Crediticio
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Step indicators
st.markdown("""
<div class="step-indicator">
    <div class="step-circle active">1</div>
    <div class="step-connector"></div>
    <div class="step-circle">2</div>
    <div class="step-connector"></div>
    <div class="step-circle">3</div>
    <div class="step-connector"></div>
    <div class="step-circle">4</div>
    <div class="step-connector"></div>
    <div class="step-circle">5</div>
    <div class="step-connector"></div>
    <div class="step-circle">6</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='info-box-futuristic'>
    <div style='display: flex; align-items: center; gap: 1rem;'>
        <div style='font-size: 1.5rem;'>🚀</div>
        <div>
            <strong style='font-size: 1.1rem;'>Guía de Análisis:</strong> Siga los pasos numerados en orden para realizar el análisis completo de riesgo crediticio.
            Cada paso representa una fase crítica en la evaluación del riesgo.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# === STEP 1: File Upload ===
st.markdown("<div class='section-header-futuristic'>📁 Paso 1: Cargar Cartera de Clientes</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Arrastre y suelte el archivo OPINT.xlsx o haga clic para seleccionar",
        type=["xlsx"],
        help="Formato esperado: columnas 'Cliente', 'Exposición USD', 'GARANTIAS'"
    )

if uploaded:
    try:
        df_cart = pd.read_excel(uploaded, sheet_name="CARTERA")
        df_cart = df_cart.rename(columns=lambda c: str(c).strip())
        
        with col2:
            st.markdown(f"""
            <div class='futuristic-card' style='text-align: center;'>
                <div style='font-size: 3rem; font-weight: 800; color: #22C55E;'>📊</div>
                <div style='font-size: 2rem; font-weight: 800; color: #22C55E; margin: 0.5rem 0;'>{len(df_cart)}</div>
                <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.7);'>
                    Clientes Cargados
                </div>
                <div style='font-size: 0.7rem; color: rgba(248, 250, 252, 0.5); margin-top: 0.5rem;'>
                    Archivo procesado exitosamente
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Update step indicator
        st.markdown("""
        <script>
        document.querySelectorAll('.step-circle')[0].classList.add('completed');
        document.querySelectorAll('.step-circle')[1].classList.add('active');
        </script>
        """, unsafe_allow_html=True)
        
        # === STEP 2: Client Selection ===
        st.markdown("<div class='section-header-futuristic'>👤 Paso 2: Seleccionar Cliente para Análisis</div>", unsafe_allow_html=True)
        
        cliente = st.selectbox(
            "Cliente a analizar:",
            df_cart.iloc[:, 0],
            help="Seleccione el cliente de la cartera cargada"
        )
        
        row = df_cart[df_cart.iloc[:, 0] == cliente].iloc[0]
        EAD_default = float(row.get("Exposición USD", 1_000_000.0))
        garantias_default = float(row.get("GARANTIAS", 600_000.0))
        
        st.markdown("#### Configuración de Exposición y Garantías")
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            st.markdown("""
            <div class='metric-card-futuristic'>
                <div style='font-size: 2rem; text-align: center; margin-bottom: 1rem;'>💰</div>
                <div style='text-align: center;'>
                    <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.7); margin-bottom: 0.5rem;'>Exposición al Default</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            EAD_sel = st.number_input(
                "EAD (USD)",
                min_value=0.0,
                value=float(EAD_default),
                step=10_000.0,
                format="%.0f",
                help="Exposición al Default",
                label_visibility="collapsed"
            )
        
        with col_in2:
            st.markdown("""
            <div class='metric-card-futuristic'>
                <div style='font-size: 2rem; text-align: center; margin-bottom: 1rem;'>📊</div>
                <div style='text-align: center;'>
                    <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.7); margin-bottom: 0.5rem;'>Gastos Asociados</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            gastos_sel = st.number_input(
                "Gastos (USD)",
                min_value=0.0,
                value=0.0,
                step=1_000.0,
                format="%.0f",
                help="Gastos asociados",
                label_visibility="collapsed"
            )
        
        with col_in3:
            st.markdown("""
            <div class='metric-card-futuristic'>
                <div style='font-size: 2rem; text-align: center; margin-bottom: 1rem;'>🛡️</div>
                <div style='text-align: center;'>
                    <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.7); margin-bottom: 0.5rem;'>Valor de Garantías</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            garantias_sel = st.number_input(
                "Garantías (USD)",
                min_value=0.0,
                value=float(garantias_default),
                step=10_000.0,
                format="%.0f",
                help="Valor de garantías",
                label_visibility="collapsed"
            )
        
        # Update step indicator
        st.markdown("""
        <script>
        document.querySelectorAll('.step-circle')[1].classList.add('completed');
        document.querySelectorAll('.step-circle')[2].classList.add('active');
        </script>
        """, unsafe_allow_html=True)
        
        # === STEP 3: Rate Configuration ===
        st.markdown("<div class='section-header-futuristic'>🧮 Paso 3: Configurar Tasa Compensatoria</div>", unsafe_allow_html=True)
        
        col_rate1, col_rate2, col_rate3 = st.columns([2, 2, 1])
        
        with col_rate1:
            st.markdown("""
            <div class='metric-card-futuristic'>
                <div style='font-size: 2rem; text-align: center; margin-bottom: 1rem;'>📈</div>
                <div style='text-align: center;'>
                    <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.7); margin-bottom: 0.5rem;'>Tasa Mensual</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            tasa_m_input = st.number_input(
                "Tasa Mensual (decimal)",
                min_value=0.0, max_value=0.20,
                value=0.025, step=0.001, format="%.4f",
                help="Ejemplo: 0.025 = 2.50% mensual",
                label_visibility="collapsed"
            )
        
        with col_rate2:
            tasa_anual_calc = to_annual_from_monthly(tasa_m_input)
            st.markdown(f"""
            <div class='futuristic-card' style='text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 1rem;'>⚡</div>
                <div style='font-size: 0.85rem; color: rgba(248, 250, 252, 0.6); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>
                    Equivalente Anual
                </div>
                <div style='font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #A78BFA, #F472B6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                    {tasa_anual_calc*100:.2f}%
                </div>
                <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5); margin-top: 0.5rem;'>
                    Tasa anualizada
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rate3:
            st.markdown("<br>", unsafe_allow_html=True)
            aplicar_tasa = st.button("⚡ Aplicar Tasa", key="apply_rate", use_container_width=True, type="primary")
        
        if aplicar_tasa:
            st.session_state["tc_ann_applied"] = tasa_anual_calc
            st.session_state["tasa_aplicada"] = True
            st.success("✅ Tasa aplicada exitosamente")
        
        if not st.session_state.get("tasa_aplicada", False):
            st.warning("⏳ Aplique la tasa para continuar con el análisis")
            st.stop()
        
        # Update step indicator
        st.markdown("""
        <script>
        document.querySelectorAll('.step-circle')[2].classList.add('completed');
        document.querySelectorAll('.step-circle')[3].classList.add('active');
        </script>
        """, unsafe_allow_html=True)
        
        # === STEP 4: Risk Score ===
        st.markdown("<div class='section-header-futuristic'>📊 Paso 4: Calificación de Riesgo del Cliente</div>", unsafe_allow_html=True)
        
        col_score1, col_score2 = st.columns([3, 1])
        
        with col_score1:
            st.markdown("""
            <div class='risk-visualization'>
                <div style='text-align: center; margin-bottom: 2rem;'>
                    <div style='font-size: 1.1rem; color: rgba(248, 250, 252, 0.8); margin-bottom: 1rem;'>
                        Deslice el indicador para ajustar la calificación de riesgo
                    </div>
                </div>
            """, unsafe_allow_html=True)
            score = st.slider(
                "Calificación del Cliente (1=Alto Riesgo, 5=Bajo Riesgo)",
                min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_score2:
            risk_level = "🔴 Alto Riesgo" if score < 2.5 else "🟡 Riesgo Medio" if score < 4.0 else "🟢 Bajo Riesgo"
            risk_color = "#EF4444" if score < 2.5 else "#EAB308" if score < 4.0 else "#22C55E"
            risk_width = "20%" if score < 2.5 else "60%" if score < 4.0 else "100%"
            
            st.markdown(f"""
            <div class='futuristic-card' style='text-align: center; border-color: {risk_color};'>
                <div style='font-size: 3rem; font-weight: 800; color: {risk_color}; margin-bottom: 1rem;'>{score:.1f}</div>
                <div style='font-size: 1.1rem; color: {risk_color}; margin-bottom: 1rem; font-weight: 600;'>
                    {risk_level}
                </div>
                <div class='progress-bar-futuristic'>
                    <div class='progress-fill' style='width: {risk_width}; background: linear-gradient(90deg, {risk_color}, {risk_color});'></div>
                </div>
                <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5); margin-top: 1rem;'>
                    Nivel de Riesgo
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Update step indicator
        st.markdown("""
        <script>
        document.querySelectorAll('.step-circle')[3].classList.add('completed');
        document.querySelectorAll('.step-circle')[4].classList.add('active');
        </script>
        """, unsafe_allow_html=True)
        
        # === STEP 5: Analyze ===
        st.markdown("<div class='section-header-futuristic'>🎯 Paso 5: Ejecutar Análisis de Riesgo</div>", unsafe_allow_html=True)
        
        col_btn = st.columns([1, 2, 1])
        with col_btn[1]:
            analizar = st.button("🚀 EJECUTAR ANÁLISIS COMPLETO", type="primary", use_container_width=True, key="analyze")
        
        if analizar:
            with st.spinner("🔍 Calculando métricas de riesgo..."):
                time.sleep(1.2)
                resultado = calcular_resultados_ejecutivo(
                    score=score,
                    EAD=EAD_sel,
                    tc_ann=st.session_state['tc_ann_applied'],
                    garantias_usd=garantias_sel,
                    gastos_usd=gastos_sel
                )
            
            decision_ok = resultado["RE_anual_simple"] >= RET_THRESHOLD
            
            st.markdown("<br/>", unsafe_allow_html=True)
            decision_color = "#22C55E" if decision_ok else "#EF4444"
            decision_icon = "✅" if decision_ok else "⛔"
            decision_text = "APROBAR CRÉDITO" if decision_ok else "RECHAZAR CRÉDITO"
            decision_subtext = "Cumple con el umbral de retorno mínimo" if decision_ok else "No cumple con el umbral de retorno mínimo"
            
            st.markdown(f"""
            <div class='decision-panel hologram-glow'>
                <div style='position: relative; z-index: 2;'>
                    <div style='font-size: 5rem; margin-bottom: 1rem;'>{decision_icon}</div>
                    <div style='font-size: 3rem; font-weight: 900; color: {decision_color}; 
                        letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem;'>
                        {decision_text}
                    </div>
                    <div style='font-size: 1.3rem; color: rgba(248, 250, 252, 0.8); margin-bottom: 2rem;'>
                        {decision_subtext}
                    </div>
                    <div style='font-size: 1.5rem; color: {decision_color}; font-weight: 700;'>
                        Retorno Esperado: {resultado["RE_anual_simple"]*100:.2f}%
                    </div>
                    <div style='font-size: 1rem; color: rgba(248, 250, 252, 0.6); margin-top: 1rem;'>
                        Umbral Mínimo: {RET_THRESHOLD*100:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br/><div class='section-header-futuristic'>📈 Métricas Clave de Riesgo</div>", unsafe_allow_html=True)
            
            colm1, colm2, colm3, colm4 = st.columns(4)
            
            metrics_data = [
                ("Probabilidad Default", f"{resultado['PD_12m']*100:.2f}%", "12 meses", "#EF4444", colm1),
                ("LGD", f"{resultado['LGD']*100:.2f}%", "Loss Given Default", "#F59E0B", colm2),
                ("Pérdida Dada Default", fmt_usd(resultado['PDD'], 0), "PDD", "#A78BFA", colm3),
                ("Retorno Esperado", f"{resultado['RE_anual_simple']*100:.2f}%", "Anual", decision_color, colm4)
            ]
            
            for label, value, subtitle, color, column in metrics_data:
                with column:
                    st.markdown(f"""
                    <div class='futuristic-card' style='text-align: center; border-left: 4px solid {color};'>
                        <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.6); 
                            text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem;'>
                            {label}
                        </div>
                        <div style='font-size: 2.2rem; font-weight: 800; color: {color}; margin: 1rem 0;'>
                            {value}
                        </div>
                        <div style='font-size: 0.85rem; color: rgba(248, 250, 252, 0.5);'>
                            {subtitle}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Update step indicator
            st.markdown("""
            <script>
            document.querySelectorAll('.step-circle')[4].classList.add('completed');
            document.querySelectorAll('.step-circle')[5].classList.add('active');
            </script>
            """, unsafe_allow_html=True)
            
            # === STEP 6: Collateral Haircut ===
            st.markdown("<div class='section-header-futuristic'>🛡️ Paso 6: Análisis de Garantías en Default</div>", unsafe_allow_html=True)
            
            try:
                garantia_bruta_base = float(row.iloc[8])
            except Exception:
                garantia_bruta_base = 0.0
            
            try:
                def leer_peso_garantia_colM(row, fallback_colnames=("Peso de la Garantía", "Peso de la Garantia")):
                    val = None
                    try: 
                        val = row.iloc[12]
                    except Exception: 
                        val = None
                    
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        for cname in fallback_colnames:
                            if cname in row.index:
                                val = row.get(cname)
                                break
                    
                    try:
                        s = str(val).strip()
                        if s == "" or s.lower() == "nan": 
                            return 1.0
                        s = s.replace("%", "").replace(",", ".")
                        x = float(s)
                        if x > 1.0: 
                            x = x / 100.0
                        return float(np.clip(x, 0.0, 1.0))
                    except Exception:
                        return 1.0
                
                peso_garantia_base = leer_peso_garantia_colM(row)
            except Exception:
                peso_garantia_base = 1.0
            
            col_edit1, col_edit2 = st.columns(2)
            
            with col_edit1:
                st.markdown("""
                <div class='metric-card-futuristic'>
                    <div style='font-size: 2rem; text-align: center; margin-bottom: 1rem;'>💎</div>
                    <div style='text-align: center;'>
                        <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.7); margin-bottom: 0.5rem;'>Garantía Bruta</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                garantia_bruta_edit = st.number_input(
                    "Garantía Bruta (USD) — Valor Realizable",
                    min_value=0.0,
                    value=float(garantia_bruta_base),
                    step=10_000.0,
                    format="%.0f",
                    help="Valor de mercado de la garantía",
                    label_visibility="collapsed"
                )
            
            with col_edit2:
                st.markdown("""
                <div class='metric-card-futuristic'>
                    <div style='font-size: 2rem; text-align: center; margin-bottom: 1rem;'>⚖️</div>
                    <div style='text-align: center;'>
                        <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.7); margin-bottom: 0.5rem;'>Factor de Calidad</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                peso_garantia_edit = st.number_input(
                    "Peso de Garantía — Quality Collateral",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(peso_garantia_base),
                    step=0.01,
                    format="%.2f",
                    help="Factor de ajuste por calidad (0-1)",
                    label_visibility="collapsed"
                )
            
            garantia_castigada = garantia_bruta_edit * peso_garantia_edit
            
            st.markdown("<br/>", unsafe_allow_html=True)
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.markdown(f"""
                <div class='futuristic-card' style='text-align: center;'>
                    <div style='font-size: 2rem; margin-bottom: 1rem;'>💰</div>
                    <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.6); text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Garantía Bruta
                    </div>
                    <div style='font-size: 2.2rem; font-weight: 800; color: #60A5FA; margin: 1rem 0;'>
                        {fmt_usd(garantia_bruta_edit, 0)}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5);'>
                        Valor Original
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.markdown(f"""
                <div class='futuristic-card' style='text-align: center;'>
                    <div style='font-size: 2rem; margin-bottom: 1rem;'>📊</div>
                    <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.6); text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Factor de Ajuste
                    </div>
                    <div style='font-size: 2.2rem; font-weight: 800; color: #A78BFA; margin: 1rem 0;'>
                        {peso_garantia_edit:.2%}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5);'>
                        Quality Collateral
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res3:
                st.markdown(f"""
                <div class='futuristic-card' style='text-align: center; border: 2px solid #22C55E;'>
                    <div style='font-size: 2rem; margin-bottom: 1rem;'>🛡️</div>
                    <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.6); text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Garantía Castigada
                    </div>
                    <div style='font-size: 2.2rem; font-weight: 800; color: #22C55E; margin: 1rem 0;'>
                        {fmt_usd(garantia_castigada, 0)}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5);'>
                        Valor Final
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption(f"**Fórmula de Cálculo:** {fmt_usd(garantia_bruta_edit, 0)} × {peso_garantia_edit:.2%} = {fmt_usd(garantia_castigada, 0)}")
            
            # Complete all steps
            st.markdown("""
            <script>
            document.querySelectorAll('.step-circle')[5].classList.add('completed');
            </script>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error al leer archivo: {e}")
        st.info("ℹ️ Verifique que el archivo tenga el formato correcto y la hoja 'CARTERA'")
        st.stop()
else:
    st.markdown("""
    <div class='futuristic-card' style='text-align: center; padding: 4rem 2rem;'>
        <div style='font-size: 4rem; margin-bottom: 2rem;'>📁</div>
        <div style='font-size: 1.5rem; color: rgba(248, 250, 252, 0.8); margin-bottom: 1rem;'>
            Esperando Archivo
        </div>
        <div style='font-size: 1rem; color: rgba(248, 250, 252, 0.6);'>
            Por favor cargue el archivo OPINT.xlsx para comenzar el análisis
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Footer
st.markdown("<br/><br/>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<div style='text-align: center; color: rgba(248, 250, 252, 0.5); font-size: 0.9rem; padding: 2rem 0;'>
    <p style='margin: 0.5rem 0; font-size: 1.1rem; color: rgba(248, 250, 252, 0.7);'>Aquila Risk Analysis System</p>
    <p style='margin: 0.5rem 0;'>Powered by Advanced Credit Modeling & AI</p>
    <p style='margin: 0.5rem 0;'>© 2025 Juan José Mostajo León · Version 6.6-AQ</p>
</div>
""", unsafe_allow_html=True)
