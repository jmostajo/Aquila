from __future__ import annotations
import streamlit as st
from fonts_embed import _embed_font_css
import numpy as np
import pandas as pd
from io import BytesIO
import json, re, hashlib, secrets
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
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
# Agregar en la sección de imports al principio del archivo
_embed_font_css()

# ===========================
# EdeX-UI THEME (CSS + Plotly + Matplotlib)
# ===========================
EDEX_UI_CSS = """
<style>
/* ---------- VARIABLES ---------- */
:root{
  color-scheme: dark !important;
  --edex-bg: #0a0e14;
  --edex-bg-2:#0f141b;
  --edex-grid:#121a25;
  --edex-cyan:#00fff6;
  --edex-green:#00ffa3;
  --edex-magenta:#ff5ea0;
  --edex-amber:#f2d17a;
  --edex-fg:#d7e2ec;
  --edex-dim:#9aa5b1;
  --radius: 12px;
  --glow: 0 0 10px rgba(0,255,246,.35), 0 0 30px rgba(0,255,246,.12);
}

/* ---------- BACKGROUND + GRID ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main{
  background: var(--edex-bg) !important;
  color: var(--edex-fg) !important;
  font-family: 'Fira Code', 'JetBrains Mono', 'Share Tech Mono', monospace;
}
.stApp::before{
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background:
    repeating-linear-gradient(0deg, rgba(0,255,246,.035) 0 1px, transparent 1px 3px),
    repeating-linear-gradient(90deg, rgba(0,255,246,.03) 0 1px, transparent 1px 24px);
}
.stApp::after{
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(0,0,0,0) 70%);
  mix-blend-mode: overlay;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--edex-bg-2), var(--edex-bg)) !important;
  border-right:1px solid rgba(0,255,246,.12) !important;
}
section[data-testid="stSidebar"] *{ font-family: inherit; }

/* ---------- HEADERS ---------- */
h1,h2,h3,h4{ color:var(--edex-cyan); text-shadow:0 0 6px rgba(0,255,246,.55); }
.section-header-enhanced{
  font-size:1.4rem; font-weight:800; color:var(--edex-cyan);
  letter-spacing:.06em; text-transform:uppercase; margin:1.8rem 0 1rem;
  position:relative; padding-bottom:.4rem;
}
.section-header-enhanced::after{
  content:''; position:absolute; left:0; bottom:0; height:2px; width:100%;
  background: linear-gradient(90deg, var(--edex-cyan), transparent 70%);
  box-shadow: var(--glow);
}

/* ---------- CARDS (metric-card) ---------- */
.metric-card{
  background: linear-gradient(180deg, rgba(16,22,30,.92), rgba(8,12,18,.96));
  border:1px solid rgba(0,255,246,.28);
  border-radius: var(--radius); padding:1rem 1.1rem;
  box-shadow: inset 0 0 0 1px rgba(0,255,246,.12), 0 8px 30px rgba(0,0,0,.35);
  position:relative; overflow:hidden; transition:.25s ease;
}
.metric-card::before{
  content:''; position:absolute; inset:0; pointer-events:none;
  background: radial-gradient(600px 120px at 0% 0%, rgba(0,255,246,.12), transparent 60%);
  opacity:.6;
}
.metric-card:hover{ transform:translateY(-2px); box-shadow: var(--glow); }

/* ---------- INFO BOX ---------- */
.info-box{
  background: linear-gradient(135deg, rgba(0,255,246,.08), rgba(0,255,163,.06));
  border-left:3px solid var(--edex-cyan); border-radius: var(--radius);
  padding: .75rem 1rem; color: var(--edex-fg);
}

/* ---------- BOTONES ---------- */
.stButton > button{
  border:1px solid rgba(0,255,246,.45) !important;
  background: rgba(10,14,20,.6) !important;
  color: var(--edex-cyan) !important;
  text-transform:uppercase; letter-spacing:.07em; font-weight:800;
  border-radius: var(--radius) !important; box-shadow: var(--glow);
  transition: all .2s ease;
}
.stButton > button:hover{
  background: rgba(0,255,246,.12) !important; transform: translateY(-1px);
}
.stButton > button:active{ transform: translateY(0); filter: brightness(1.1); }

/* ---------- INPUTS ---------- */
input, textarea, select{
  background: rgba(8,12,18,.85) !important;
  color: var(--edex-fg) !important;
  border:1px solid rgba(0,255,246,.25) !important;
  border-radius:10px !important;
}
input:focus, textarea:focus, select:focus{
  outline:none !important; box-shadow: 0 0 0 2px rgba(0,255,246,.35) !important;
}

/* ---------- SLIDER ---------- */
.stSlider [data-baseweb="slider"]{
  height:8px !important; border-radius:10px;
  background: linear-gradient(90deg, var(--edex-magenta), var(--edex-cyan), var(--edex-green)) !important;
}

/* ---------- ALERTAS ---------- */
[data-testid="stAlert"]{
  background: rgba(10,14,20,.65); border:1px solid rgba(0,255,246,.22);
  border-left:4px solid var(--edex-cyan); border-radius: var(--radius);
}

/* ---------- TABLAS / DATAFRAME ---------- */
[data-testid="stDataFrame"] .row-widget, .stDataFrame{ color: var(--edex-fg); }
.stDataFrame table{ border:1px solid rgba(0,255,246,.18); }
.stDataFrame thead tr{ background: rgba(0,255,246,.08); }
.stDataFrame th, .stDataFrame td{ border-color: rgba(0,255,246,.12) !important; }

/* ---------- DIVISORES / LINEAS ---------- */
hr, .stDivider{
  border: none; height:1px; background: linear-gradient(90deg, var(--edex-cyan), transparent);
  box-shadow: var(--glow); opacity:.6; margin: .8rem 0 !important;
}

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background: var(--edex-bg-2); }
::-webkit-scrollbar-thumb{ background: linear-gradient(var(--edex-cyan), var(--edex-green)); border-radius:8px; }

/* ---------- PULSE / DESTACADO ---------- */
@keyframes pulse-glow { 0%,100%{ box-shadow:0 0 20px rgba(0,255,246,0.2);} 50%{ box-shadow:0 0 40px rgba(0,255,246,0.4);} }
.pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
</style>
"""
st.markdown(EDEX_UI_CSS, unsafe_allow_html=True)

# (Opcional) Cargar tipografías monoespaciadas desde Google si _embed_font_css() no las incluye
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Plotly template estilo EdeX-UI
pio.templates["edex"] = go.layout.Template(
    layout=go.Layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0a0e14",
        font=dict(family="Fira Code, Share Tech Mono, monospace", color="#00fff6"),
        colorway=["#00fff6", "#00ffa3", "#7a7dff", "#f2d17a", "#ff5ea0"],
        xaxis=dict(gridcolor="#121a25", zerolinecolor="#121a25", linecolor="#00fff6"),
        yaxis=dict(gridcolor="#121a25", zerolinecolor="#121a25", linecolor="#00fff6"),
        margin=dict(l=50, r=30, t=50, b=40)
    )
)
pio.templates.default = "edex"

# Matplotlib estilo EdeX-UI - CORREGIDO
def apply_matplotlib_edex():
    mpl.rcParams.update({
        "figure.facecolor": "#0a0e14",
        "axes.facecolor": "#0a0e14",
        "savefig.facecolor": "#0a0e14",
        "text.color": "#00fff6",
        "axes.edgecolor": "#00fff6",  # Corregido: usar color hexadecimal en lugar de rgba
        "axes.labelcolor": "#00fff6",
        "xtick.color": "#d7e2ec",
        "ytick.color": "#d7e2ec",
        "grid.color": "#121a25",
        "grid.linestyle": "-",
        "axes.grid": True,
        "font.family": "monospace",
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.titlecolor": "#00fff6",
        "lines.linewidth": 2.0,
    })
apply_matplotlib_edex()

# === Logo (Hexa.png) support ===
BASE_DIR = Path(__file__).parent.resolve()
LOGO_CANDIDATES = [
    Path.home() / "Desktop" / "Fly.png",   # Desktop
    BASE_DIR / "Fly.png",                   # project root
    BASE_DIR / "assets" / "Fly.png",
    BASE_DIR / "static" / "Fly.png",
    BASE_DIR / "images" / "Fly.png",
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

# ===========================
# AUTH GATE — Login + Welcome
# ===========================
def _hash(p: str) -> str:
    return hashlib.sha256(p.encode("utf-8")).hexdigest()

# Universal password for all users
UNIVERSAL_PASSWORD = "Aquila2025!"
UNIVERSAL_PASSWORD_HASH = _hash(UNIVERSAL_PASSWORD)

def login_gate():
    # States
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if "welcome_done" not in st.session_state:
        st.session_state.welcome_done = False

    # If everything passed, continue
    if st.session_state.auth and st.session_state.welcome_done:
        return True

    # If not authenticated, show form
    if not st.session_state.auth:
        st.markdown("<div class='section-header-enhanced'>System Authentication</div>", unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                email = st.text_input(
                    "Email Address", 
                    key="login_email", 
                    autocomplete="email", 
                    placeholder="your.email@domain.com"
                )
                password = st.text_input(
                    "Password", 
                    type="password", 
                    key="login_password", 
                    placeholder="Enter your password"
                )
                
                ok = st.form_submit_button("Authenticate", use_container_width=True)

        if ok:
            # Verify any email + universal password
            if email.strip() and _hash(password) == UNIVERSAL_PASSWORD_HASH:
                st.session_state.auth = True
                st.session_state.user_email = email.strip().lower()
                
                # Extract name from email (part before @)
                name = email.split("@")[0].replace(".", " ").title()
                st.session_state.user_name = name
                
                st.rerun()
            else:
                st.error("Invalid credentials. Please verify your email and password.")
        st.stop()

    # If authenticated but welcome not shown yet, show it
    if st.session_state.auth and not st.session_state.welcome_done:
        # Name to display
        email = st.session_state.get("user_email", "")
        name = st.session_state.get("user_name", "User")

        st.markdown(f"""
        <div class='metric-card' style='padding:3rem 2rem; text-align:center; border: 1px solid rgba(0, 255, 246, 0.3);'>
            <div style='font-size:2.5rem; font-weight:700; color:#00ffa3; letter-spacing:.02em; margin-bottom:1.5rem;'>
                Authentication Successful
            </div>
            <div style='font-size:1.5rem; color:rgba(215,226,236,.9); margin-bottom:2rem;'>
                Welcome, {name}
            </div>
            <div style='color:rgba(215,226,236,.7); font-size:1rem; line-height:1.6;'>
                AQUILA Credit Risk Analysis System<br/>
                You are now authorized to access the risk assessment dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
            if st.button("Access Risk Dashboard", type="primary", use_container_width=True):
                st.session_state.welcome_done = True
                st.rerun()
        st.stop()

# Execute gate BEFORE building the rest of the app
login_gate()
# ===========================
# (Opcional) Logout en Sidebar
# — Coloca esto dentro de tu with st.sidebar: existente
# ===========================
# with st.sidebar:
#     if st.session_state.get("auth", False):
#         st.caption(f"Usuario: {st.session_state.get('user_email','')}")
#         if st.button("Cerrar sesión", use_container_width=True):
#             for k in ["auth","user_email","user_name","welcome_done","show_welcome","tasa_aplicada","tc_ann_applied"]:
#                 st.session_state.pop(k, None)
#             st.rerun()


# === Sidebar ===
with st.sidebar:
    # Show your logo in the sidebar
    if logo_img:
        st.image(logo_img, use_container_width=True)
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 🦅 AQUILA")
    st.caption("Análisis de Riesgo Crediticio Inteligente")
    st.divider()
    
    RET_THRESHOLD = st.number_input(
        "Umbral de Retorno Mínimo",
        min_value=0.0, max_value=1.0,
        value=0.12, step=0.01, format="%.2f",
        help="Retorno esperado mínimo para aprobar (12% default)"
    )
    
    st.divider()
    st.caption(f"© {datetime.now().year} · Juan José Mostajo León")
    st.caption(f"Version {APP_VERSION}")

# Optional centered logo above header
if logo_b64:
    st.markdown(
        f"<div style='text-align:center; padding-top: 1rem;'>"
        f"<img src='data:image/png;base64,{logo_b64}' alt='AQUILA logo' "
        f"style='height:72px; margin-bottom:10px; filter: drop-shadow(0 0 12px rgba(0,255,246,.4));'/>"
        f"</div>",
        unsafe_allow_html=True
    )

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #00fff6 0%, #00ffa3 50%, #f2d17a 100%); 
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;'>
        AQUILA
    </h1>
    <p style='font-size: 1.2rem; color: rgba(215, 226, 236, 0.75);'>
        Sistema de Decisión de Riesgo Crediticio
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<div class='info-box'>
    <strong>Guía Rápida:</strong> Siga los pasos numerados en orden para realizar el análisis completo de riesgo crediticio.
</div>
""", unsafe_allow_html=True)

# === STEP 1: File Upload ===
st.markdown("<div class='section-header-enhanced'>📁 Paso 1: Cargar Cartera</div>", unsafe_allow_html=True)

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
            st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 2.5rem; font-weight: 800; color: #00ffa3;'>✓</div>
                <div style='font-size: 0.9rem; color: rgba(215, 226, 236, 0.75); margin-top: 0.5rem;'>
                    Archivo cargado<br/>
                    <strong>{}</strong> clientes
                </div>
            </div>
            """.format(len(df_cart)), unsafe_allow_html=True)
        
        # === STEP 2: Client Selection ===
        st.markdown("<div class='section-header-enhanced'>👤 Paso 2: Seleccionar Cliente</div>", unsafe_allow_html=True)
        
        cliente = st.selectbox(
            "Cliente a analizar:",
            df_cart.iloc[:, 0],
            help="Seleccione el cliente de la cartera cargada"
        )
        
        row = df_cart[df_cart.iloc[:, 0] == cliente].iloc[0]
        EAD_default = float(row.get("Exposición USD", 1_000_000.0))
        garantias_default = float(row.get("GARANTIAS", 600_000.0))
        
        st.markdown("#### Configuración de Exposición")
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            EAD_sel = st.number_input(
                "💰 EAD (USD)",
                min_value=0.0,
                value=float(EAD_default),
                step=10_000.0,
                format="%.0f",
                help="Exposición al Default"
            )
        
        with col_in2:
            gastos_sel = st.number_input(
                "📊 Gastos (USD)",
                min_value=0.0,
                value=0.0,
                step=1_000.0,
                format="%.0f",
                help="Gastos asociados"
            )
        
        with col_in3:
            garantias_sel = st.number_input(
                "🛡️ Garantías (USD)",
                min_value=0.0,
                value=float(garantias_default),
                step=10_000.0,
                format="%.0f",
                help="Valor de garantías"
            )
        
        # === STEP 3: Rate Configuration ===
        st.markdown("<div class='section-header-enhanced'>🧮 Paso 3: Configurar Tasa Compensatoria</div>", unsafe_allow_html=True)
        
        col_rate1, col_rate2, col_rate3 = st.columns([2, 2, 1])
        
        with col_rate1:
            tasa_m_input = st.number_input(
                "Tasa Mensual (decimal)",
                min_value=0.0, max_value=0.20,
                value=0.025, step=0.001, format="%.4f",
                help="Ejemplo: 0.025 = 2.50% mensual"
            )
        
        with col_rate2:
            tasa_anual_calc = to_annual_from_monthly(tasa_m_input)
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.85rem; color: rgba(215, 226, 236, 0.7); margin-bottom: 0.5rem;'>
                    EQUIVALENTE ANUAL
                </div>
                <div style='font-size: 2rem; font-weight: 800; color: #7a7dff;'>
                    {tasa_anual_calc*100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rate3:
            st.markdown("<br>", unsafe_allow_html=True)
            aplicar_tasa = st.button("✓ Aplicar", key="apply_rate", use_container_width=False, type="primary")
        
        if aplicar_tasa:
            st.session_state["tc_ann_applied"] = tasa_anual_calc
            st.session_state["tasa_aplicada"] = True
            st.success("Tasa aplicada exitosamente")
        
        if not st.session_state.get("tasa_aplicada", False):
            st.warning("⏳ Aplique la tasa para continuar")
            st.stop()
        
        # === STEP 4: Risk Score ===
        st.markdown("<div class='section-header-enhanced'>📊 Paso 4: Calificación de Riesgo</div>", unsafe_allow_html=True)
        
        col_score1, col_score2 = st.columns([3, 1])
        
        with col_score1:
            score = st.slider(
                "Calificación del Cliente (1=Alto Riesgo, 5=Bajo Riesgo)",
                min_value=1.0, max_value=5.0, value=3.0, step=0.1
            )
        
        with col_score2:
            risk_level = "🔴 Alto" if score < 2.5 else "🟡 Medio" if score < 4.0 else "🟢 Bajo"
            risk_color = "#ff5ea0" if score < 2.5 else "#f2d17a" if score < 4.0 else "#00ffa3"
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; border-color: {risk_color};'>
                <div style='font-size: 2.5rem; font-weight: 800;'>{score:.1f}</div>
                <div style='font-size: 0.9rem; color: {risk_color}; margin-top: 0.5rem;'>
                    <strong>{risk_level} Riesgo</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # === STEP 5: Analyze ===
        st.markdown("<div class='section-header-enhanced'>🎯 Paso 5: Ejecutar Análisis</div>", unsafe_allow_html=True)
        
        col_btn = st.columns([1, 2, 1])
        with col_btn[1]:
            analizar = st.button("🚀 ANALIZAR RIESGO", type="primary", use_container_width=False, key="analyze")
        
        if analizar:
            with st.spinner("Calculando métricas de riesgo..."):
                time.sleep(0.8)
                resultado = calcular_resultados_ejecutivo(
                    score=score,
                    EAD=EAD_sel,
                    tc_ann=st.session_state['tc_ann_applied'],
                    garantias_usd=garantias_sel,
                    gastos_usd=gastos_sel
                )
            
            decision_ok = resultado["RE_anual_simple"] >= RET_THRESHOLD
            
            st.markdown("<br/>", unsafe_allow_html=True)
            decision_color = "#00ffa3" if decision_ok else "#ff5ea0"
            decision_icon = "✅" if decision_ok else "⛔"
            decision_text = "APROBAR CRÉDITO" if decision_ok else "RECHAZAR CRÉDITO"
            
            st.markdown(f"""
            <div class='metric-card pulse-glow' style='
                text-align: center; 
                padding: 3rem 2rem; 
                border: 3px solid {decision_color};
                background: linear-gradient(135deg, 
                    rgba({'0, 255, 163' if decision_ok else '255, 94, 160'}, 0.15) 0%, 
                    rgba({'0, 255, 163' if decision_ok else '255, 94, 160'}, 0.05) 100%);
            '>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>{decision_icon}</div>
                <div style='font-size: 2.5rem; font-weight: 900; color: {decision_color}; 
                    letter-spacing: 0.05em; text-transform: uppercase;'>
                    {decision_text}
                </div>
                <div style='font-size: 1.2rem; color: rgba(215, 226, 236, 0.85); margin-top: 1rem;'>
                    Retorno Esperado: <strong style='color: {decision_color};'>{resultado["RE_anual_simple"]*100:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br/><div class='section-header-enhanced'>📈 Métricas Clave</div>", unsafe_allow_html=True)
            
            colm1, colm2, colm3, colm4 = st.columns(4)
            
            metrics_data = [
                ("Probabilidad Default", f"{resultado['PD_12m']*100:.2f}%", "12 meses", "#ff5ea0", colm1),
                ("LGD", f"{resultado['LGD']*100:.2f}%", "Loss Given Default", "#f2d17a", colm2),
                ("Pérdida Dada Default", fmt_usd(resultado['PDD'], 0), "PDD", "#7a7dff", colm3),
                ("Retorno Esperado", f"{resultado['RE_anual_simple']*100:.2f}%", "Anual", decision_color, colm4)
            ]
            
            for label, value, subtitle, color, column in metrics_data:
                with column:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 4px solid {color};'>
                        <div style='font-size: 0.85rem; color: rgba(215, 226, 236, 0.7); 
                            text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>
                            {label}
                        </div>
                        <div style='font-size: 2rem; font-weight: 800; color: {color};'>
                            {value}
                        </div>
                        <div style='font-size: 0.8rem; color: rgba(215, 226, 236, 0.6); margin-top: 0.3rem;'>
                            {subtitle}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === STEP 6: Collateral Haircut ===
            st.markdown("<br/><div class='section-header-enhanced'>🛡️ Paso 6: Castigar Garantías en Caso de Default Definitivo </div>", unsafe_allow_html=True)
            
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
                garantia_bruta_edit = st.number_input(
                    "💎 Garantía Bruta (USD) — Valor Realizable",
                    min_value=0.0,
                    value=float(garantia_bruta_base),
                    step=10_000.0,
                    format="%.0f",
                    help="Valor de mercado de la garantía"
                )
            
            with col_edit2:
                peso_garantia_edit = st.number_input(
                    "⚖️ Peso de Garantía — Quality Collateral",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(peso_garantia_base),
                    step=0.01,
                    format="%.2f",
                    help="Factor de ajuste por calidad (0-1)"
                )
            
            garantia_castigada = garantia_bruta_edit * peso_garantia_edit
            
            st.markdown("<br/>", unsafe_allow_html=True)
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.85rem; color: rgba(215, 226, 236, 0.7); 
                        text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Garantía Bruta
                    </div>
                    <div style='font-size: 2rem; font-weight: 800; color: #00fff6;'>
                        {fmt_usd(garantia_bruta_edit, 0)}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(215, 226, 236, 0.6); margin-top: 0.3rem;'>
                        Valor Original
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.85rem; color: rgba(215, 226, 236, 0.7); 
                        text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Factor de Ajuste
                    </div>
                    <div style='font-size: 2rem; font-weight:800; color: #7a7dff;'>
                        {peso_garantia_edit:.2%}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(215, 226, 236, 0.6); margin-top: 0.3rem;'>
                        Quality Collateral
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res3:
                st.markdown(f"""
                <div class='metric-card' style='border: 2px solid #00ffa3;'>
                    <div style='font-size: 0.85rem; color: rgba(215, 226, 236, 0.7); 
                        text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Garantía Castigada
                    </div>
                    <div style='font-size: 2rem; font-weight: 800; color: #00ffa3;'>
                        {fmt_usd(garantia_castigada, 0)}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(215, 226, 236, 0.6); margin-top: 0.3rem;'>
                        Valor Final
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption(f"**Fórmula:** {fmt_usd(garantia_bruta_edit, 0)} × {peso_garantia_edit:.2%} = {fmt_usd(garantia_castigada, 0)}")
    
    except Exception as e:
        st.error(f"Error al leer archivo: {e}")
        st.info("Verifique que el archivo tenga el formato correcto")
        st.stop()
else:
    st.warning("⏳ Esperando archivo... Por favor cargue OPINT.xlsx para continuar")
    st.stop()

# Footer
st.markdown("<br/><br/>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<div style='text-align: center; color: rgba(215, 226, 236, 0.6); font-size: 0.85rem;'>
    <p>Aquila Risk Analysis System · EdeX-UI Theme</p>
    <p>© 2025 Juan José Mostajo León · Version 6.6-AQ</p>
</div>
""", unsafe_allow_html=True)
