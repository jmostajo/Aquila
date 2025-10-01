from __future__ import annotations
import streamlit as st
from fonts_embed import _embed_font_css

_embed_font_css()

# ===========================
# 2) FORZAR TEMA OSCURO CRÍTICO
# ===========================
DARK_THEME_CSS = """
<style>
:root {
  color-scheme: dark !important;

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {
  background-color: #0F1721 !important;
  color: #F8FAFC !important;
  transition: none !important;

div, section, header, main, aside, nav {
  background-color: transparent !important;

section[data-testid="stSidebar"] {
  background-color: #1A2332 !important;

.stApp {
  background: linear-gradient(135deg, #0F1721 0%, #1A2332 100%) !important;

[data-testid="stAppViewContainer"] > div:first-child {
  background-color: #0F1721 !important;
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


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

# --- Sidebar reset gate ---
with st.sidebar:
    if st.button("🔄 Reset Face Gate"):
        for k in ("entry_unlocked", "last_snapshot", "gate_seen_at"):
            st.session_state.pop(k, None)
        st.rerun()

# === FACE WELCOME OVERLAY =====================================================
WELCOME_CSS = """
<style>
.face-welcome-overlay { position: fixed; inset: 0; width: 100vw; height: 100vh;
  background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.92) 50%, rgba(15,23,42,0.98) 100%);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); z-index: 9999;
  display: flex; justify-content: center; align-items: center;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif; }
.face-welcome-glass { background: linear-gradient(135deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.05) 100%);
  backdrop-filter: blur(40px); -webkit-backdrop-filter: blur(40px);
  border: 1px solid rgba(255,255,255,0.15); border-radius: 24px; padding: 2.2rem;
  max-width: 90vw; width: 800px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1); }
.face-welcome-title { font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 1rem; letter-spacing: -0.02em;
  background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.face-welcome-subtitle { font-size: 1.05rem; color: rgba(255,255,255,0.75); text-align: center; margin-bottom: 1.4rem; }
.face-snapshot-container { border-radius: 16px; overflow: hidden; margin: 1.1rem 0; border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 25px -5px rgba(0,0,0,0.30); }
.face-badges-container { display: flex; flex-wrap: wrap; gap: .75rem; justify-content: center; margin: 1.1rem 0; }
.face-badge { padding: .45rem .9rem; border-radius: 999px; font-size: .85rem; font-weight: 600; letter-spacing: .02em; border: 1px solid; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
.badge-yunet { background: rgba(96,165,250,.15); border-color: rgba(96,165,250,.40); color: #93C5FD; }
.badge-secure { background: rgba(167,139,250,.15); border-color: rgba(167,139,250,.40); color: #C4B5FD; }
.badge-timestamp { background: rgba(244,114,182,.15); border-color: rgba(244,114,182,.40); color: #F9A8D4; }
.progress-bar-container { height: 6px; background: rgba(255,255,255,0.10); border-radius: 10px; overflow: hidden; margin: 1rem 0; }
.progress-bar-fill { height: 100%; background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6); border-radius: 10px; transition: width .25s ease; width: 0%; }
.progress-message { text-align: center; font-size: 1rem; color: rgba(255,255,255,0.95); font-weight: 500; margin-top: .4rem; }
@media (max-width: 768px) { .face-welcome-glass { padding: 1.4rem; margin: 1rem; max-width: 95vw; } .face-welcome-title { font-size: 2rem; } .face-badges-container { flex-direction: column; align-items: center; } }
</style>
"""

def _force_gate_flag() -> bool:
    try:
        v = st.query_params.get("force_gate")
        if isinstance(v, list):
            return "1" in v
        return str(v) == "1"
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            return "1" in qp.get("force_gate", [])
        except Exception:
            return False

def render_face_gate_reset() -> None:
    with st.sidebar:
        st.markdown("---")
        if st.button("🔄 Reset Face Gate", type="secondary", key="reset_face_gate_btn", use_container_width=True):
            for k in ("entry_unlocked", "last_snapshot", "gate_seen_at"):
                st.session_state.pop(k, None)
            st.rerun()

def face_welcome_gate_ui(auto_continue: bool = True, delay_sec: float = 1.8) -> None:
    ok = os.getenv("AQUILA_FACE_OK") == "1"
    snap_path = os.getenv("AQUILA_FACE_SNAPSHOT")
    force = os.getenv("AQUILA_FORCE_GATE") == "1" or _force_gate_flag()
    last_snapshot = st.session_state.get("last_snapshot")
    entry_unlocked = st.session_state.get("entry_unlocked", False)
    should_show = ok and (force or not entry_unlocked or (snap_path and snap_path != last_snapshot))
    if not should_show:
        return

    st.markdown(WELCOME_CSS, unsafe_allow_html=True)
    st.markdown('<div class="face-welcome-overlay"><div class="face-welcome-glass">', unsafe_allow_html=True)
    st.markdown('<div class="face-welcome-title">Face Detected — Welcome</div>', unsafe_allow_html=True)
    st.markdown('<div class="face-welcome-subtitle">Identity verified • Secure access granted</div>', unsafe_allow_html=True)

    if snap_path and os.path.exists(snap_path):
        try:
            img = Image.open(snap_path)
            st.markdown('<div class="face-snapshot-container">', unsafe_allow_html=True)
            st.image(img, use_container_width=True, caption="Snapshot capturado al aprobar el Face Gate")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar el snapshot: {e}")
    else:
        st.info("📷 No snapshot disponible")

    now_hms = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div class="face-badges-container">
          <div class="face-badge badge-yunet">YuNet • On-device</div>
          <div class="face-badge badge-secure">Aquila Secure Gate</div>
          <div class="face-badge badge-timestamp">Detected at {now_hms} • Live</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bar_ph = st.empty()
    msg_ph = st.empty()
    messages = [
        "🔍 Verificando integridad del sistema…",
        "🔐 Validando credenciales seguras…",
        "🎨 Preparando interfaz de usuario…",
        "🚀 Inicializando módulos principales…",
    ]
    t0 = time.time()
    while True:
        prog = min((time.time() - t0) / max(0.1, delay_sec), 1.0)
        bar_ph.markdown(
            f'<div class="progress-bar-container"><div class="progress-bar-fill" style="width:{prog*100:.0f}%"></div></div>',
            unsafe_allow_html=True,
        )
        idx = min(int(prog * len(messages)), len(messages) - 1)
        msg_ph.markdown(f'<div class="progress-message">{messages[idx]}</div>', unsafe_allow_html=True)
        if prog >= 1.0:
            break
        time.sleep(0.10)

    st.toast("Face detection approved ✅")
    st.balloons()
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.session_state["entry_unlocked"] = True
    st.session_state["last_snapshot"] = snap_path
    st.session_state["gate_seen_at"] = datetime.now().isoformat()

    if auto_continue:
        time.sleep(0.5)
        st.rerun()
    else:
        if st.button("🎯 Entrar a Aquila", type="primary", key="enter_app_btn"):
            st.rerun()
    st.stop()

render_face_gate_reset()
face_welcome_gate_ui(auto_continue=True, delay_sec=1.8)
# === FIN FACE WELCOME OVERLAY ================================================

# --- Vision/router (opcional) ---
vision_router = None
wait_for_face = None
try:
    from vision_detect import router as vision_router
except Exception as e1:
    try:
        from vision import router as vision_router  # type: ignore
    except Exception as e2:
        print(f"[init] Warning: Vision router not available (vision_detect: {e1}; vision: {e2})")

try:
    from facegate import wait_for_face  # type: ignore
except Exception as e:
    print(f"[init] Warning: Face-gate not available ({e})")
    wait_for_face = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    use_gate = os.getenv("FACE_GATE", "1") == "1"
    if use_gate:
        if wait_for_face is None:
            print("[face-gate] Skipped: facegate module not available")
        else:
            print("[face-gate] Initializing face detection...")
            ok = wait_for_face(
                device=int(os.getenv("CAMERA_INDEX", "0")),
                timeout=float(os.getenv("FACE_GATE_TIMEOUT", "10")),
                consec=int(os.getenv("FACE_GATE_CONSEC", "3")),
                width=int(os.getenv("CAM_WIDTH", "640")),
                height=int(os.getenv("CAM_HEIGHT", "480")),
                fps=int(os.getenv("CAM_FPS", "30")),
                score_thr=float(os.getenv("SCORE_THR", "0.6")),
            )
            if not ok:
                print("[face-gate] No face detected -> aborting")
                raise SystemExit(1)
    print("[server] Starting up..."); yield; print("[server] Shutting down...")

app = FastAPI(lifespan=lifespan)
if vision_router is not None:
    app.include_router(vision_router)
    print("[init] Vision detection endpoints mounted at /v1/vision")
else:
    print("[init] Vision endpoints not available")

# === Rutas/branding ===
BASE_DIR = Path(__file__).parent.resolve()
LOGO_CANDIDATES = [BASE_DIR / "Hexa.png", BASE_DIR / "assets" / "Hexa.png",
                   BASE_DIR / "static" / "Hexa.png", BASE_DIR / "images" / "Hexa.png"]

def _first_existing(paths):
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None

RESOLVED_LOGO = _first_existing(LOGO_CANDIDATES)
PAGE_ICON = str(RESOLVED_LOGO) if RESOLVED_LOGO else "📊"
st.set_page_config(
    page_title="Aquila — Análisis de Riesgo Crediticio",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<style>.material-icons, .material-icons-outlined{display:none!important;}</style>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_logo(path: Path | None):
    if not path: return None
    try: return Image.open(path)
    except Exception: return None

logo_img = load_logo(RESOLVED_LOGO)
AUTHOR_NAME = "Juan José Mostajo León"
AUTHOR_TAG = f"© {datetime.now().year} · {AUTHOR_NAME}"
SLOGAN_LINE = "Análisis de Riesgo Crediticio Inteligente"

# === Constantes y helpers ===
APP_VERSION = "6.6-AQ"
CO_ANUAL_FIJO = 0.015
TM_ANUAL_FIJO = 0.0075
PD12_ANCLA_1, PD12_ANCLA_5 = 0.80, 0.05
EPS = 1e-12

pio.templates["curay"] = dict(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        colorway=["#0C1B2A", "#CBA135", "#5F6B7A", "#18B277", "#E05F5F"]
    )
)
pio.templates.default = "curay"

css = """
<style>
/* ---------- FIXED LAYOUT: same look on all browsers ---------- */
:root{
  --sidebar-w: 300px;        /* fixed sidebar width */
  --content-w: 1280px;       /* fixed central content width */
  --content-max: 1400px;     /* optional cap for very large screens */
  --bg:#0F1721; --text:#F8FAFC; --text-muted:#94A3B8;
  --primary:#0C1B2A; --accent:#CBA135; --success:#18B277; --danger:#E05F5F;

/* Don’t collapse below our design width — show horizontal scroll instead */
html, body { min-width: calc(var(--sidebar-w) + var(--content-w) + 60px); }

/* Sidebar: fixed width */
section[data-testid="stSidebar"]{
  min-width: var(--sidebar-w) !important;
  max-width: var(--sidebar-w) !important;

/* Main content: fixed width, centered */
.main .block-container{
  max-width: var(--content-w) !important;
  min-width: var(--content-w) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 1.5rem !important;

/* On very large monitors, allow a gentle cap while keeping look */
@media (min-width: 1700px){
  .main .block-container{
    max-width: var(--content-max) !important;
    min-width: var(--content-max) !important;

/* Keep headings and widgets from jumping around */
h1, h2, h3, h4, h5, h6 { line-height: 1.2; }
.stSlider, .stNumberInput, .stButton, .stMetric { box-sizing: border-box; }

/* Your original theme bits (safe to keep) */
.stApp{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background: linear-gradient(135deg,#0F1721 0%,#1A2332 100%);
  color: var(--text);
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{ display:none; }

.exec-kpi{
  background: linear-gradient(135deg,rgba(255,255,255,0.06) 0%,rgba(255,255,255,0.02) 100%);
  border:1px solid rgba(255,255,255,0.12);
  border-radius:16px; padding:1.8rem 1.5rem; backdrop-filter:blur(10px);
  transition:transform .2s, box-shadow .2s;
.exec-kpi:hover{ transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,0.3); }
.kpi-label{ font-size:.8rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.75rem; font-weight:600; }
.kpi-value{ font-size:2.8rem; font-weight:800; color:var(--text); line-height:1; margin-bottom:.5rem; }

.decision-hero{
  background:linear-gradient(135deg,rgba(203,161,53,0.15) 0%,rgba(203,161,53,0.05) 100%);
  border:2px solid rgba(203,161,53,0.4); border-radius:24px; padding:3rem 2rem; margin:2rem 0; text-align:center;
.decision-result{ font-size:3.5rem; font-weight:900; margin:1.5rem 0; text-transform:uppercase; letter-spacing:.03em; }
.decision-accept{ color:var(--success); text-shadow:0 0 30px rgba(24,178,119,0.6); }
.decision-reject{ color:var(--danger); text-shadow:0 0 30px rgba(224,95,95,0.6); }

/* Optional: slightly scale down on Windows 125% zoom to fit better */
@media (min-width: 1200px){
  .main .block-container{ transform-origin: top center; }

/* Keep metrics and buttons tidy */
[data-testid="stMetricValue"]{ font-size:2rem; font-weight:700; }
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,var(--accent) 0%,#D4B15F 100%);
  color:var(--primary); border:none; border-radius:12px;
  padding:.9rem 2.5rem; font-size:1.1rem; font-weight:700; letter-spacing:.05em;
  box-shadow:0 4px 20px rgba(203,161,53,0.4); transition:all .3s;
.stButton>button[kind="primary"]:hover{ transform:translateY(-2px); box-shadow:0 6px 30px rgba(203,161,53,0.6); }
.section-header{ font-size:1.4rem; font-weight:700; color:var(--text); margin:2rem 0 1rem; padding-bottom:.5rem; border-bottom:2px solid rgba(203,161,53,0.3); }

/* Keep mobile tweaks minimal (you can delete if you prefer exact desktop look) */
@media (max-width: 768px){
  .kpi-value{ font-size:2rem; }
  .decision-result{ font-size:2.5rem; }
  .decision-hero{ padding:2rem 1rem; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# === KPI CARDS CSS (auto-injected) ===
KPI_CSS = """
<style>
.kpi-wrap{ display:flex; gap:1rem; }
.kpi-card{
  flex:1; background-color:#111827; border-left:6px solid #60A5FA;
  border-radius:12px; padding:1.25rem 1.25rem 1rem 1.25rem;
  box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}
.kpi-card .kpi-title{ font-size:0.95rem; font-weight:700; color:#93A3B8; margin-bottom:.35rem; }
.kpi-card .kpi-val{ font-size:2.1rem; font-weight:900; color:#F8FAFC; line-height:1.1; }
.kpi-card .kpi-sub{ font-size:.95rem; margin-top:.25rem; color:#cbd5e1; }
.kpi-delta.up{ color:#22C55E; font-weight:700; }
.kpi-delta.down{ color:#EF4444; font-weight:700; }
@media (max-width: 1024px){ .kpi-wrap{ flex-direction:column; } }
</style>
"""
st.markdown(KPI_CSS, unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    if logo_img is not None:
        st.image(logo_img, use_container_width=True)
    st.markdown("### AQUILA")
    st.caption(SLOGAN_LINE)
    st.divider()
    RET_THRESHOLD = st.number_input(
        "Umbral de Retorno Mínimo",
        min_value=0.0, max_value=1.0,
        value=0.12, step=0.01, format="%.2f",
        help="Retorno esperado mínimo para aprobar (12% default)"
    )
    st.divider()
    st.caption(AUTHOR_TAG)

# Header
col_logo, col_title = st.columns([1, 8])
with col_logo:
    if logo_img: st.image(logo_img, width=80)
with col_title:
    st.markdown("# AQUILA")
    st.caption("Sistema de Decisión de Riesgo Crediticio")
st.divider()

# === Utilidades core ===
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

def leer_peso_garantia_colM(
    row: pd.Series,
    fallback_colnames: tuple[str, ...] = ("Peso de la Garantía", "Peso de la Garantia")
) -> float:
    val = None
    try: val = row.iloc[12]  # Columna M (13ª)
    except Exception: val = None
    if val is None or (isinstance(val, float) and np.isnan(val)):
        for cname in fallback_colnames:
            if cname in row.index:
                val = row.get(cname); break
    try:
        s = str(val).strip()
        if s == "" or s.lower() == "nan": return 1.0
        s = s.replace("%", "").replace(",", ".")
        x = float(s)
        if x > 1.0: x = x / 100.0
        return float(np.clip(x, 0.0, 1.0))
    except Exception:
        return 1.0

# === Cálculo principal ===
def calcular_resultados_ejecutivo(
    score: float, EAD: float, tc_ann: float, garantias_usd: float, gastos_usd: float = 0.0
):
    lam1, lam5 = lambda_anchors()
    lam = lambda_from_score(score, lam1, lam5)

    PD_12m = 1.0 - np.exp(-lam)
    PD1 = pd_hazard_months(lam, 12)
    PD2_cond = pd_hazard_months(lam, 3)

    LGD = lgd_politica(garantias_usd, EAD)
    ECL = LGD * EAD

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
    return {"PD_12m": PD_12m, "LGD": LGD, "ECL": ECL, "RE_anual_simple": RE_anual_simple,
            "EV_VP": EV_VP, "multiplo_vp": multiplo_vp, "Texp": Texp}

# === Carga de cartera ===
@st.cache_data(show_spinner=False)
def load_cartera(uploaded):
    df = pd.read_excel(uploaded, sheet_name="CARTERA")
    df = df.rename(columns=lambda c: str(c).strip())
    return df

def demo_portfolio() -> pd.DataFrame:
    return pd.DataFrame({
        "Cliente": ["Empresa Alpha", "Empresa Beta", "Empresa Gamma"],
        "Exposición USD": [1_000_000, 750_000, 1_250_000],
        "GARANTIAS": [600_000, 300_000, 900_000],
    })

# === Página principal ===
st.markdown('<p class="section-header">📊 Análisis de Riesgo Crediticio</p>', unsafe_allow_html=True)
st.info("👤 **Guía para usuarios:** Siga los pasos en orden para realizar el análisis completo.")

# Paso 1: Cargar
st.markdown("### 📁 Paso 1: Cargar Cartera")
st.markdown("Arrastre y suelte el archivo **OPINT.xlsx** descargado o haga clic para seleccionarlo")
uploaded = st.file_uploader(
    "Seleccione o arrastre el archivo Excel",
    type=["xlsx"],
    help="Formato esperado: columnas 'Cliente', 'Exposición USD', 'GARANTIAS'",
    label_visibility="collapsed"
)

if uploaded:
    try:
        df_cart = load_cartera(uploaded)
        st.success("✅ Paso 1 completado: Cartera cargada exitosamente")

        # Paso 2: Seleccionar cliente
        st.markdown("---")
        st.markdown("### 👤 Paso 2: Seleccionar Cliente")

        cliente = st.selectbox(
            "Seleccione el cliente a analizar:",
            df_cart.iloc[:, 0],
            help="Elija el cliente de la lista cargada"
        )

        row = df_cart[df_cart.iloc[:, 0] == cliente].iloc[0]
        EAD_default = float(row.get("Exposición USD", 1_000_000.0))
        garantias_default = float(row.get("GARANTIAS", 600_000.0))

        st.success(f"✅ Paso 2 completado: Cliente '{cliente}' seleccionado")

        # ====== Inputs editables (con +/-) ======
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            EAD_sel = st.number_input(
                "EAD (USD)",
                min_value=0.0,
                value=float(EAD_default),
                step=10_000.0,
                format="%.0f",
                help="Exposición utilizada para el análisis"
            )
        with col_in2:
            gastos_sel = st.number_input(
                "Gastos (USD)",
                min_value=0.0,
                value=0.0,
                step=1_000.0,
                format="%.0f",
                help="Gastos asociados (opcional)"
            )
        with col_in3:
            garantias_sel = st.number_input(
                "Garantías (USD)",
                min_value=0.0,
                value=float(garantias_default),
                step=10_000.0,
                format="%.0f",
                help="Ajusta libremente la garantía antes del castigo"
            )

    except Exception as e:
        st.error(f"❌ Error al leer archivo: {e}")
        st.info("💡 Verifique que el archivo tenga el formato correcto (columnas: Cliente, Exposición USD, GARANTIAS)")
        st.stop()
else:
    st.warning("⏳ Esperando archivo... Por favor cargue **OPINT.xlsx** para continuar")
    st.stop()

# Paso 3: Tasa
st.markdown("---")
st.markdown("### 🧮 Paso 3: Configurar Tasa Compensatoria")
col_calc1, col_calc2, col_calc3 = st.columns([2, 2, 1])
with col_calc1:
    tasa_m_input = st.number_input(
        "Tasa Mensual (decimal)",
        min_value=0.0, max_value=1.0,
        value=0.025, step=0.001, format="%.4f",
        help="Ejemplo: 0.025 = 2.50% mensual"
    )
with col_calc2:
    tasa_anual_calc = to_annual_from_monthly(tasa_m_input)
    st.metric("Equivalente Anual", f"{tasa_anual_calc*100:.2f}%", help=f"Tasa efectiva anual: {tasa_anual_calc:.6f}")
with col_calc3:
    st.markdown("<br>", unsafe_allow_html=True)
    aplicar_tasa = st.button("✅ Aplicar", key="apply_calc", use_container_width=True, type="secondary")

if aplicar_tasa:
    st.session_state["tc_ann_applied"] = tasa_anual_calc
    st.session_state["tasa_aplicada"] = True
    st.success(f"✅ Paso 3 completado: Tasa anual {tasa_anual_calc*100:.2f}% aplicada para los cálculos")

if not st.session_state.get("tasa_aplicada", False):
    st.warning("⏳ Haga clic en '✅ Aplicar' para continuar al siguiente paso")
    st.stop()

# Paso 4: Score
st.markdown("---")
st.markdown("### 📊 Paso 4: Calificación de Riesgo del Cliente")
score = st.slider(
    "Seleccione la calificación (1=Alto Riesgo, 5=Bajo Riesgo)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.1,
    help="Calificación crediticia del cliente basada en análisis previo"
)
if score < 2.5:
    st.error(f"🔴 **Alto Riesgo** (Calificación: {score:.1f})")
elif score < 4.0:
    st.warning(f"🟡 **Riesgo Medio** (Calificación: {score:.1f})")
else:
    st.success(f"🟢 **Bajo Riesgo** (Calificación: {score:.1f})")
st.success("✅ Paso 4 completado: Calificación de riesgo definida")

# Paso 5: Ejecutar análisis
st.markdown("---")
st.markdown("### 🎯 Paso 5: Ejecutar Análisis de Riesgo")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analizar = st.button("🚀 ANALIZAR RIESGO", type="primary", use_container_width=True, key="analyze_btn")

if analizar:
    with st.spinner("🔄 Calculando métricas de riesgo..."):
        time.sleep(0.5)
        resultado = calcular_resultados_ejecutivo(
            score=score,
            EAD=EAD_sel,
            tc_ann=st.session_state['tc_ann_applied'],
            garantias_usd=garantias_sel,
            gastos_usd=gastos_sel
        )

    decision_ok = resultado["RE_anual_simple"] >= RET_THRESHOLD
    st.success("✅ Paso 5 completado: Análisis finalizado")

    st.markdown("---")
    st.markdown("## 📊 RESULTADO DEL ANÁLISIS")
    st.markdown(
        f'''
        <div class="decision-hero">
            <div style="font-size: 1.2rem; color: var(--text-muted); margin-bottom: 1rem;">
                DECISIÓN DE CRÉDITO
            </div>
            <div class="decision-result {'decision-accept' if decision_ok else 'decision-reject'}">
                {'✅ APROBAR CRÉDITO' if decision_ok else '⛔ RECHAZAR CRÉDITO'}
            </div>
            <div class="decision-subtitle">
                Retorno Esperado: <strong>{resultado["RE_anual_simple"]*100:.2f}%</strong>
                {'(Supera umbral mínimo)' if decision_ok else '(Por debajo del umbral)'}
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 📈 Métricas Clave del Análisis")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f'''<div class="exec-kpi"><div class="kpi-label">Probabilidad de Default</div>
                <div class="kpi-value">{resultado["PD_12m"]*100:.1f}%</div>
                <div class="kpi-context">Probabilidad de 12 meses</div></div>''',
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            f'''<div class="exec-kpi"><div class="kpi-label">Pérdida por Default</div>
                <div class="kpi-value">{round(resultado["LGD"]*100, 2):.2f}%</div>
                <div class="kpi-context">LGD (Loss Given Default)</div></div>''',
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            f'''<div class="exec-kpi"><div class="kpi-label">Pérdida Esperada</div>
                <div class="kpi-value">{fmt_usd(resultado["ECL"],2)}</div>
                <div class="kpi-context">Expected Credit Loss</div></div>''',
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            f'''<div class="exec-kpi"><div class="kpi-label">Retorno Esperado</div>
                <div class="kpi-value" style="color: {'var(--success)' if decision_ok else 'var(--danger)'};">
                    {resultado["RE_anual_simple"]*100:.2f}%</div>
                <div class="kpi-context">Umbral: {RET_THRESHOLD*100:.0f}%</div></div>''',
            unsafe_allow_html=True
        )

    # ========== PASO 6: CASTIGAR GARANTÍAS EN CASO DE DEFAULT DEFINITIVO (INTERACTIVO) ==========
st.markdown("---")
st.markdown("### 🛡️ Paso 6: Castigar garantías en caso de default definitivo")

# 1) Leemos el peso base desde la hoja (columna M) con tolerancia
try:
    peso_base_colM = leer_peso_garantia_colM(row)   # lee columna M de la hoja CARTERA
except Exception:
    peso_base_colM = 1.0

# 2) Inicializamos valores editables en session_state (solo 1 vez)
if "garantia_bruta_editable" not in st.session_state:
    st.session_state.garantia_bruta_editable = float(garantias_default)

if "peso_garantia_editable" not in st.session_state:
    st.session_state.peso_garantia_editable = float(peso_base_colM)

# 3) Controles con +/- (st.number_input ya trae steppers)
col_edit1, col_edit2 = st.columns(2)
with col_edit1:
    garantia_bruta_edit = st.number_input(
        "Garantía bruta (USD) — editable",
        min_value=0.0,
        value=float(st.session_state.garantia_bruta_editable),
        step=10_000.0,  # ajusta el tamaño del “+ / -” a tu gusto
        format="%.0f",
        help="Valor de garantía total del cliente. Úsalo para simular mayor/menor calidad de colateral."
    )
with col_edit2:
    peso_garantia_edit = st.number_input(
        "Peso de la garantía (col. M) — editable",
        min_value=0.0, max_value=1.0,
        value=float(st.session_state.peso_garantia_editable),
        step=0.01,      # “+ / -” de 0.01
        format="%.2f",
        help="Factor de calidad de colateral (0 a 1). Proviene de la columna M, pero puedes ajustarlo."
    )

# 4) Calculamos garantía castigada en vivo
garantia_castigada_edit = float(garantia_bruta_edit) * float(peso_garantia_edit)

# 5) Mostramos tarjetas con deltas vs. valores base (para que se vea cómo cambia)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        "Garantía bruta (editable)",
        f"${garantia_bruta_edit:,.0f}",
        delta=f"{garantias_default:,.0f} base"
    )
with c2:
    st.metric(
        "Peso de garantía (editable)",
        f"{peso_garantia_edit:.2f}",
        delta=f"{peso_base_colM:.2f} base"
    )
with c3:
    st.metric(
        "Garantía castigada (default)",
        f"${garantia_castigada_edit:,.0f}"
    )

st.caption("Regla: Garantía castigada = Garantía bruta × Peso de la garantía (quality collateral, columna M).")

# 6) Guardamos para auditoría / uso downstream (si luego quieres usarlo en PV_t2_def o LGD)
st.session_state.update({
    "peso_garantia_colM": float(peso_garantia_edit),
    "garantia_castigada_default": float(garantia_castigada_edit),
    "garantia_bruta_editable": float(garantia_bruta_edit),
    "peso_garantia_editable": float(peso_garantia_edit),
})


# Footer auditoría
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.caption(f"**Versión:** {APP_VERSION}")
with col_info2:
    st.caption(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d')}")
with col_info3:
    if "last_analysis_time" in st.session_state:
        st.caption(f"**Último análisis:** {st.session_state['last_analysis_time']}")
# deploy Tue Sep 30 23:49:06 UTC 2025

# deploy 2025-10-01T01:45:23Z


