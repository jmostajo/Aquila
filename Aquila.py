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

_embed_font_css()

# ===========================
# ENHANCED DARK THEME WITH MODERN AESTHETICS
# ===========================
DARK_THEME_CSS = """
<style>
:root {
  color-scheme: dark !important;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {
  background-color: #0F1721 !important;
  color: #F8FAFC !important;
  transition: none !important;
}

div, section, header, main, aside, nav {
  background-color: transparent !important;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1A2332 0%, #0F1721 100%) !important;
  border-right: 1px solid rgba(96, 165, 250, 0.1) !important;
}

.stApp {
  background: radial-gradient(ellipse at top, #1e3a5f 0%, #0F1721 50%, #0a0f1a 100%) !important;
}

[data-testid="stAppViewContainer"] > div:first-child {
  background-color: transparent !important;
}

/* Animated gradient background effect */
@keyframes gradient-shift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

.stApp::before {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(45deg, 
    rgba(96, 165, 250, 0.03) 0%, 
    rgba(167, 139, 250, 0.03) 25%,
    rgba(244, 114, 182, 0.03) 50%,
    rgba(96, 165, 250, 0.03) 75%,
    rgba(167, 139, 250, 0.03) 100%);
  background-size: 400% 400%;
  animation: gradient-shift 15s ease infinite;
  pointer-events: none;
  z-index: 0;
}

/* Enhanced metric cards */
.metric-card {
  background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 33, 0.9) 100%);
  border: 1px solid rgba(96, 165, 250, 0.2);
  border-radius: 16px;
  padding: 1.5rem;
  backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.metric-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.metric-card:hover::before {
  opacity: 1;
}

.metric-card:hover {
  transform: translateY(-4px);
  border-color: rgba(96, 165, 250, 0.4);
  box-shadow: 0 12px 48px rgba(96, 165, 250, 0.2);
}

/* Enhanced input fields */
.stNumberInput input, .stSelectbox select {
  background: rgba(15, 23, 33, 0.6) !important;
  border: 1px solid rgba(96, 165, 250, 0.2) !important;
  border-radius: 10px !important;
  color: #F8FAFC !important;
  padding: 0.75rem !important;
  transition: all 0.3s ease !important;
}

.stNumberInput input:focus, .stSelectbox select:focus {
  border-color: rgba(96, 165, 250, 0.5) !important;
  box-shadow: 0 0 20px rgba(96, 165, 250, 0.15) !important;
}

/* Enhanced slider */
.stSlider [data-baseweb="slider"] {
  background: linear-gradient(to right, 
    rgb(239, 68, 68) 0%, 
    rgb(251, 146, 60) 25%, 
    rgb(234, 179, 8) 50%, 
    rgb(132, 204, 22) 75%, 
    rgb(34, 197, 94) 100%) !important;
  height: 8px !important;
  border-radius: 10px !important;
}

/* Enhanced buttons */
.stButton button {
  font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

.stButton button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 30px rgba(96, 165, 250, 0.3) !important;
}

/* Section headers with gradient underline */
.section-header-enhanced {
  font-size: 1.8rem;
  font-weight: 700;
  color: #F8FAFC;
  margin: 2.5rem 0 1.5rem;
  padding-bottom: 0.75rem;
  position: relative;
}

.section-header-enhanced::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, transparent 100%);
  border-radius: 10px;
}

/* Info boxes with icons */
.info-box {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
  border-left: 4px solid #60A5FA;
  border-radius: 12px;
  padding: 1rem 1.5rem;
  margin: 1rem 0;
  backdrop-filter: blur(10px);
}

/* Pulse animation for important elements */
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 20px rgba(96, 165, 250, 0.2); }
  50% { box-shadow: 0 0 40px rgba(96, 165, 250, 0.4); }
}

.pulse-glow {
  animation: pulse-glow 2s ease-in-out infinite;
}
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

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

# === Page config ===
st.set_page_config(
    page_title="Aquila — Análisis de Riesgo Crediticio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Sidebar ===
with st.sidebar:
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

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%); 
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;'>
        AQUILA
    </h1>
    <p style='font-size: 1.2rem; color: rgba(248, 250, 252, 0.7);'>
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
                <div style='font-size: 2.5rem; font-weight: 800; color: #22C55E;'>✓</div>
                <div style='font-size: 0.9rem; color: rgba(248, 250, 252, 0.7); margin-top: 0.5rem;'>
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
                <div style='font-size: 0.85rem; color: rgba(248, 250, 252, 0.6); margin-bottom: 0.5rem;'>
                    EQUIVALENTE ANUAL
                </div>
                <div style='font-size: 2rem; font-weight: 800; color: #A78BFA;'>
                    {tasa_anual_calc*100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rate3:
            st.markdown("<br>", unsafe_allow_html=True)
            aplicar_tasa = st.button("✓ Aplicar", key="apply_rate", use_container_width=True, type="primary")
        
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
            risk_color = "#EF4444" if score < 2.5 else "#EAB308" if score < 4.0 else "#22C55E"
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
            analizar = st.button("🚀 ANALIZAR RIESGO", type="primary", use_container_width=True, key="analyze")
        
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
            decision_color = "#22C55E" if decision_ok else "#EF4444"
            decision_icon = "✅" if decision_ok else "⛔"
            decision_text = "APROBAR CRÉDITO" if decision_ok else "RECHAZAR CRÉDITO"
            
            st.markdown(f"""
            <div class='metric-card pulse-glow' style='
                text-align: center; 
                padding: 3rem 2rem; 
                border: 3px solid {decision_color};
                background: linear-gradient(135deg, 
                    rgba({'34, 197, 94' if decision_ok else '239, 68, 68'}, 0.15) 0%, 
                    rgba({'34, 197, 94' if decision_ok else '239, 68, 68'}, 0.05) 100%);
            '>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>{decision_icon}</div>
                <div style='font-size: 2.5rem; font-weight: 900; color: {decision_color}; 
                    letter-spacing: 0.05em; text-transform: uppercase;'>
                    {decision_text}
                </div>
                <div style='font-size: 1.2rem; color: rgba(248, 250, 252, 0.8); margin-top: 1rem;'>
                    Retorno Esperado: <strong style='color: {decision_color};'>{resultado["RE_anual_simple"]*100:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br/><div class='section-header-enhanced'>📈 Métricas Clave</div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            metrics_data = [
                ("Probabilidad Default", f"{resultado['PD_12m']*100:.2f}%", "12 meses", "#EF4444", col1),
                ("LGD", f"{resultado['LGD']*100:.2f}%", "Loss Given Default", "#F59E0B", col2),
                ("Pérdida Esperada", fmt_usd(resultado['ECL'], 0), "ECL", "#A78BFA", col3),
                ("Retorno Esperado", f"{resultado['RE_anual_simple']*100:.2f}%", "Anual", decision_color, col4)
            ]
            
            for label, value, subtitle, color, column in metrics_data:
                with column:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 4px solid {color};'>
                        <div style='font-size: 0.85rem; color: rgba(248, 250, 252, 0.6); 
                            text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>
                            {label}
                        </div>
                        <div style='font-size: 2rem; font-weight: 800; color: {color};'>
                            {value}
                        </div>
                        <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5); margin-top: 0.3rem;'>
                            {subtitle}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === STEP 6: Collateral Haircut ===
            st.markdown("<br/><div class='section-header-enhanced'>🛡️ Paso 6: Castigar Garantías</div>", unsafe_allow_html=True)
            
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
                    <div style='font-size: 0.85rem; color: rgba(248, 250, 252, 0.6); 
                        text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Garantía Bruta
                    </div>
                    <div style='font-size: 2rem; font-weight: 800; color: #60A5FA;'>
                        {fmt_usd(garantia_bruta_edit, 0)}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5); margin-top: 0.3rem;'>
                        Valor Original
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.85rem; color: rgba(248, 250, 252, 0.6); 
                        text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Factor de Ajuste
                    </div>
                    <div style='font-size: 2rem; font-weight:800; color: #A78BFA;'>
                        {peso_garantia_edit:.2%}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5); margin-top: 0.3rem;'>
                        Quality Collateral
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res3:
                st.markdown(f"""
                <div class='metric-card' style='border: 2px solid #22C55E;'>
                    <div style='font-size: 0.85rem; color: rgba(248, 250, 252, 0.6); 
                        text-transform: uppercase; margin-bottom: 0.5rem;'>
                        Garantía Castigada
                    </div>
                    <div style='font-size: 2rem; font-weight: 800; color: #22C55E;'>
                        {fmt_usd(garantia_castigada, 0)}
                    </div>
                    <div style='font-size: 0.8rem; color: rgba(248, 250, 252, 0.5); margin-top: 0.3rem;'>
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
<div style='text-align: center; color: rgba(248, 250, 252, 0.5); font-size: 0.85rem;'>
    <p>Aquila Risk Analysis System · Powered by Advanced Credit Modeling</p>
    <p>© 2025 Juan José Mostajo León · Version 6.6-AQ</p>
</div>
""", unsafe_allow_html=True)
