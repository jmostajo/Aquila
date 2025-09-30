from __future__ import annotations
# ==== Imports base ====
import os, shutil, tempfile, subprocess, time, uuid, html
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Dict, Tuple, Union
from datetime import datetime
from io import BytesIO
import secrets
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from PIL import Image
from scipy.linalg import eig, inv, expm, solve
from scipy.integrate import solve_ivp
from numpy.typing import ArrayLike
import hashlib


# ========= Resolución robusta de rutas (antes de set_page_config) =========
BASE_DIR = Path(__file__).parent.resolve()

# Probaremos varias ubicaciones comunes
LOGO_CANDIDATES = [
    BASE_DIR / "Hexa.png",
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
def load_logo_bytes(path: Optional[Path]) -> Optional[bytes]:
    """Carga el logo como bytes, tolerante si no existe."""
    if path is None or not path.exists():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None

logo_bytes = load_logo_bytes(RESOLVED_LOGO)

# ---------- Streamlit: DEBE ser lo primero de Streamlit ----------
PAGE_ICON = str(RESOLVED_LOGO) if RESOLVED_LOGO else "📊"
st.set_page_config(
    page_title="Aquila — Financial Risk & PD",
    page_icon=PAGE_ICON,
    layout="wide",
)

# ---------- Metadatos ----------
AUTHOR_NAME = "Juan José Mostajo León"
AUTHOR_TAG = f"© {datetime.now().year} · {AUTHOR_NAME} · Autor intelectual"
SLOGAN_LINE = "Análisis Quantitativo Unificado Inteligente en Líneas de Crédito"
SLOGAN_ACRONYM = (
    "<span style='color:#f59e0b;font-weight:700'>A</span>nálisis "
    "<span style='color:#f59e0b;font-weight:700'>Q</span>uantitativo "
    "<span style='color:#f59e0b;font-weight:700'>U</span>nificado "
    "<span style='color:#f59e0b;font-weight:700'>I</span>nteligente en "
    "<span style='color:#f59e0b;font-weight:700'>L</span>íneas de "
    "<span style='color:#f59e0b;font-weight:700'>A</span>crédito"
)

# ---------- Estilos ----------
st.markdown("""
<style>
  .block-container { padding-top: 1.0rem; }
  [data-testid="stSidebar"] img { margin-bottom: 0.6rem; }
  .aquila-badge {
    display:inline-block; padding:.25rem .6rem; border-radius:999px;
    border:1px solid #334155; background:#0b1220; color:#9ca3af;
    font-size:.8rem; margin-right:.4rem
  }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    if logo_bytes:
        st.image(logo_bytes, use_container_width=True)
    else:
        st.info("🔎 Coloca **Hexa.png** junto a *Aquila.py* o en **assets/**.")
    st.markdown("### AQUILA")
    st.caption(SLOGAN_LINE)
    st.markdown(SLOGAN_ACRONYM, unsafe_allow_html=True)
    st.divider()
    st.caption("Probability of Default • Risk Analytics")
    st.markdown(AUTHOR_NAME)
    st.caption(AUTHOR_TAG)

# ---------- Header ----------
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if logo_bytes:
        st.image(logo_bytes, use_container_width=True)
with col_title:
    st.markdown("## AQUILA")
    st.caption(SLOGAN_LINE)
    st.markdown(
        '<span class="aquila-badge">IFRS-9 Ready</span>'
        '<span class="aquila-badge">v6.5 • AQ-Neuro</span>'
        '<span class="aquila-badge">Transparencia total</span>',
        unsafe_allow_html=True
    )

# ====== A partir de aquí continúa tu lógica de negocio / app ======

# ---------- Demo content (replace with your app) ----------
st.divider()
ScalarOrFunc = Union[float, Callable[[float], float]]

def _as_callable(x: ScalarOrFunc) -> Callable[[float], float]:
    """Return a(t) callable given a scalar or a function."""
    if callable(x):
        return x
    val = float(x)
    return lambda _: val

def build_T(alpha: float, gamma: float, delta: float) -> np.ndarray:
    r"""
    Build the one-period transition matrix

        T = [[1-α, α,   0],
             [ γ , 0,   δ],
             [ 0 , 0,   1]]
    """
    T = np.array([[1.0 - alpha, alpha, 0.0],
                  [gamma,       0.0,   delta],
                  [0.0,         0.0,   1.0]], dtype=float)
    # Keep rows stochastic (light guard)
    T[0] /= max(T[0].sum(), 1.0)
    # row 1 already sums to gamma + delta -> we accept as given
    return T

@dataclass
class EigenDecomp:
    values: np.ndarray      # λ1..λ3
    right_vecs: np.ndarray  # columns are eigenvectors
    inv_right: np.ndarray
    stationary_index: int
    transient_indices: Tuple[int, int]

def eigen_decompose(T: np.ndarray, rtol: float = 1e-10) -> EigenDecomp:
    """Eigendecomposition for time-invariant T. Identifies the λ≈1 stationary eigenvalue."""
    w, V = eig(T, left=False, right=True)
    if np.allclose(w.imag, 0, rtol):
        w = w.real
        V = V.real
    idx_stationary = int(np.argmin(np.abs(w - 1.0)))
    trans = tuple(sorted(set(range(3)) - {idx_stationary}))
    Vinv = inv(V)
    return EigenDecomp(values=w, right_vecs=V, inv_right=Vinv,
                       stationary_index=idx_stationary,
                       transient_indices=trans)

def transient_roots_closed_form(alpha: float, gamma: float) -> Tuple[float, float]:
    r"""Solve λ^2 + (α-1)λ - αγ = 0  →  (λ_minus, λ_plus)"""
    a = 1.0
    b = (alpha - 1.0)
    c = -alpha * gamma
    disc = max(b*b - 4*a*c, 0.0)
    s = np.sqrt(disc)
    l1 = (-b - s) / 2.0
    l2 = (-b + s) / 2.0
    return (min(l1, l2), max(l1, l2))

def p_t_via_powers(p0: ArrayLike, T: np.ndarray, t: int) -> np.ndarray:
    """p_t = p0 · T^t for integer t≥0 (fast matrix power)."""
    p0 = np.asarray(p0, dtype=float).reshape(1, 3)
    if t == 0:
        return p0.copy()
    return p0 @ np.linalg.matrix_power(T, int(t))

def p_t_via_eig(p0: ArrayLike, ed: EigenDecomp, t: Union[int, float]) -> np.ndarray:
    """p_t = p0 · V · Λ^t · V^{-1} (time-invariant T)."""
    p0 = np.asarray(p0, dtype=float).reshape(1, 3)
    lambdas = ed.values
    diag = np.diag([float(l)**float(t) if l != 0 else (0.0 if t > 0 else 1.0) for l in lambdas])
    return p0 @ ed.right_vecs @ diag @ ed.inv_right

def p_t_linear_combo(p0: ArrayLike, ed: EigenDecomp, t: Union[int, float]) -> Dict[str, float]:
    """Return dict with pP(t), pS1(t), pS2(t) computed via eig method."""
    v = p_t_via_eig(p0, ed, t).ravel()
    return {"pP(t)": float(v[0]), "pS1(t)": float(v[1]), "pS2(t)": float(v[2])}

def EV_from_probs(p_t: ArrayLike, PV: ArrayLike) -> float:
    """EV(t) = Σ_i p_i(t)·PV_i"""
    p = np.asarray(p_t, dtype=float).ravel()
    PV = np.asarray(PV, dtype=float).ravel()
    return float(p @ PV)

def EV_path(p0: ArrayLike, T: np.ndarray, PV: ArrayLike, t_grid: Sequence[int]) -> pd.DataFrame:
    """DataFrame with columns: t, pP, pS1, pS2, EV."""
    rows = []
    for t in t_grid:
        pt = p_t_via_powers(p0, T, t).ravel()
        rows.append({"t": t, "pP": pt[0], "pS1": pt[1], "pS2": pt[2], "EV": EV_from_probs(pt, PV)})
    return pd.DataFrame(rows)

def finite_diff_sensitivities(
    p0: ArrayLike,
    alpha: float, gamma: float, delta: float,
    PV: ArrayLike,
    t: int,
    eps: float = 1e-6
) -> Dict[str, Dict[str, float]]:
    """
    Central differences of probabilities and EV wrt α, γ, δ at horizon t.
    Returns keys: dp/dalpha, dp/dgamma, dp/ddelta, dEV/dalpha, dEV/dgamma, dEV/ddelta.
    """
    def bump(a, g, d, which, h):
        if which == "alpha":
            a += h
        if which == "gamma":
            g += h
        if which == "delta":
            d += h
        p = p_t_via_powers(p0, build_T(a, g, d), t).ravel()
        return p, EV_from_probs(p, PV)

    out: Dict[str, Dict[str, float]] = {}
    for name in ("alpha", "gamma", "delta"):
        p_plus, EV_plus = bump(alpha, gamma, delta, name, +eps)
        p_minus, EV_minus = bump(alpha, gamma, delta, name, -eps)
        dp = (p_plus - p_minus) / (2 * eps)
        dEV = (EV_plus - EV_minus) / (2 * eps)
        out[f"dp/d{name}"] = {"pP": float(dp[0]), "pS1": float(dp[1]), "pS2": float(dp[2])}
        out[f"dEV/d{name}"] = float(dEV)
    return out

def generator_Q(alpha: float, gamma: float, delta: float) -> np.ndarray:
    r"""
    Continuous-time generator:

        dP/dt  = -α P + γ S1
        dS1/dt =  α P - (γ+δ) S1
        dS2/dt =  δ S1

    Q =
      [[-α,    α,    0],
       [ γ, -(γ+δ),  δ],
       [ 0,    0,    0]]
    """
    return np.array([[-alpha,     alpha,       0.0],
                     [ gamma, -(gamma+delta),  delta],
                     [ 0.0,        0.0,        0.0]], dtype=float)

def backward_EV_ode(
    t: float, v: np.ndarray,
    alpha_t: ScalarOrFunc, gamma_t: ScalarOrFunc, delta_t: ScalarOrFunc
) -> np.ndarray:
    """Backward ODE (no discount): ∂v/∂t = -Q(t) v, with v = [EV_P, EV_S1, EV_S2]^T."""
    a = _as_callable(alpha_t)(t)
    g = _as_callable(gamma_t)(t)
    d = _as_callable(delta_t)(t)
    Q = generator_Q(a, g, d)
    return -(Q @ v)

def solve_backward_EV(
    T_horizon: float,
    EV_T: ArrayLike,
    alpha_t: ScalarOrFunc,
    gamma_t: ScalarOrFunc,
    delta_t: ScalarOrFunc,
    t_eval: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve v'(t) = -Q(t)v(t),  v(T)=EV_T, for t∈[0,T]. Returns (t_grid, v(t))."""
    EV_T = np.asarray(EV_T, dtype=float).ravel()
    if t_eval is None:
        t_eval = np.linspace(0.0, float(T_horizon), 121)
    sol = solve_ivp(
        fun=lambda tt, vv: backward_EV_ode(tt, vv, alpha_t, gamma_t, delta_t),
        t_span=(T_horizon, 0.0), y0=EV_T, t_eval=np.flip(t_eval),
        rtol=1e-8, atol=1e-10
    )
    if not sol.success:
        raise RuntimeError(f"Backward ODE failed: {sol.message}")
    return np.flip(sol.t), np.flip(sol.y.T, axis=0)  # increasing t, shape (n,3)

def solve_v_matrix_IF_piecewise(
    t_grid: ArrayLike,
    Q_list: Sequence[np.ndarray],
    EV_T: ArrayLike,
    r_list: Optional[Sequence[float]] = None,
    c_list: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soluciona v'(t) = -(Q(t) - r(t) I) v(t) - c(t),  v(T)=EV_T,
    usando factor integrante matricial en malla de *tramos constantes*.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    assert np.all(np.diff(t_grid) > 0), "t_grid debe ser estrictamente creciente"

    N = len(t_grid) - 1
    EV_T = np.asarray(EV_T, dtype=float).ravel()
    n = EV_T.shape[0]

    if r_list is None:
        r_list = [0.0] * N
    if c_list is None:
        c_list = [np.zeros(n)] * N

    assert len(Q_list) == N and len(r_list) == N and len(c_list) == N

    I = np.eye(n)
    v_grid = np.zeros((N + 1, n), dtype=float)
    v_grid[-1] = EV_T  # v(T)

    for k in range(N - 1, -1, -1):
        dt = t_grid[k + 1] - t_grid[k]
        Qk = np.asarray(Q_list[k], dtype=float)
        rk = float(r_list[k])
        ck = np.asarray(c_list[k], dtype=float)

        A = Qk - rk * I
        Phi = expm(-dt * A)

        # término particular: A^{-1}(I-Φ)c, usando solve por estabilidad numérica
        rhs = (I - Phi) @ ck
        try:
            part = solve(A, rhs, assume_a="gen")
        except Exception:
            # fallback: ∫_0^dt exp(-τA) c dτ ≈ dt * 0.5 * [I + exp(-dt A)] c
            part = 0.5 * dt * (ck + (Phi @ ck))

        v_grid[k] = Phi @ v_grid[k + 1] + part

    return t_grid, v_grid

def build_Q_piecewise_from_funcs(
    t_grid: ArrayLike,
    alpha_t: ScalarOrFunc,
    gamma_t: ScalarOrFunc,
    delta_t: ScalarOrFunc,
) -> Sequence[np.ndarray]:
    """Construye Q_list (piecewise-constant) evaluando α,γ,δ en el extremo derecho de cada tramo."""
    t = np.asarray(t_grid, dtype=float)
    Q_list = []
    for k in range(len(t) - 1):
        tk1 = t[k + 1]  # right-end sampling
        a = _as_callable(alpha_t)(tk1)
        g = _as_callable(gamma_t)(tk1)
        d = _as_callable(delta_t)(tk1)
        Q_list.append(generator_Q(a, g, d))
    return Q_list

# ============================== Plots ==============================

def plot_prob_paths(df: pd.DataFrame, title: str = "Multi-period probabilities"):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.plot(df["t"], df["pP"],  label="pP(t)")
    ax.plot(df["t"], df["pS1"], label="pS1(t)")
    ax.plot(df["t"], df["pS2"], label="pS2(t)")
    ax.set_xlabel("t (periods)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return fig

def plot_EV(df: pd.DataFrame, title: str = "EV(t)"):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    ax.plot(df["t"], df["EV"], label="EV(t)")
    ax.set_xlabel("t (periods)")
    ax.set_ylabel("Expected Value")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return fig

def plot_sensitivity_bars(sens: Dict[str, Dict[str, float]]):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    # dEV
    dEV = [sens["dEV/dalpha"], sens["dEV/dgamma"], sens["dEV/ddelta"]]
    ax[0].bar(["α", "γ", "δ"], dEV)
    ax[0].set_title("∂EV/∂θ")
    ax[0].grid(alpha=0.3)
    # dp for pS2
    dpS2 = [sens["dp/dalpha"]["pS2"], sens["dp/dgamma"]["pS2"], sens["dp/ddelta"]["pS2"]]
    ax[1].bar(["α", "γ", "δ"], dpS2)
    ax[1].set_title("∂pS2/∂θ")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig

# ====================== RENDER: PD Algorithm Tab ======================

def render_aquilaeigen_tab():
    # ---------------- Controls ----------------
    st.subheader("1) Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha = st.number_input("α (P→S1)", min_value=0.0, max_value=1.0,
                                value=0.10, step=0.01, format="%.3f")
    with c2:
        gamma = st.number_input("γ (S1→P)", min_value=0.0, max_value=1.0,
                                value=0.05, step=0.01, format="%.3f")
    with c3:
        delta = st.number_input("δ (S1→S2)", min_value=0.0, max_value=1.0,
                                value=0.20, step=0.01, format="%.3f")
    with c4:
        horizon = st.number_input("Max periods (t_max)", min_value=1, max_value=240,
                                  value=24, step=1)

    c5, c6 = st.columns(2)
    with c5:
        p0_choice = st.selectbox("Initial distribution p0",
                                 ["Start in P", "Custom"], index=0)
    with c6:
        PV_choice = st.selectbox("State PVs (cashflows)",
                                 ["[100, 60, 20]", "Custom"], index=0)

    if p0_choice == "Start in P":
        p0 = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        try:
            p0 = np.array(
                [float(x) for x in st.text_input("p0 as three numbers (sum≈1)", "1,0,0").split(",")],
                dtype=float
            )
        except Exception:
            p0 = np.array([1.0, 0.0, 0.0], dtype=float)
        s = max(p0.sum(), 1e-12)
        p0 = p0 / s

    if PV_choice == "[100, 60, 20]":
        PV = np.array([100.0, 60.0, 20.0], dtype=float)
    else:
        try:
            PV = np.array(
                [float(x) for x in st.text_input("PV vector (P,S1,S2)", "100,60,20").split(",")],
                dtype=float
            )
        except Exception:
            PV = np.array([100.0, 60.0, 20.0], dtype=float)

    # ---------------- Transition Matrix ----------------
    st.subheader("2) Transition Matrix")
    T = build_T(alpha, gamma, delta)
    st.latex(r"""
T(\alpha,\gamma,\delta)=
\begin{bmatrix}
1-\alpha & \alpha & 0\\
\gamma & 0 & \delta\\
0 & 0 & 1
\end{bmatrix}
""")
    st.dataframe(pd.DataFrame(T, index=["P","S1","S2"], columns=["P","S1","S2"]),
                 use_container_width=True)

    # ---------------- Eigen Decomposition ----------------
    st.subheader("3) Eigen Decomposition")
    st.latex(r"\text{Find } \lambda, v \text{ such that } T v = \lambda v.")
    ed = eigen_decompose(T)
    lam_df = pd.DataFrame({
        "λ": ed.values,
        "role": ["stationary" if i == ed.stationary_index else "transient" for i in range(3)]
    })
    st.dataframe(lam_df, use_container_width=True)
    st.markdown("**Right eigenvectors (columns of V)**")
    st.dataframe(pd.DataFrame(ed.right_vecs, index=["P","S1","S2"],
                              columns=[f"v{i+1}" for i in range(3)]),
                 use_container_width=True)
    st.markdown("**Closed-form transient roots** from $\\lambda^2 + (\\alpha-1)\\lambda - \\alpha\\gamma=0$")
    lam_minus, lam_plus = transient_roots_closed_form(alpha, gamma)
    st.write({"λ_minus": float(lam_minus), "λ_plus": float(lam_plus)})

    # ---------------- Multi-period probabilities ----------------
    st.subheader("4) Multi-period probabilities")
    st.latex(r"p_t = p_0\,T^t = p_0\,V\,\Lambda^t V^{-1}")
    t_grid = np.arange(0, int(horizon) + 1, dtype=int)
    df_path = EV_path(p0, T, PV, t_grid)
    st.dataframe(
        df_path.style.format({"pP": "{:.4f}", "pS1": "{:.4f}", "pS2": "{:.4f}", "EV": "{:.2f}"}),
        use_container_width=True, hide_index=True
    )

    t_show = st.slider("Show eigen-based p(t) at t =", 0, int(horizon), 12)
    comp = p_t_linear_combo(p0, ed, t_show)
    st.write({k: round(v, 6) for k, v in comp.items()})

    # Plots
    st.pyplot(plot_prob_paths(df_path, "Discrete-time probabilities"))
    st.pyplot(plot_EV(df_path, "EV(t) from discrete-time"))

    # ---------------- Sensitivity analysis ----------------
    st.subheader("5) Sensitivity (finite differences)")
    st.latex(
        r"\frac{\partial p_i(t)}{\partial \theta},\;\; \frac{\partial EV(t)}{\partial \theta}"
        r"\quad \text{for }\theta\in\{\alpha,\gamma,\delta\}"
    )
    t_sens = st.slider("Sensitivity horizon t", 1, int(horizon), min(12, int(horizon)))
    sens = finite_diff_sensitivities(p0, alpha, gamma, delta, PV, t=t_sens, eps=1e-6)
    st.json(sens)
    st.pyplot(plot_sensitivity_bars(sens))

           # ---------------- 6) Continuous-time “PDE” (backward ODE) ----------------
    st.subheader("6) Continuous-time Backward ODE (PDE form)")

    # (opcional) algo de estilo KaTeX
    st.markdown(
        """
<style>
.katex-display { font-size: 1.15em; }
.katex .base   { line-height: 1.25; }
</style>
""",
        unsafe_allow_html=True
    )

    st.markdown("**Modelo en tiempo continuo (CTMC) y ecuación backward**")
    st.markdown(
        "Sea $X_t\\in\\{P,S1,S2\\}$ un proceso en tiempo continuo con **generador** $Q(t)$ "
        "que determina las tasas de salto entre estados. Definimos el vector de valores de estado:"
    )

    # v(t)
    st.latex(r"""
v(t)=
\begin{bmatrix}
EV_P(t)\\[2pt]
EV_{S1}(t)\\[2pt]
EV_{S2}(t)
\end{bmatrix},\qquad t\in[0,T].
""")

    st.markdown("**Generador (sin descuento ni flujo corriente)** para tasas $\\alpha(t),\\gamma(t),\\delta(t)$:")

    # Q(t)
    st.latex(r"""
Q(t)=
\begin{bmatrix}
-\alpha(t) & \alpha(t) & 0\\
\gamma(t)  & -\big(\gamma(t)+\delta(t)\big) & \delta(t)\\
0          & 0                               & 0
\end{bmatrix}.
""")

    st.markdown("**Ecuación backward (sin descuento):**")

    # dv/dt = -Qv, v(T)=EV_T
    st.latex(r"""
\frac{\mathrm{d}v}{\mathrm{d}t} \;=\; -\,Q(t)\,v(t),\qquad v(T)=EV_T.
""")

    st.markdown(
        "donde $EV_T=\\big[\\,EV_P(T),\\,EV_{S1}(T),\\,EV_{S2}(T)\\,\\big]^{\\top}$ es la **condición terminal**."
    )
    st.markdown(
        "Si la distribución inicial de estados es "
        "$p_0=[\\,p_P(0),\\,p_{S1}(0),\\,p_{S2}(0)\\,]$, el valor escalar es:"
    )

    # EV(t) = p0 v(t)
    st.latex(r"""EV(t)=p_0\,v(t).""")

    st.markdown("---")
    st.markdown("**Extensión con descuento y flujo corriente.** Con tasa de descuento $r(t)$ y flujo por estado $c(t)\in\mathbb{R}^3$:")

    # dv/dt with r(t) and c(t)
    st.latex(r"""
\frac{\mathrm{d}v}{\mathrm{d}t}
\;=\;
-\big(Q(t)-r(t)\,I\big)\,v(t)\;-\;c(t),\qquad v(T)=EV_T.
""")

    st.markdown("---")
    st.markdown("**Solución por factor integrante (tramo constante).** En un tramo $[t_k,t_{k+1}]$ donde "
                "$Q(t)\equiv Q_k$, $r(t)\equiv r_k$, $c(t)\equiv c_k$, defina:")

    # A_k, Delta t, Phi_k
    st.latex(r"""A_k:=Q_k-r_k I,\qquad \Delta t:=t_{k+1}-t_k,\qquad \Phi_k:=\exp\!\big(-\,\Delta t\,A_k\big).""")

    st.markdown("Entonces:")

    # v(t_k) propagation
    st.latex(r"""
v(t_k)\;=\;\Phi_k\,v(t_{k+1})\;+\;A_k^{-1}\,\big(I-\Phi_k\big)\,c_k.
""")

    st.markdown("(Para $c_k=\mathbf{0}$ y $r_k=0$: $v(t_k)=\exp(-\Delta t\,Q_k)\,v(t_{k+1})$.)")

    st.markdown("---")
    st.markdown("**Relación con tiempo discreto.** Si se fija un paso $\\Delta t$ y $Q$ es constante por tramo:")
    st.latex(r"""T_{\Delta t}\;=\;\exp\!\big(\Delta t\,Q\big),\qquad p_{t+\Delta t}\;=\;p_t\,T_{\Delta t}\,.""")

    # ====== Controles de la sección CT ======
    cT1, cT2, cT3 = st.columns(3)
    with cT1:
        T_hor_years = st.number_input("Terminal time T (years)",
                                      min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    with cT2:
        try:
            EV_T_vec = np.array(
                [float(x) for x in st.text_input(
                    "Terminal payoff EV_T (P,S1,S2)", "120,80,40").split(",")],
                dtype=float
            )
            if EV_T_vec.shape[0] != 3:
                raise ValueError
        except Exception:
            EV_T_vec = np.array([120.0, 80.0, 40.0], dtype=float)
    with cT3:
        ct_rates = st.selectbox("Rates in time", ["Constant", "Linear drift"], index=0)

    # Copias locales para evitar problemas de alcance
    alpha_val = float(alpha)
    gamma_val = float(gamma)
    delta_val = float(delta)

    # Definir funciones de tasas en el tiempo
    if ct_rates == "Constant":
        alpha_t = _as_callable(alpha_val)
        gamma_t = _as_callable(gamma_val)
        delta_t = _as_callable(delta_val)
    else:
        k_a = st.number_input("α drift per year", -1.0, 1.0, 0.00, 0.01)
        k_g = st.number_input("γ drift per year", -1.0, 1.0, 0.00, 0.01)
        k_d = st.number_input("δ drift per year", -1.0, 1.0, 0.00, 0.01)
        alpha_t = lambda t: float(np.clip(alpha_val + k_a * t, 0.0, 1.0))
        gamma_t = lambda t: float(np.clip(gamma_val + k_g * t, 0.0, 1.0))
        delta_t = lambda t: float(np.clip(delta_val + k_d * t, 0.0, 1.0))

    # Resolver ODE backward
    t_ct, v_ct = solve_backward_EV(T_hor_years, EV_T_vec, alpha_t, gamma_t, delta_t)
    EV_scalar_ct = (p0 @ v_ct.T).ravel()
    df_ct = pd.DataFrame({
        "t": t_ct,
        "EV_P": v_ct[:, 0],
        "EV_S1": v_ct[:, 1],
        "EV_S2": v_ct[:, 2],
        "EV_scalar": EV_scalar_ct
    })

    st.dataframe(
        df_ct.head(12).style.format(
            {c: "{:.4f}" for c in ["EV_P", "EV_S1", "EV_S2", "EV_scalar"]}
        ),
        use_container_width=True, hide_index=True
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    ax.plot(df_ct["t"], df_ct["EV_scalar"], label="EV_scalar(t) = p0·v(t)")
    ax.plot(df_ct["t"], df_ct["EV_P"], label="EV_P(t)", alpha=0.65)
    ax.plot(df_ct["t"], df_ct["EV_S1"], label="EV_S1(t)", alpha=0.65)
    ax.plot(df_ct["t"], df_ct["EV_S2"], label="EV_S2(t)", alpha=0.65)
    ax.set_xlabel("t (years)")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Backward ODE solution (continuous time)")
    st.pyplot(fig)



# ───────────────────── HMM (3 estados) — helper ─────────────────────
def hmm_three_state_pd(lam: float, mv: int, mp: int,
                       alpha: float | None = None,
                       gamma: float = 0.40,     # cura mensual (ejemplo)
                       delta: float = 0.60,     # default final mensual
                       beta:  float = 0.0,      # salto directo a S2 (normalmente 0)
                       force_resolution_at_t2: bool = True):
    """
    Devuelve PD1 y PD2_cond desde un HMM 3-estados:
    P (performing), S1 (distressed), S2 (default final, absorbente).
    - PD1 = P[ estado=S1 en t=mv ]  (si quieres incluir saltos directos ⇒ sumar S2)
    - PD2_cond = P[ S2 en t=mv+mp | estado era S1 en t=mv ]
    """
    lam = float(max(lam, 0.0))
    # α: P→S1; por defecto lo tomamos del hazard mensual de Poisson
    if alpha is None:
        alpha = 1.0 - np.exp(-lam / 12.0)

    if force_resolution_at_t2:
        # anula permanencia en S1: γ+δ=1
        gamma = float(gamma)
        delta = float(delta)
        total = max(gamma + delta, 1e-12)
        gamma, delta = gamma / total, delta / total
        s1_stay = 0.0
    else:
        # permite permanencia en S1
        s1_stay = max(0.0, 1.0 - (gamma + delta))

    # Matriz de transición mensual
    A = np.array([
        [max(0.0, 1.0 - alpha - beta), alpha, beta],
        [gamma, s1_stay, delta],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    # Distribuciones
    A_mv = np.linalg.matrix_power(A, int(mv))
    pi0 = np.array([1.0, 0.0, 0.0], dtype=float)
    pi_t1 = pi0 @ A_mv

    PD1_onlyS1 = float(pi_t1[1])
    PD1_any = float(pi_t1[1] + pi_t1[2])  # si quisieras contar saltos directos

    # Evoluciona mp meses desde S1 puro
    A_mp = np.linalg.matrix_power(A, int(mp))
    q_from_S1 = np.array([0.0, 1.0, 0.0]) @ A_mp

    if force_resolution_at_t2:
        PD2_cond = float(q_from_S1[2])
    else:
        denom = float(q_from_S1[0] + q_from_S1[2])
        PD2_cond = float(q_from_S1[2] / denom) if denom > 1e-12 else 0.0

    return {
        "A": A,
        "alpha": alpha, "gamma": gamma, "delta": delta, "beta": beta,
        "PD1": PD1_onlyS1, "PD1_any": PD1_any, "PD2_cond": PD2_cond
    }

# --- Ejemplo de ruteo (fuera de esta función en tu app principal) ---
# page = st.sidebar.radio("📌 Menú", [..., "PD Algorithm"], index=0)
# if page == "PD Algorithm":
#     render_pd_algorithm_tab()



# ─────────────────── Plotly Template Corporativo ────────────────────────
pio.templates["curay"] = dict(
    layout=dict(
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=13),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        colorway=["#0C1B2A", "#CBA135", "#5F6B7A", "#18B277", "#E05F5F"]
    )
)
pio.templates.default = "curay"

# ─────────────────── Tipografía matemática (estilo LaTeX) ───────────────
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.default"] = "rm"
plt.rcParams["font.size"] = 12

# ───────────────────────── 0) PAGE, VERSION & BRANDING ───────────────────
APP_VERSION = "6.5-AQ-Neuro"

# Brand tokens (edit in sidebar)
st.sidebar.header("🎨 Apariencia")
ACCENT = st.sidebar.color_picker("Color acento (oro)", value="#CBA135")
NAVY = st.sidebar.color_picker("Navy corporativo", value="#0C1B2A")
INK = "#202733"
SLATE = "#5F6B7A"
BG = "#0F1721"
LIGHT_BG = "#121B26"
COMPACT = st.sidebar.toggle("Modo compacto (menos espacios)", value=False)
HC       = st.sidebar.toggle("♿ Alto contraste", value=False)
PRESENT  = st.sidebar.toggle("🖥️ Modo presentación", value=False)
YC_MODE  = st.sidebar.toggle("🟠 YC demo mode", value=True)
# ───────────────────── Paletas (árbol & HMM) ─────────────────────
def _hex_to_rgba(hex_str: str, a: float = 1.0) -> str:
    h = hex_str.strip().lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

DEFAULT_TREE = {
    "pay1": "#18B277",   # t1: paga
    "cura": "#CBA135",   # t2: cura
    "def2": "#E05F5F",   # t2: default
    "pd1":  "#5F6B7A",   # raíz -> distress (PD1)
    "edge": "#0C1B2A",   # color neutro por si se necesita
    "label": "#0C1B2A",
    "time_label": "#0f172a",              # dark navy text for T=0/1/2
    "time_grid":  "rgba(15,23,42,0.18)",  # subtle vertical grid lines
    "time_band":  "rgba(15,23,42,0.04)",  # optional faint column bands
}
DEFAULT_HMM = {
    "alpha": "#CBA135",  # P->S1
    "gamma": "#18B277",  # S1->P
    "delta": "#E05F5F",  # S1->S2
    "beta":  "#64748B",  # P->S2 (si usas beta)
    "stay":  "#0C1B2A",  # loops
    "node_P":  "#18B277",
    "node_S1": "#CBA135",
    "node_S2": "#E05F5F",
    "node_edge": "#0C1B2A",
}

st.session_state.setdefault("pal_tree", DEFAULT_TREE.copy())
st.session_state.setdefault("pal_hmm",  DEFAULT_HMM.copy())

with st.sidebar.expander("🎨 Colores (árbol & HMM)", expanded=False):
    st.caption("Ajusta los colores y de paso sirve de glosario visual.")

    c1, c2, c3 = st.columns(3)

    # Árbol (neural)
    with c1:
        st.markdown("**Árbol (neural)**")
        t_pay1 = st.color_picker("t₁: paga",   value=st.session_state["pal_tree"]["pay1"])
        t_cura = st.color_picker("t₂: cura",   value=st.session_state["pal_tree"]["cura"])
        t_def2 = st.color_picker("t₂: default",value=st.session_state["pal_tree"]["def2"])
        t_pd1  = st.color_picker("PD₁ (raíz→distress)", value=st.session_state["pal_tree"]["pd1"])

    # HMM (aristas)
    with c2:
        st.markdown("**HMM: aristas**")
        h_alpha = st.color_picker("α: P→S1", value=st.session_state["pal_hmm"]["alpha"])
        h_gamma = st.color_picker("γ: S1→P", value=st.session_state["pal_hmm"]["gamma"])
        h_delta = st.color_picker("δ: S1→S2", value=st.session_state["pal_hmm"]["delta"])
        h_beta  = st.color_picker("β: P→S2", value=st.session_state["pal_hmm"]["beta"])

    # HMM (nodos/loops)
    with c3:
        st.markdown("**HMM: nodos/loops**")
        nP  = st.color_picker("Nodo P",  value=st.session_state["pal_hmm"]["node_P"])
        nS1 = st.color_picker("Nodo S1", value=st.session_state["pal_hmm"]["node_S1"])
        nS2 = st.color_picker("Nodo S2", value=st.session_state["pal_hmm"]["node_S2"])
        h_stay = st.color_picker("Loop (stay)", value=st.session_state["pal_hmm"]["stay"])

    # Guardar en estado
    st.session_state["pal_tree"] = {
        "pay1": t_pay1, "cura": t_cura, "def2": t_def2, "pd1": t_pd1,
        "edge": DEFAULT_TREE["edge"], "label": DEFAULT_TREE["label"],
    }
    st.session_state["pal_hmm"] = {
        "alpha": h_alpha, "gamma": h_gamma, "delta": h_delta, "beta": h_beta, "stay": h_stay,
        "node_P": nP, "node_S1": nS1, "node_S2": nS2, "node_edge": DEFAULT_HMM["node_edge"],
    }

    if st.button("↺ Reset paletas", use_container_width=True):
        st.session_state["pal_tree"] = DEFAULT_TREE.copy()
        st.session_state["pal_hmm"]  = DEFAULT_HMM.copy()

    # Mini-leyenda visual
    st.markdown(f"""
    <div class="card" style="margin-top:8px;padding:10px;display:flex;gap:10px;flex-wrap:wrap">
      <span class="badge-help"><span class="dot" style="background:{t_pay1}"></span> t₁: paga</span>
      <span class="badge-help"><span class="dot" style="background:{t_cura}"></span> t₂: cura</span>
      <span class="badge-help"><span class="dot" style="background:{t_def2}"></span> t₂: default</span>
      <span class="badge-help"><span class="dot" style="background:{h_alpha}"></span> α P→S1</span>
      <span class="badge-help"><span class="dot" style="background:{h_gamma}"></span> γ S1→P</span>
      <span class="badge-help"><span class="dot" style="background:{h_delta}"></span> δ S1→S2</span>
    </div>
    """, unsafe_allow_html=True)


def apply_yc_defaults():
    st.session_state.setdefault("score", 3.40)
    st.session_state.setdefault("EAD", 1_000_000.0)
    st.session_state.setdefault("tc_ann", 0.25)
    st.session_state.setdefault("mv", 12)
    st.session_state.setdefault("mp", 3)
    st.session_state.setdefault("garantia_val0", 600_000.0)
    st.session_state.setdefault("gastos_usd", 0.0)

if YC_MODE:
    np.random.default_rng(42)  # Monte Carlo determinístico
    apply_yc_defaults()

# Política: CO fijo anual (1.50%) y tm anual fija (0.75%)
CO_ANUAL_FIJO = 0.015
TM_ANUAL_FIJO = 0.0075  # 0.75% anual fijo
HERO_SUBTITLE = (
    f"Score → PD · LGD • Regla del fondo (CO 1.50% fijo) • tm fija {TM_ANUAL_FIJO*100:.2f}% • Transparencia total"
)


# ───────────────────────── CSS (limpio y consistente) ────────────────────
css = f"""
<style>
:root {{
  --navy:{NAVY}; --ink:{INK}; --slate:{SLATE}; --gold:{ACCENT};
  --bg:{BG}; --light-bg:{LIGHT_BG}; --card-bg:#0f1721;
  --radius: 18px;
  --gap:{'10px' if COMPACT else '18px'};
  --pad:{'16px' if COMPACT else '22px'};
  --pad-lg:{'20px' if COMPACT else '28px'};
}}
.stApp {{
  background:
    radial-gradient(90rem 90rem at -30% -30%, #1b2840 0%, transparent 35%),
    radial-gradient(80rem 80rem at 130% -30%, #2d1b40 0%, transparent 35%),
    var(--bg);
  color: #e7edf6; font-weight: 450;
}}
* {{ scroll-behavior: smooth; }}
:focus-visible {{ outline: 2px solid var(--gold); outline-offset: 2px; }}

[data-testid="stSidebar"] > div:first-child {{
  background: linear-gradient(180deg, #0e1725, #0a1320);
  border-right: 1px solid rgba(255,255,255,0.06);
}}
.hero {{
  padding: calc(var(--pad-lg) + 6px) var(--pad-lg);
  border-radius: 24px;
  background:
    linear-gradient(135deg, rgba(203,161,53,.12), rgba(203,161,53,0) 45%),
    linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 30px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,0.03);
}}
.hero h1 {{ margin: 0 0 6px; letter-spacing:.2px; font-size: 1.65rem; color:#f5f7fb; }}
.hero p {{ margin:0; color:#b9c4d3; font-size:.98rem; }}
.badge {{
  display:inline-block; padding:4px 10px; font-size:.78rem;
  border-radius:999px; background: rgba(203,161,53,.16); color:#f2e6c9;
  border:1px solid rgba(203,161,53,.35);
}}
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.08);
  border-radius: var(--radius);
  padding: var(--pad);
  box-shadow: 0 8px 20px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,0.02);
  backdrop-filter: saturate(120%) blur(4px);
}}
.kpi h4 {{ margin:0; color:#a9b7c7; font-weight:500; font-size:.85rem; letter-spacing:.3px }}
.kpi .val {{ font-size:1.65rem; margin-top:6px; color:#f6f8fb; font-weight:680 }}
.kpi .foot {{ margin-top:3px; color:#8fa0b3; font-size:.82rem }}
.rr-wrap {{ margin-top:6px; }}
.bar {{ position:relative; height:12px; border-radius:999px; background: rgba(255,255,255,.08);
        overflow:hidden; border:1px solid rgba(255,255,255,.1); }}
.bar .fill {{ position:absolute; inset:0; width:0%;
  background: linear-gradient(90deg, rgba(24,178,119,.9), rgba(203,161,53,.9) 60%, rgba(224,95,95,.95));
  transition: width .25s ease; }}
.bar .tick {{ position:absolute; top:-4px; width:2px; height:20px; background:var(--gold);
  box-shadow:0 0 0 2px rgba(203,161,53,.2); }}
.ribbon {{display:flex; gap:10px; align-items:center; margin:10px 0 2px}}
.pill {{padding:6px 12px; border-radius:999px; font-weight:700; letter-spacing:.3px}}
.pill.ok {{background:rgba(24,178,119,.18); border:1px solid rgba(24,178,119,.45); color:#d6fff0}}
.pill.no {{background:rgba(224,95,95,.18); border:1px solid rgba(224,95,95,.45); color:#ffdddd}}
.small {{font-size:.82rem; color:#a8b4c4}}
.footer {{ color:#9fb1c7; font-size:.85rem; margin-top:10px }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"] {{ font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
.kpi .val, .small, .badge, .pill, .hero p, .hero h1 {{ font-variant-numeric: tabular-nums; }}
.card{{transition:transform .18s ease, box-shadow .18s ease}}
.card:hover{{transform:translateY(-1px); box-shadow:0 14px 30px rgba(0,0,0,.40)}}
:root {{ --bg:{'#0B0F16' if HC else BG}; }}
[data-testid="stSidebar"] {{ display: {'none' if PRESENT else 'block'}; }}
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown(
    f"""
    <div class="hero">
      <span class="badge">Modelo Aquila · IFRS-9 Ready · v{APP_VERSION}</span>
      <h1>Arquitectura Binaria de Riesgo de Crédito — Probabilidad de Default</h1>
      <p>{HERO_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
# 👉 Pon esto inmediatamente después de tu bloque Hero (st.markdown(...) del hero)
sp1, sp2, spot = st.columns([1, 1, 1])  # el último queda a la derecha
with spot:
    with st.popover("📚 Glosario rápido"):
        st.markdown("""
**PD(12m):** prob. de default en 12 meses.  
**LGD:** pérdida en caso de default (1 − recuperación).  
**RE (simple):** `tc·(1−PD12) − LGD·PD12`.  
**EV (VP):** valor esperado descontado del árbol t₀→t₁→t₂.  
**mv/mp:** meses a t₁ / meses t₁→t₂.  
**HMM:** 3 estados {P,S1,S2} para refinar timing de default/cura.
""")

st.markdown(f"""
<style>
/* —— Microinteracciones y glow premium —— */
.card{{
  position:relative;
  transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  box-shadow: 0 8px 28px rgba(0,0,0,.35), 0 2px 0 rgba(0,0,0,.08) inset;
}}
.card:hover{{transform:translateY(-1px); box-shadow:0 16px 40px rgba(0,0,0,.45)}}
.card:active{{transform:translateY(0); box-shadow:0 8px 24px rgba(0,0,0,.35)}}

/* Botones con halo suave en hover */
.stButton>button{{
  border-radius:14px !important; font-weight:700;
  box-shadow: 0 6px 18px rgba(0,0,0,.25);
  transition: box-shadow .18s ease, transform .12s ease;
}}
.stButton>button:hover{{ box-shadow:0 12px 28px rgba(0,0,0,.38)}}
.stButton>button:active{{ transform:translateY(1px) }}

/* Badges de ayuda */
.badge-help{{
  display:inline-flex; gap:6px; align-items:center;
  background: rgba(255,255,255,.08); color:#dfe7f1; border:1px solid rgba(255,255,255,.16);
  border-radius:999px; padding:6px 10px; font-size:.82rem;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.06), 0 4px 14px rgba(0,0,0,.28);
}}
.badge-help .dot{{width:8px;height:8px;border-radius:999px;background: {ACCENT}; box-shadow:0 0 12px {ACCENT};}}


/* Tooltip minimal */
.tooltip-wrap{{ position:relative; display:inline-block; }}
.tooltip-wrap .tip{{
  visibility:hidden; opacity:0; transition:opacity .18s ease;
  position:absolute; z-index:10; bottom:130%; left:50%; transform:translateX(-50%);
  background:#0b1020; color:#e6eef9; border:1px solid rgba(255,255,255,.1);
  padding:10px 12px; border-radius:10px; min-width:220px; text-align:left;
  box-shadow:0 14px 32px rgba(0,0,0,.45);
}}
.tooltip-wrap:hover .tip{{ visibility:visible; opacity:1; }}
.tooltip-wrap .tip:after{{
  content:""; position:absolute; top:100%; left:50%; transform:translateX(-50%);
  border-width:8px; border-style:solid; border-color:#0b1020 transparent transparent transparent;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.accept-card{display:flex;flex-direction:column;gap:10px}
.accept-grid{display:grid;grid-template-columns:1fr auto auto;gap:10px;align-items:center}
.accept-k{color:#a9b7c7;font-size:.88rem}
.accept-v{color:#f6f8fb;font-weight:700}
.icon-cell{width:34px;height:34px;display:grid;place-items:center}
.pill-final{padding:8px 14px;border-radius:999px;font-weight:800;letter-spacing:.3px;display:inline-flex;gap:8px;align-items:center}
.pill-final.ok{background:rgba(24,178,119,.18);border:1px solid rgba(24,178,119,.45);color:#d6fff0}
.pill-final.no{background:rgba(224,95,95,.18);border:1px solid rgba(224,95,95,.45);color:#ffdddd}

.chk svg{width:28px;height:28px}
.chk .circle{stroke-dasharray:166;stroke-dashoffset:166;animation:dash .7s ease forwards}
.chk .tick{stroke-dasharray:48;stroke-dashoffset:48;animation:dash .5s .2s ease forwards}
.x svg{width:28px;height:28px}
.x .line{stroke-dasharray:38;stroke-dashoffset:38;animation:dash .45s ease forwards}
.spin{width:22px;height:22px;border:3px solid rgba(255,255,255,.25);border-top-color:var(--gold);border-radius:50%;animation:spin .8s linear infinite}
@keyframes dash{to{stroke-dashoffset:0}}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Oculta la píldora de la ribbon (ACEPTAR/NO ACEPTAR) pero deja el RR y el Run ID */
.ribbon .pill{ display:none !important; }
</style>
""", unsafe_allow_html=True)

# ────────────────────── 1) PARÁMETROS GLOBALES ──────────────────────────
PD12_ANCLA_1, PD12_ANCLA_5 = 0.80, 0.05
THRESHOLD_RR_DEFAULT = 0.22
ACTUALIZACION_ECL = "Semestral (IFRS 9)"
EPS = 1e-12

CAT_GARANTIAS = {
    "Letra/Pagaré + Aval (alto patrimonio)": {},
    "Letra/Pagaré +/- Aval (medio patrimonio)": {},
    "Letra/Pagaré +/- Aval (bajo patrimonio)": {},
    "Cheque representante (alto patrimonio)": {},
    "Cheque representante (medio patrimonio)": {},
    "Cheque representante (bajo patrimonio)": {},
    "Hipoteca predio rústico": {},
    "Hipoteca predio urbano": {},
    "Fideicomiso predio rústico (realización)": {},
    "Fideicomiso predio urbano (realización)": {},
    "Fideicomiso planta de packing": {},
    "Flujos cedidos a fideicomiso": {},
    "Activo mobiliario / Warrant": {},
    "Activo mobiliario en fideicomiso": {},
}
ALIAS_GARANTIAS = {
    "letra/pagaré con aval personal - alto patrimonio": "Letra/Pagaré + Aval (alto patrimonio)",
    "letra/pagaré sin/con aval personal - medio patrimonio": "Letra/Pagaré +/- Aval (medio patrimonio)",
    "letra/pagaré sin/con aval personal - bajo patrimonio": "Letra/Pagaré +/- Aval (bajo patrimonio)",
    "cheque de representante con alto patrimonio": "Cheque representante (alto patrimonio)",
    "cheque de representante con medio patrimonio": "Cheque representante (medio patrimonio)",
    "cheque de representante con bajo patrimonio": "Cheque representante (bajo patrimonio)",
    "inmueble en hipoteca - predio rústico/rural": "Hipoteca predio rústico",
    "inmueble en hipoteca - predio urbano": "Hipoteca predio urbano",
    "garantía real en fideicomiso - valor de realización - predio rústico/rural": "Fideicomiso predio rústico (realización)",
    "garantía real en fideicomiso - valor de realización - predio urbano": "Fideicomiso predio urbano (realización)",
    "garantía real en fideicomiso - planta de packing": "Fideicomiso planta de packing",
    "flujos cedidos a fideicomiso": "Flujos cedidos a fideicomiso",
    "activo mobiliario - valor de realización / warrant": "Activo mobiliario / Warrant",
    "activo mobiliario cedido en fideicomiso": "Activo mobiliario en fideicomiso",
}

# ───────────────────── 2) UTILIDADES ────────────────────────────────────
def help_badge(text:str, tip:str):
    st.markdown(
        f"""
        <span class="tooltip-wrap">
          <span class="badge-help"><span class="dot"></span>{text}</span>
          <span class="tip">{tip}</span>
        </span>
        """,
        unsafe_allow_html=True
    )

def _add_glow_line(fig, x, y, base_color, base_width=6, layers=3):
    for i in range(layers, 0, -1):
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(width=base_width + i*4, color=f"rgba(203,161,53,{0.08*i})", shape="spline"),
            hoverinfo="skip", showlegend=False
        ))

def _add_glow_node(fig, x, y, layers=3):
    for i in range(layers, 0, -1):
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=30 + i*8, color=f"rgba(203,161,53,{0.06*i})", line=dict(width=0)),
            hoverinfo="skip", showlegend=False
        ))

def clamp(x, lo, hi): return max(lo, min(hi, x))

def fmt_pct(x, p=2):
    try:
        return "—" if x is None or not np.isfinite(x) else f"{x*100:.{p}f}%"
    except Exception:
        return "—"

def fmt_usd(x, p=0):
    try:
        return "—" if x is None or not np.isfinite(x) else f"${x:,.{p}f}"
    except Exception:
        return "—"

import re  # <-- missing import

def _parse_pesos(raw) -> list[float]:
    """
    Parse weights like '80%, 10%, 10' or [80,10,10] into decimals [0.8, 0.1, 0.1].
    Returns [] when input is empty/NaN.
    """
    # empty / NaN guards
    if raw is None:
        return []
    try:
        # handles pandas/numpy NaN without importing pandas explicitly
        if raw != raw:  # NaN check
            return []
    except Exception:
        pass

    # numeric single value
    if isinstance(raw, (int, float)):
        v = float(raw)
        return [max(0.0, v/100.0 if v > 1.0 else v)]

    # string-like list
    s = str(raw).strip().strip("[](){}")
    if not s:
        return []

    tokens = re.split(r"[\s,;:/|+&]+", s)
    out: list[float] = []
    for tok in tokens:
        tok = tok.replace("%", "").replace(",", ".")
        try:
            v = float(tok)
        except ValueError:
            continue
        out.append(max(0.0, v/100.0 if v > 1.0 else v))
    return out


def parse_peso_total(raw) -> float:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)): return 1.0
    try:
        token = re.split(r"[\s,;:/|+&]+", str(raw).strip().strip("[](){}"))[0]
        v = float(token.replace("%","").replace(",", "."))
    except:
        return 1.0
    return clamp(v/100.0 if v>1.0 else v, 0.0, 1.0)

def normalizar_tipo_garantia(s: str) -> str:
    if not isinstance(s, str): return ""
    key = s.strip().lower()
    if key in ALIAS_GARANTIAS: return ALIAS_GARANTIAS[key]
    for k in CAT_GARANTIAS.keys():
        if key == k.lower(): return k
    return ""

def construir_garantias_desde_CARTERA(row: pd.Series, garantia_val: float) -> pd.DataFrame:
    cols_out = ["Tipo","Valor USD"]
    if "TIPO DE GARANTIA" in row.index:
        tipos_raw = "" if pd.isna(row.get("TIPO DE GARANTIA")) else str(row.get("TIPO DE GARANTIA"))
    else:
        try: tipos_raw = str(row.iloc[11])
        except: tipos_raw = ""
    tipos = [t.strip() for t in re.split(r"[;,/|+&]+", tipos_raw) if t.strip()]
    canon = [normalizar_tipo_garantia(t) for t in tipos]
    canon = [c for c in canon if c]
    if not canon:
        return pd.DataFrame([{"Tipo": tipos_raw if tipos_raw else "Garantía", "Valor USD": float(garantia_val)}], columns=cols_out)

    if   "Peso de la Garantía" in row.index: pesos_raw = row.get("Peso de la Garantía")
    elif "Peso de la Garantia" in row.index: pesos_raw = row.get("Peso de la Garantia")
    else:
        try: pesos_raw = row.iloc[12]
        except: pesos_raw = None
    pesos = _parse_pesos(pesos_raw)
    n = len(canon)
    if not pesos: weights = [1.0/n]*n
    else:
        if len(pesos)<n: pesos += [0.0]*(n-len(pesos))
        elif len(pesos)>n: pesos = pesos[:n]
        s = sum(pesos);  weights = ([1.0/n]*n) if s<=0 else [p/s for p in pesos]
    return pd.DataFrame([{"Tipo":c, "Valor USD": float(garantia_val)*w} for c,w in zip(canon,weights)], columns=cols_out)

def valor_maximo_garantias(df_g: pd.DataFrame, EAD: float) -> float:
    EAD = max(float(EAD), 0.0)
    if EAD <= 0 or df_g is None or df_g.empty: return 0.0
    valores = np.maximum(pd.to_numeric(df_g["Valor USD"], errors="coerce").fillna(0.0).values, 0.0)
    return min(float(np.sum(valores)), EAD)

def lgd_politica(df_g: pd.DataFrame, EAD: float) -> float:
    EAD = max(float(EAD), 0.0)
    if EAD <= 0: return 0.0
    Vmax = valor_maximo_garantias(df_g, EAD)
    return clamp(1.0 - (Vmax/EAD), 0.0, 1.0)

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
    ln_lam = ln1 + alpha * (ln5 - ln1)
    return float(np.exp(ln_lam))

def pd_hazard_months(lam: float, m: int) -> float:
    m = int(max(0, m)); t_years = m / 12.0
    return 1.0 - np.exp(-max(lam, 0.0) * t_years)

def to_monthly(r_ann: float) -> float:
    r = np.clip(float(r_ann), -0.99, 10.0)
    return (1.0 + r) ** (1.0/12.0) - 1.0
def to_annual_from_monthly(r_m: float) -> float:
    """Convierte tasa mensual a anual efectiva: (1+r_m)^12 - 1"""
    r = np.clip(float(r_m), -0.99, 10.0)  # límites defensivos
    return (1.0 + r) ** 12 - 1.0

def annualize_from_monthly(r_m: float, periods: int = 12) -> float:
    """Convierte una tasa periódica (mensual) a efectiva anual."""
    r_m = float(r_m)
    return (1.0 + r_m) ** periods - 1.0

def ensure_tc_ann(tc_value: float, base: str = "mensual") -> float:
    """
    Devuelve tc en anual efectivo.
    base: 'mensual' si el input es mensual (p.ej. 0.025), 'anual' si ya es anual.
    """
    base = (base or "").lower()
    if base.startswith("men"):
        return annualize_from_monthly(tc_value, 12)
    return float(tc_value)


def peso_garantia_col_M(row: pd.Series) -> float:
    try:
        if "Peso de la Garantía" in row.index:
            raw = row["Peso de la Garantía"]
        elif "Peso de la Garantia" in row.index:
            raw = row["Peso de la Garantia"]
        else:
            raw = row.iloc[12]  # COLUMNA M
    except Exception:
        raw = None
    return parse_peso_total(raw)


# ---- secrets helper (env-first; only touch st.secrets if present)
import os
def _get_secret(name: str, default=None):
    v = os.environ.get(name)
    if v is not None:
        return v
    try:
        import streamlit as st
        return st.secrets[name]  # only parsed if a secrets file exists
    except Exception:
        return default


# ───────────────────── 2.1) CORE CÁLCULO (POLÍTICA + tm fija) ───────────
def calcular_resultados_curay(
    score: float, EAD: float,
    tc_ann: float, co_ann: float,
    mv: int, mp: int,
    df_g: pd.DataFrame,
    gastos_usd: float,
    w_guarantee: float = 1.0,
    pd12_anchor_1=PD12_ANCLA_1, pd12_anchor_5=PD12_ANCLA_5,
    hmm: dict | None = None
):
    """
    Cálculo principal con opción de HMM para sustituir PD1 y PD2_cond.
    Requiere que existan: lambda_anchors, lambda_from_score, pd_hazard_months,
    valor_maximo_garantias, lgd_politica, clamp, to_monthly, TM_ANUAL_FIJO,
    y (si se usa HMM) hmm_three_state_pd().
    """
    # --- PD / hazard (base Poisson) ---
    lam1, lam5 = lambda_anchors(pd12_anchor_1, pd12_anchor_5)
    lam = lambda_from_score(score, lam1, lam5)

    PD_12m   = 1.0 - np.exp(-lam)
    PD1      = pd_hazard_months(lam, mv)
    PD2_cond = pd_hazard_months(lam, mp)

    # --- HMM opcional: sustituye PD1 y PD2_cond si está activo ---
    hmm_meta = None
    if hmm and hmm.get("use", False):
        hmm_meta = hmm_three_state_pd(
            lam=lam, mv=mv, mp=mp,
            alpha=hmm.get("alpha"),
            gamma=hmm.get("gamma", 0.40),
            delta=hmm.get("delta", 0.60),
            beta=hmm.get("beta", 0.0),
            force_resolution_at_t2=hmm.get("force_resolution_at_t2", True)
        )
        PD1 = float(hmm_meta["PD1"])             # sustituye PD₁
        PD2_cond = float(hmm_meta["PD2_cond"])   # sustituye PD₂|₁

    # --- Garantías y LGD (definir ANTES de usar) ---
    Vmax    = valor_maximo_garantias(df_g, EAD)
    LGD     = lgd_politica(df_g, EAD)
    G_total = float(pd.to_numeric(df_g["Valor USD"], errors="coerce").fillna(0.0).sum())
    w_G     = clamp(float(w_guarantee), 0.0, 1.0)

    # --- Recovery Rate (decimal) ---
    RECOVERY_RATE = max(0.0, 1.0 - float(LGD))  # = 1 − LGD

    # --- Tasas ---
    tc_m = to_monthly(tc_ann)
    co_m = to_monthly(co_ann)
    tm_m = to_monthly(TM_ANUAL_FIJO)            # fija por política

    # --- Flujos ---
    CF_t1      = float(EAD) * (1 + tc_m) ** int(mv)
    CF_t2_cura = float(EAD) * (1 + tc_m) ** (int(mv) + int(mp)) * (1 + tm_m) ** int(mp)

    # Default definitivo: SOLO w_G * G (sin gastos ni descuento)
    PV_t2_def  = w_G * G_total

    # VP (t1 y cura sí quedan a VP)
    PV_t1      = CF_t1 / ((1 + co_m) ** int(mv))
    PV_t2_cura = CF_t2_cura / ((1 + co_m) ** (int(mv) + int(mp)))

    # --- EV (valor presente) ---
    EV_VP = (1 - PD1) * PV_t1 + PD1 * ((1 - PD2_cond) * PV_t2_cura + PD2_cond * PV_t2_def)

    Texp        = float(mv) + float(PD1) * float(mp)
    multiplo_vp = EV_VP / float(EAD) if EAD > 0 else np.nan
    ret_ann_vp  = (
        multiplo_vp ** (12.0 / Texp) - 1.0
        if (EAD > 0 and Texp > 0 and np.isfinite(multiplo_vp) and multiplo_vp > 0)
        else np.nan
    )

    # --- Métricas de riesgo (con LGD) ---
    ECL             = PD_12m * LGD * float(EAD)
    gastos_rel      = (float(gastos_usd) / float(EAD)) if EAD > 0 else 0.0
    RE_anual_simple = tc_ann * (1 - PD_12m) - LGD * PD_12m

    out = {
        "lam": lam, "lam1": lam1, "lam5": lam5,
        "PD_12m": PD_12m, "PD1": PD1, "PD2_cond": PD2_cond,
        "LGD": LGD, "Vmax": Vmax, "ECL": ECL,
        "PV_t1": PV_t1, "PV_t2_cura": PV_t2_cura, "PV_t2_def": PV_t2_def,
        "EV_VP": EV_VP, "multiplo_vp": multiplo_vp, "ret_ann_vp": ret_ann_vp, "Texp": Texp,
        "tc_ann": tc_ann, "co_ann": co_ann, "tc_m": tc_m, "tm_m": tm_m,
        "gastos_usd": gastos_usd, "gastos_rel": gastos_rel,
        "RE_anual_simple": RE_anual_simple,
        # dejamos el factor (sin EAD) para no romper la UI previa
        "w_G": w_G, "G_total": G_total,
        "RECOVERY_RATE": RECOVERY_RATE,
    }
    if hmm_meta is not None:
        out["HMM"] = hmm_meta
    return out

# ================== RL / Policy Lab Helpers ==================
from dataclasses import dataclass
import itertools

# ---- Action grid (pricing, tenor, collateral ask) ----
@dataclass
class Action:
    tc_ann: float   # compensatoria anual
    mv: int         # meses hasta t1
    mp: int         # meses t1->t2
    w_G: float      # peso operativo de garantías (0..1)

def build_action_grid(tc_list, mv_list, mp_list, wG_list):
    grid = []
    for t, mv, mp, w in itertools.product(tc_list, mv_list, mp_list, wG_list):
        grid.append(Action(float(t), int(mv), int(mp), float(w)))
    return grid

# ---- Context featurizer (tabular; feel free to add features) ----
def context_features(score: float, EAD: float, garantia_val: float, co_ann: float) -> np.ndarray:
    # Basic, stable features for LinUCB; expand as you wish
    return np.array([
        1.0,
        float(score),
        np.log(max(EAD, 1.0)),
        np.log(max(garantia_val, 1.0)),
        float(co_ann),
        float(score**2),
    ], dtype=float)

# ---- Reward from your engine (choose metric) ----
def evaluate_action(context: dict, act: Action, reward_metric: str = "ret_ann_vp") -> tuple[float, dict]:
    """
    reward_metric: "ret_ann_vp" (annualized VP return) or "ev_mult" (EV/EAD - 1)
    """
    # Pull current state & guarantees
    score = float(context.get("score", 3.0))
    EAD = float(context.get("EAD", 1_000_000.0))
    co_ann = float(context.get("co_ann", CO_ANUAL_FIJO))
    gastos = float(context.get("gastos_usd", 0.0))

    df_g = context.get("df_g")
    if df_g is None or df_g.empty:
        df_g = pd.DataFrame([{"Tipo":"Garantía", "Valor USD": float(context.get("garantia_val0", 0.0))}])

    res = calcular_resultados_curay(
        score=score, EAD=EAD,
        tc_ann=act.tc_ann, co_ann=co_ann,
        mv=act.mv, mp=act.mp,
        df_g=df_g, gastos_usd=gastos,
        w_guarantee=act.w_G,
        pd12_anchor_1=st.session_state.get("pd12_anchor_1", PD12_ANCLA_1),
        pd12_anchor_5=st.session_state.get("pd12_anchor_5", PD12_ANCLA_5),
        hmm=None  # RL runs on your baseline engine; you can switch to HMM if desired
    )

    ev_mult = (res["EV_VP"] / EAD) - 1.0 if EAD > 0 else -1.0
    r = float(res.get("ret_ann_vp")) if reward_metric == "ret_ann_vp" else float(ev_mult)
    return r, res

# ---- LinUCB (per-action contextual bandit) ----
class LinUCB:
    """
    Independent LinUCB heads per action (A_a, b_a).
    Reward must be bounded (we clip), contexts are small vectors.
    """
    def __init__(self, n_actions: int, d: int, alpha: float = 1.0, reward_clip: float = 1.0):
        self.nA = n_actions; self.d = d; self.alpha = float(alpha)
        self.A = [np.eye(d, dtype=float) for _ in range(n_actions)]
        self.b = [np.zeros((d, 1), dtype=float) for _ in range(n_actions)]
        self.reward_clip = float(reward_clip)

    def _theta(self, a: int) -> np.ndarray:
        return np.linalg.solve(self.A[a], self.b[a])  # (d,1)

    def select(self, x: np.ndarray) -> int:
        x = x.reshape(-1, 1)  # (d,1)
        ucb = np.empty(self.nA, dtype=float)
        for a in range(self.nA):
            theta = self._theta(a)
            mu = float(theta.T @ x)
            s = float(np.sqrt(x.T @ np.linalg.solve(self.A[a], x)))
            ucb[a] = mu + self.alpha * s
        return int(np.argmax(ucb))

    def update(self, a: int, x: np.ndarray, r: float):
        x = x.reshape(-1, 1)
        r = float(np.clip(r, -self.reward_clip, self.reward_clip))  # keep numerics sane
        self.A[a] += x @ x.T
        self.b[a] += r * x

# ───────────────────── 2.2) VISUALIZACIÓN — Helpers ─────────────────────
def kpi_card(title:str, value:str, foot:str=""):
    st.markdown(
        f"""
        <div class="card kpi">
          <h4>{title}</h4>
          <div class="val">{value}</div>
          <div class="foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================== PANEL ==================
def acceptance_panel(res: dict, ret_threshold: float = 0.12):
    """
    Render a single acceptance panel based on RE (anual, simple).
    This should be called exactly ONCE by the page/controller, not from the roulette.
    """
    ret_val = float(res.get("RE_anual_simple", float("nan")))
    ret_ok  = np.isfinite(ret_val) and (ret_val >= ret_threshold)

    def _icon(ok: bool):
        color = "#18B277" if ok else "#E05F5F"
        svg_ok = """
        <div class="chk">
          <svg viewBox="0 0 52 52" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round">
            <circle class="circle" cx="26" cy="26" r="22"></circle>
            <path class="tick" d="M14 27 L23 36 L38 18"></path>
          </svg>
        </div>"""
        svg_x = """
        <div class="x">
          <svg viewBox="0 0 52 52" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round">
            <line class="line" x1="16" y1="16" x2="36" y2="36"></line>
            <line class="line" x1="36" y1="16" x2="16" y2="36"></line>
          </svg>
        </div>"""
        return f'<div class="icon-cell" style="color:{color}">{svg_ok if ok else svg_x}</div>'

    final_ok = ret_ok
    body = f"""
    <div class="card accept-card">
      <div class="accept-grid">
        <div class="accept-k">RE (anual, simple) ≥ {ret_threshold*100:.0f}%</div>
        <div class="accept-v">{ret_val*100:.2f}%</div>
        {_icon(ret_ok)}
      </div>
      <div>
        <span class="pill-final {'ok' if final_ok else 'no'}">
          {'✅ ACEPTAR' if final_ok else '⛔ NO ACEPTAR'}
          <span class="spin" style="margin-left:6px"></span>
        </span>
      </div>
    </div>
    """
    st.markdown(body, unsafe_allow_html=True)


def roulette_acceptance(res: dict, ret_threshold: float = 0.12, spins: int = 3):
    """
    Show the roulette animation/verdict *without* rendering the acceptance panel.
    Returns a boolean with the final decision so the caller can render acceptance_panel() once.
    """
    # --- Recuperar valor de RE (anual, simple) ---
    ret_val = float(res.get("RE_anual_simple", float("nan")))
    ret_ok  = np.isfinite(ret_val) and (ret_val >= ret_threshold)

    # --- Decisión final = solo RE (si en el futuro agregas reglas, ajusta esta variable) ---
    final_ok = ret_ok

    # --- Labels y colores ---
    labels = [f"RE (simple) ≥ {ret_threshold*100:.0f}%", "Veredicto"]
    values = [1, 1]
    col_ok, col_no = "#18B277", "#E05F5F"
    colors = [col_ok if ret_ok else col_no,
              col_ok if final_ok else col_no]

    # --- Figura tipo ruleta ---
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        textinfo="label", textfont_size=13,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        rotation=0, pull=[0, 0.08]
    ))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        annotations=[dict(
            text=("✅ <b>ACEPTAR</b>" if final_ok else "⛔ <b>NO ACEPTAR</b>"),
            x=0.5, y=0.5,
            font=dict(size=20, color=(col_ok if final_ok else col_no)),
            showarrow=False
        )]
    )

    # Render animado en un placeholder (omitimos render inicial para evitar parpadeo duplicado)
    placeholder = st.empty()
    run_key = st.session_state.get("accept_run_id") or secrets.token_hex(4)
    total_deg = 360 * max(2, min(6, int(spins))) + np.random.randint(0, 360)
    steps = 28

    for i, ang in enumerate(np.linspace(0, total_deg, steps)):
        fig.update_traces(rotation=float(ang % 360))
        placeholder.plotly_chart(fig, use_container_width=True, key=f"roulette_{run_key}_{i}")
        time.sleep(0.03)

    # congelar el frame final (opcional)
    placeholder.plotly_chart(fig, use_container_width=True, key=f"roulette_{run_key}_final")

    # 🔁 Importante: NO renderizamos acceptance_panel aquí para evitar duplicados.
    return final_ok


# ================== RIBBON ==================
def decision_ribbon(decision_ok: bool, run_id: str = ""):
    """
    Small ribbon pill for a quick verdict tag. This is independent from the roulette/panel.
    """
    pill_class = "ok" if decision_ok else "no"
    pill_text  = "ACEPTAR" if decision_ok else "NO ACEPTAR"
    st.markdown(
        f"""
        <div class="ribbon">
           <span class="pill {pill_class}">{pill_text}</span>
           <span class="small" style="margin-left:auto;">Run ID: <b>{run_id}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


    

# (A) Poisson PD(t)
def plot_poisson_pd_for_client(lam: float, horizon_months: int = 36, show_linear=True, zoom=True):
    lam = float(max(lam, 0.0))
    t_meses = np.arange(0, int(horizon_months) + 1)
    t_anios = t_meses / 12.0
    pd_cum = 1.0 - np.exp(-lam * t_anios)
    pd12 = 1.0 - np.exp(-lam * 1.0)

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=170)
    ax.plot(t_meses, pd_cum, linewidth=2, label="Poisson: 1 - exp(-λt)")
    if show_linear:
        pd_lin = np.minimum(lam * t_anios, 1.0)
        ax.plot(t_meses, pd_lin, linestyle="--", linewidth=1, label="Aprox. lineal: λt")
    ax.scatter([12], [pd12], s=36)
    ax.annotate(f"PD(12m) = {pd12*100:.2f}%", xy=(12, pd12),
                xytext=(13, min(1.0, pd12 + 0.10)),
                arrowprops=dict(arrowstyle='->', lw=1))
    ax.set_title("Proceso de Poisson — Probabilidad acumulada de default por cliente")
    ax.set_xlabel("Meses"); ax.set_ylabel("Probabilidad acumulada de default")
    ax.set_ylim(0, min(1.0, max(0.12, float(pd_cum.max()) * (1.2 if zoom else 1.0))))
    ax.grid(alpha=0.25); ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

# (B) Extrapolación PD
def plot_poisson_pd_extrapolation(lam: float, horizon_months: int = 60,
                                  show_band: bool = True, band_pct: float = 0.20,
                                  zoom: bool = True):
    lam = float(max(lam, 0.0))
    t_meses = np.arange(0, int(horizon_months) + 1)
    t_anios = t_meses / 12.0
    pd_base = 1.0 - np.exp(-lam * t_anios)

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=170)
    ax.plot(t_meses, pd_base, linewidth=2, label=f"Base (λ={lam:.4f})")
    if show_band and lam > 0:
        lam_lo, lam_hi = lam * (1 - band_pct), lam * (1 + band_pct)
        pd_lo = 1.0 - np.exp(-lam_lo * t_anios)
        pd_hi = 1.0 - np.exp(-lam_hi * t_anios)
        ax.fill_between(t_meses, pd_lo, pd_hi, alpha=0.20, label=f"Banda ±{int(band_pct*100)}% en λ")
    for m in [12, 24, 36, 48, 60, 84, 120]:
        if m <= horizon_months:
            ax.axvline(m, linestyle=":", linewidth=0.8)
            y = pd_base[int(m/12*12)] if m < len(pd_base) else pd_base[-1]
            ax.text(m + 0.4, min(0.98, y + 0.03), f"{y*100:.1f}%", fontsize=9, va="bottom")
    ax.set_title("Extrapolación de PD (Poisson)")
    ax.set_xlabel("Meses"); ax.set_ylabel("Probabilidad acumulada de default")
    ax.set_ylim(0, min(1.0, max(0.15, float(pd_base.max()) * (1.25 if zoom else 1.0))))
    ax.grid(alpha=0.25); ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

# 👉 Poisson con rectángulos bajo la curva (Riemann) + integral en el gráfico
def plot_pd12_accum_area(lam: float,
                         months: int = 12,
                         riemann: str = "left",         # "left" | "right" | "mid"
                         show_delta_labels: bool = False,
                         show_integral: bool = True):    # ← NUEVO
    lam = float(max(lam, 0.0))

    # Tiempo (meses y años)
    t_m = np.arange(0, months + 1, 1)          # 0..months (meses)
    t_y = t_m / 12.0                            # años
    PD   = 1.0 - np.exp(-lam * t_y)             # PD(t_m) = 1 - e^{-λ·(t_m/12)}
    dPD  = np.diff(PD)
    PD12 = PD[12] if months >= 12 else PD[-1]

    # Alturas de rectángulos (Riemann)
    if riemann == "right":
        heights = PD[1:]
    elif riemann == "mid":
        t_mid_y = (np.arange(months) + 0.5) / 12.0
        heights = 1.0 - np.exp(-lam * t_mid_y)
    else:  # "left"
        heights = PD[:-1]

    # Gráfico
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=170)

    # Rectángulos (ancho=1 mes)
    ax.bar(np.arange(months), heights, width=1.0, align='edge',
           color="#3b82f6", alpha=0.18, edgecolor="#3b82f6", linewidth=0.8,
           label="Rectángulos bajo la curva")

    # Curva Poisson (en meses: t_m/12 en el exponente)
    ax.plot(t_m, PD, lw=2.4, color="#1f2937",
            label=r"$PD(t_m)=1-e^{-\lambda\,t_m/12}$")

    # Puntos discretos
    ax.scatter(t_m, PD, s=18, color="#f59e0b", zorder=3, label="PD acumulada (discreta)")

    # Punto + texto en 12m
    if months >= 12:
        ax.scatter([12], [PD12], s=42, color="#ef4444", zorder=4)
        ax.annotate(f"PD(12m) = {PD12*100:.2f}%",
                    xy=(12, PD12),
                    xytext=(12.6, min(0.98, PD12 + 0.10)),
                    arrowprops=dict(arrowstyle='->', lw=1.0, color="#111827"),
                    fontsize=10, color="#111827")

    # Etiquetas ΔPD_m (opcional)
    if show_delta_labels:
        for k in range(months):
            ax.text(k + 0.5, heights[k] + 0.008, f"Δ{(k+1)}: {dPD[k]*100:.2f}%",
                    ha="center", va="bottom", fontsize=9, color="#334155")

    # ✅ Integral en meses (0→12)
    if show_integral:
        # Mostramos ambas formas: meses y años (lado a lado, arriba a la derecha)
        txt = (
    r"$PD(12m)=\int_{0}^{12}\frac{\lambda}{12} e^{-\lambda u/12}\,du \;=\; 1-e^{-\lambda}$"
)

        ax.text(0.98, 0.65, txt,
        ha="right", va="top",
        transform=ax.transAxes,
        fontsize=12, color="#111827",
        bbox=dict(boxstyle="round,pad=0.35",
                  fc="white", ec="#cbd5e1", lw=1, alpha=0.95))

        

    ax.set_title("PD(12m) — rectángulos bajo la curva (Poisson)")
    ax.set_xlabel("Meses"); ax.set_ylabel("Probabilidad acumulada de default")
    ax.set_xlim(0, months)
    ax.set_ylim(0, min(1.0, max(0.12, float(PD.max()) * 1.18)))
    ax.grid(alpha=0.25); ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

def show_pd_monthly_table(lam: float, months: int = 12, riemann: str = "left"):
    lam = float(max(lam, 0.0))
    t = np.arange(0, months + 1) / 12.0                      # años
    PD = 1.0 - np.exp(-lam * t)                              # acumulada Poisson
    dPD = np.diff(PD)                                        # incrementos mensuales
    S_prev = 1.0 - PD[:-1]                                   # supervivencia al inicio de cada mes
    p_cond = 1.0 - np.exp(-lam / 12.0)                       # prob. condicional por mes (constante en Poisson)

    # Altura del rectángulo que estás usando en el gráfico
    if riemann == "right":
        rect_h = PD[1:]                                      # altura = PD fin de mes m
    elif riemann == "mid":
        t_mid = (np.arange(months) + 0.5) / 12.0
        rect_h = 1.0 - np.exp(-lam * t_mid)                  # PD en punto medio
    else:  # "left"
        rect_h = PD[:-1]                                     # altura = PD inicio del mes m

    df = pd.DataFrame({
        "Mes": np.arange(1, months + 1, dtype=int),
        "PD_acum_fin_mes": PD[1:],                           # PD(t_m)
        "ΔPD_m": dPD,                                        # PD(t_m) - PD(t_{m-1})
        "Supervivencia_prev (S_{m-1})": S_prev,              # 1 - PD(t_{m-1})
        "p_cond_mensual": np.full(months, p_cond),           # 1 - e^{-λ/12}
        "Altura_rectángulo": rect_h,                         # según esquema Riemann
    })
    df["ΣΔPD_hasta_m"] = df["ΔPD_m"].cumsum()

    # Fila total (PD12m)
    total_row = pd.DataFrame([{
        "Mes": "Total (12m)",
        "PD_acum_fin_mes": PD[min(12, months)],
        "ΔPD_m": df["ΔPD_m"].sum(),
        "Supervivencia_prev (S_{m-1})": np.nan,
        "p_cond_mensual": p_cond,
        "Altura_rectángulo": np.nan,
        "ΣΔPD_hasta_m": df["ΔPD_m"].sum(),
    }])

    df_out = pd.concat([df, total_row], ignore_index=True)

    st.markdown("**📋 Detalle mensual de PD (Poisson)**")
    st.dataframe(
        df_out.style.format({
            "PD_acum_fin_mes": "{:.2%}",
            "ΔPD_m": "{:.2%}",
            "Supervivencia_prev (S_{m-1})": "{:.2%}",
            "p_cond_mensual": "{:.2%}",
            "Altura_rectángulo": "{:.2%}",
            "ΣΔPD_hasta_m": "{:.2%}",
        }),
        use_container_width=True, hide_index=True
    )
# ==== Helpers visuales (ÚNICA DEFINICIÓN) ====
def fmt_usd(x: float) -> str:
    try:
        return f"$ {float(x):,.0f}"
    except Exception:
        return "$ —"

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.strip().lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{float(alpha)})"

DEFAULT_TREE = {
    "edge": "#94a3b8",
    "label": "#0f172a",
    "pay1": "#18B277",
    "pd1":  "#eab308",
    "cura": "#60a5fa",
    "def2": "#ef4444",
}



# ───────────── (C) Árbol probabilístico — clásico (matplotlib) ──────────
def plot_decision_tree_clean(
    res: dict, mv: int, mp: int, cliente: str = "",
    scale_by_prob: bool = True,
):
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    PD1     = float(res.get("PD1", 0.0))
    PD2     = float(res.get("PD2_cond", 0.0))
    PV_t1   = float(res.get("PV_t1", 0.0))
    PV_cura = float(res.get("PV_t2_cura", 0.0))
    PV_def  = float(res.get("PV_t2_def", 0.0))

    x0, x1, x2 = 0.08, 0.43, 0.78
    bw, bh = 0.32, 0.135

    green, green2 = "#2E7D32", "#216328"
    edge, txt, txt2 = "#0f172a", "#ffffff", "#253045"

    def box(ax, x, y, w, h, lines, fsize=12, vpad=0.22):
        if isinstance(lines, str): lines = [lines]
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.016,rounding_size=0.022",
            fc=green, ec=green2, lw=1.4, zorder=3
        )
        rect.set_path_effects([
            pe.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace=(0, 0, 0, 0.18)),
            pe.Normal()
        ])
        ax.add_patch(rect)
        ny = len(lines)
        for i, line in enumerate(lines):
            y_line = y + h/2 + (((ny-1)/2 - i) * vpad * h)
            ax.text(x + w/2, y_line, line, ha="center", va="center",
                    fontsize=fsize, color=txt, zorder=4)

    def pill(ax, x, y, s):
        ax.text(x, y, s, fontsize=10, color=txt2, ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.20",
                          fc="#ffffff", ec="#cbd5e1", lw=1.0))

    def connect(ax, x1_, y1_, x2_, y2_, prob, label, up=True, t=0.30, ybump=0.045):
        lw = min(5.0, 1.6 + (4.0 * prob if scale_by_prob else 0.0))
        rad = -0.18 if up else 0.18
        arr = FancyArrowPatch((x1_, y1_), (x2_, y2_),
                              arrowstyle='-|>', mutation_scale=14,
                              lw=lw, color=edge,
                              connectionstyle=f"arc3,rad={rad}", zorder=2)
        arr.set_capstyle('round'); ax.add_patch(arr)
        xm = x1_ + t*(x2_-x1_); ym = y1_ + t*(y2_-y1_) + (ybump if up else -ybump)
        pill(ax, xm, ym, label)

    fig, ax = plt.subplots(figsize=(12.2, 5.9), dpi=190)
    ax.set_facecolor("#ffffff"); ax.axis("off")
    title = f"Árbol probabilístico de flujos — {cliente}" if cliente else "Árbol probabilístico de flujos"
    ax.text(0.02, 0.965, title, fontsize=20, color=edge, ha="left", va="top", transform=ax.transAxes)

    box(ax, x0, 0.56 - bh/2, bw, bh, ["Financiamiento por", r"$X$"])
    box(ax, x1, 0.80 - bh/2, bw, bh, ["Te pagan y recibes", rf"$X(1+i)^{{{mv}}}$", f"PV = {fmt_usd(PV_t1)}"])
    box(ax, x1, 0.30 - bh/2, bw, bh, [r"No pagan en $t_1$", r"pero puedes recuperar en $t_2$", f"Recovery = {max(0,1-res.get('LGD',0.0))*100:.2f}%", f"LGD = {res.get('LGD',0.0)*100:.2f}%"], fsize=11, vpad=0.20)
    box(ax, x2, 0.62 - bh/2, bw, bh, [r"Te pagan en $t_2$", rf"$X(1+i)^{{{mv+mp}}}(1+m)^{{{mp}}}$", f"PV = {fmt_usd(res.get('PV_t2_cura',0.0))}"], fsize=11, vpad=0.20)
    box(ax, x2, 0.14 - bh/2, bw, bh, [r"Default definitivo", r"$\mathrm{PV} = w_G \cdot G$", f"w_G = {res.get('w_G',1.0)*100:.0f}%  ·  G = {fmt_usd(res.get('G_total',0.0))}", f"PV = {fmt_usd(res.get('PV_t2_def',0.0))}"], fsize=11, vpad=0.20)

    connect(ax, x0+bw, 0.56, x1, 0.80, 1-res['PD1'], f"1 − PD₁ = {(1-res['PD1']):.1%}", up=True,  t=0.30, ybump=0.045)
    connect(ax, x0+bw, 0.56, x1, 0.30, res['PD1'],     f"PD₁ = {res['PD1']:.1%}",         up=False, t=0.30, ybump=0.045)
    connect(ax, x1+bw, 0.30, x2, 0.62, 1-res['PD2_cond'], f"1 − PD₂ = {(1-res['PD2_cond']):.1%}", up=True,  t=0.63, ybump=0.055)
    connect(ax, x1+bw, 0.30, x2, 0.14, res['PD2_cond'],   f"PD₂ = {res['PD2_cond']:.1%}",         up=False, t=0.63, ybump=0.055)

    ax.text(0.02, 0.055, rf"$i_m={res['tc_m']:.4f}$ (mensual)  ·  $m_m={res['tm_m']:.4f}$ (mensual)  ·  $mv={mv}$, $mp={mp}$",
            fontsize=11, color="#475569", transform=ax.transAxes)

    plt.subplots_adjust(left=0.02, right=0.985, top=0.93, bottom=0.08)
    st.pyplot(fig, use_container_width=True)

# ───────────── (D) Contribuciones EV — tabla ─────────────
def show_ev_contributions(res: dict):
    PD1     = float(res.get("PD1", 0.0))
    PD2     = float(res.get("PD2_cond", 0.0))
    PV_t1   = float(res.get("PV_t1", 0.0))
    PV_cura = float(res.get("PV_t2_cura", 0.0))
    PV_def  = float(res.get("PV_t2_def", 0.0))

    p_pay1  = 1 - PD1
    p_cura  = PD1 * (1 - PD2)
    p_def2  = PD1 * PD2

    ev_pay1 = p_pay1 * PV_t1
    ev_cura = p_cura * PV_cura
    ev_def2 = p_def2 * PV_def
    EV_VP   = ev_pay1 + ev_cura + ev_def2

    df = pd.DataFrame({
        "Rama": ["t₁: paga", "t₂: cura", "t₂: default final"],
        "Prob.": [p_pay1, p_cura, p_def2],
        "PV":    [PV_t1,  PV_cura, PV_def],
        "Aporte EV (VP)": [ev_pay1, ev_cura, ev_def2],
    })
    st.markdown("**Aporte al EV (VP) por rama**")
    st.dataframe(
        df.style.format({
            "Prob.": "{:.1%}",
            "PV":    lambda x: fmt_usd(x),
            "Aporte EV (VP)": lambda x: fmt_usd(x),
        }),
        use_container_width=True, hide_index=True
    )
    st.markdown(f"**EV (VP) total:** {fmt_usd(EV_VP)}")

# ═════════════════════════  NEURAL “PROBABILITY NETWORK”  ════════════════
def _lerp(a, b, t): return a + (b - a) * t
def _rgb_hex(r,g,b): return f"rgba({int(r)},{int(g)},{int(b)},1.0)"

def color_for_prob(p: float) -> str:
    p = clamp(p, 0.0, 1.0)
    g = (24,178,119)   # #18B277
    y = (203,161,53)   # #CBA135
    r = (224,95,95)    # #E05F5F
    if p <= 0.6:
        t = p/0.6
        c = (_lerp(g[0], y[0], t), _lerp(g[1], y[1], t), _lerp(g[2], y[2], t))
    else:
        t = (p-0.6)/0.4
        c = (_lerp(y[0], r[0], t), _lerp(y[1], r[1], t), _lerp(y[2], r[2], t))
    return _rgb_hex(*c)

def _curve(x1,y1,x2,y2,curv=0.18, n=30):
    xm = (x1+x2)/2.0
    ym = (y1+y2)/2.0 + curv*(x2-x1)
    t = np.linspace(0,1,n)
    bx = (1-t)**2 * x1 + 2*(1-t)*t * xm + t**2 * x2
    by = (1-t)**2 * y1 + 2*(1-t)*t * ym + t**2 * y2
    return bx, by

def plot_neural_probability_tree(
    res: dict, mv: int, mp: int,
    mode: str = "Probabilidad",
    show_labels: bool = True,
    scale_strength: float = 1.0,
    title: str = "Neural Probability Network",
    show_time_titles: bool = True,
    time_title_style: str = "node",  # "node" | "column" | "both"
    highlight: str | None = None,
    palette: dict | None = None,
):
    import plotly.graph_objects as go

    pal = (palette or st.session_state.get("pal_tree") or {}).copy()
    pal = {**DEFAULT_TREE, **pal}  # fallback seguro

    PD1   = float(res.get("PD1", 0.0))
    PD2   = float(res.get("PD2_cond", 0.0))
    PV1   = float(res.get("PV_t1", 0.0))
    PVc   = float(res.get("PV_t2_cura", 0.0))
    PVd   = float(res.get("PV_t2_def", 0.0))

    p_pay1, p_cura, p_def2 = (1-PD1), (PD1*(1-PD2)), (PD1*PD2)
    ev_pay1, ev_cura, ev_def2 = p_pay1*PV1, p_cura*PVc, p_def2*PVd
    ev_sum = max(1e-9, abs(ev_pay1)+abs(ev_cura)+abs(ev_def2))

    nodes = {
        "root": {"x":0.06,"y":0.50,"label":"t₀: Emisión X"},
        "pay1": {"x":0.40,"y":0.80,"label":f"t₁: Paga\nPV={fmt_usd(PV1)}"},
        "def1": {"x":0.40,"y":0.20,"label":f"t₁: No paga\nPD₁={PD1:.1%}"},
        "cura2":{"x":0.80,"y":0.70,"label":f"t₂: Cura\nPV={fmt_usd(PVc)}"},
        "def2": {"x":0.80,"y":0.10,"label":f"t₂: Default final\nPV={fmt_usd(PVd)}"},
    }
    edges = [
        ("root","pay1", p_pay1, ev_pay1,   0.20, "pay1"),
        ("root","def1", PD1,    ev_cura+ev_def2, -0.20, "pd1"),
        ("def1","cura2", 1-PD2, ev_cura,   0.20, "cura"),
        ("def1","def2",  PD2,   ev_def2,  -0.20, "def2"),
    ]

    def _curve(x1,y1,x2,y2,curv=0.18,n=40):
        import numpy as np
        xm=(x1+x2)/2; ym=(y1+y2)/2+curv*(x2-x1)
        t=np.linspace(0,1,n)
        bx=(1-t)**2*x1+2*(1-t)*t*xm+t**2*x2
        by=(1-t)**2*y1+2*(1-t)*t*ym+t**2*y2
        return bx,by

    fig = go.Figure()

    # Nodos
    for k,v in nodes.items():
        fig.add_trace(go.Scatter(
            x=[v["x"]], y=[v["y"]], mode="markers+text",
            marker=dict(size=18, color="white",
                        line=dict(width=2,color="rgba(12,27,42,0.85)")),
            text=[v["label"]], textfont=dict(color=pal["label"]),
            textposition="top center",
            hoverinfo="skip", showlegend=False
        ))

    # Aristas
    for u,v,p,ev,curv,ck in edges:
        strength = (p if mode=="Probabilidad" else abs(ev)/ev_sum)
        width = 2.0 + 16.0*(strength**0.7)*float(scale_strength)
        color = pal.get(ck, pal["edge"])
        bx,by = _curve(nodes[u]["x"],nodes[u]["y"],nodes[v]["x"],nodes[v]["y"],curv=curv)
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(width=width, color=color, shape="spline"),
            hovertemplate=(f"{u}→{v}<br>p={p:.2%}" if mode=="Probabilidad"
                           else f"{u}→{v}<br>aporte EV={fmt_usd(ev)}")+"<extra></extra>",
            showlegend=False
        ))
        if show_labels:
            mid=len(bx)//2
            label = f"{p:.1%}" if mode=="Probabilidad" else fmt_usd(ev)
            fig.add_trace(go.Scatter(
                x=[bx[mid]], y=[by[mid]], mode="text",
                text=[label], textfont=dict(color=pal["label"]),
                textposition="top center",
                hoverinfo="skip", showlegend=False
            ))

    # ── Strong, always-visible time headers ───────────────────────────────
    if show_time_titles:
        tcolor = pal.get("time_label", "#0f172a")
        gcolor = pal.get("time_grid",  "rgba(15,23,42,0.18)")
        bcolor = pal.get("time_band",  "rgba(15,23,42,0.04)")

        # optional faint bands per column (helps legibility even on screenshots)
        if time_title_style in ("column", "both"):
            # left band around x≈0.06, mid at 0.40, right at 0.80
            band_w = 0.18  # width of each band
            for xc in (0.06, 0.40, 0.80):
                fig.add_vrect(x0=xc-band_w/2, x1=xc+band_w/2, fillcolor=bcolor,
                              line_width=0, layer="below")

        # visible grid lines + headers
        if time_title_style in ("column", "both"):
            for x, txt in ((0.06,"<b>T = 0</b>"), (0.40,"<b>T = 1</b>"), (0.80,"<b>T = 2</b>")):
                fig.add_vline(x=x, line=dict(width=1.0, color=gcolor))
                fig.add_annotation(x=x, y=1.03, text=txt, showarrow=False,
                                   font=dict(color=tcolor, size=14),
                                   bgcolor="rgba(255,255,255,0.85)",  # halo for contrast
                                   bordercolor="rgba(0,0,0,0.08)", borderwidth=1)

        if time_title_style in ("node", "both"):
            # place slightly inside the plotting area above nodes
            fig.add_annotation(x=nodes["root"]["x"], y=0.97, text="<b>T = 0</b>", showarrow=False,
                               font=dict(color=tcolor, size=14),
                               bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.08)", borderwidth=1)
            fig.add_annotation(x=0.40, y=0.97, text="<b>T = 1</b>", showarrow=False,
                               font=dict(color=tcolor, size=14),
                               bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.08)", borderwidth=1)
            fig.add_annotation(x=0.80, y=0.97, text="<b>T = 2</b>", showarrow=False,
                               font=dict(color=tcolor, size=14),
                               bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.08)", borderwidth=1)

    # Nodo destacado (opcional)
    if highlight in ("pay1","cura2","def2"):
        hx,hy = nodes[highlight]["x"], nodes[highlight]["y"]
        fig.add_trace(go.Scatter(x=[hx], y=[hy], mode="markers",
                                 marker=dict(size=60, color=_hex_to_rgba(pal.get(highlight.replace("2",""), "#CBA135"), 0.10)),
                                 hoverinfo="skip", showlegend=False))

    fig.update_layout(
        title=title,
        title_font_color=pal.get("label", "#0f172a"),  # ensure high-contrast title
        margin=dict(l=20, r=20, t=56, b=20),
        xaxis=dict(visible=False, range=[0,1]),
        yaxis=dict(visible=False, range=[0,1]),
        paper_bgcolor="white", plot_bgcolor="white",
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

# --- put these next to your other viz helpers ---
def plot_hmm_3state_graph(meta: dict, mv: int, mp: int,
                           title="HMM (3 estados) — transiciones mensuales",
                           palette: dict | None = None):
    import numpy as np
    import plotly.graph_objects as go

    pal = (palette or st.session_state.get("pal_hmm") or {}).copy()
    pal = {**DEFAULT_HMM, **pal}  # fallback seguro

    A = np.array(meta["A"], dtype=float)
    alpha, gamma, delta, beta = meta["alpha"], meta["gamma"], meta["delta"], meta["beta"]
    s1_stay = A[1, 1]

    # layout + color de halo por nodo
    nodes = {
        "P":  dict(x=0.12, y=0.70, halo=_hex_to_rgba(pal["node_P"], 0.12)),
        "S1": dict(x=0.58, y=0.70, halo=_hex_to_rgba(pal["node_S1"], 0.12)),
        "S2": dict(x=0.58, y=0.22, halo=_hex_to_rgba(pal["node_S2"], 0.12)),
    }

    def add_node(fig, key, label):
        nd = nodes[key]
        fig.add_trace(go.Scatter(
            x=[nd["x"]], y=[nd["y"]], mode="markers+text",
            marker=dict(size=34, color="white",
                        line=dict(width=3, color=pal["node_edge"])),
            text=[f"<b>{label}</b>"], textposition="top center",
            hovertemplate=f"<b>{label}</b><extra></extra>", showlegend=False
        ))
        # halo de color del nodo
        fig.add_trace(go.Scatter(
            x=[nd["x"]], y=[nd["y"]], mode="markers",
            marker=dict(size=84, color=nd["halo"]),
            hoverinfo="skip", showlegend=False
        ))

    def curve(x1, y1, x2, y2, curv=0.18, n=30):
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2 + curv * (x2 - x1)
        t = np.linspace(0, 1, n)
        bx = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * xm + t ** 2 * x2
        by = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * ym + t ** 2 * y2
        return bx, by

    def add_edge(fig, u, v, p, label=None, curv=0.18, key_color="stay"):
        if p <= 1e-9:
            return
        ux, uy = nodes[u]["x"], nodes[u]["y"]
        vx, vy = nodes[v]["x"], nodes[v]["y"]
        bx, by = curve(ux, uy, vx, vy, curv=curv)
        width = 2.0 + 14.0 * (p ** 0.7)
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(width=width, color=pal.get(key_color, pal["stay"]), shape="spline"),
            hovertemplate=f"{u}→{v}<br>p={p:.2%}<extra></extra>",
            showlegend=False
        ))
        mid = len(bx) // 2
        fig.add_trace(go.Scatter(
            x=[bx[mid]], y=[by[mid]], mode="text",
            text=[label or f"{p:.1%}"], textposition="top center",
            hoverinfo="skip", showlegend=False
        ))

    fig = go.Figure()
    for k, label in {"P": "P (performing)", "S1": "S1 (distressed)", "S2": "S2 (default)"}.items():
        add_node(fig, k, label)

    # aristas mensuales (A)
    add_edge(fig, "P",  "P",  A[0, 0], "stay", curv=-0.12, key_color="stay")
    add_edge(fig, "P",  "S1", A[0, 1], f"α={alpha:.2%}", key_color="alpha")
    add_edge(fig, "P",  "S2", A[0, 2], f"β={beta:.2%}" if beta > 0 else None, curv=-0.10, key_color="beta")
    add_edge(fig, "S1", "P",  A[1, 0], f"γ={gamma:.2%}", key_color="gamma")
    add_edge(fig, "S1", "S1", A[1, 1], "stickiness" if s1_stay > 0 else None, curv=-0.12, key_color="stay")
    add_edge(fig, "S1", "S2", A[1, 2], f"δ={delta:.2%}", curv=0.10, key_color="delta")
    add_edge(fig, "S2", "S2", 1.0, "absorbente", curv=0.0, key_color="stay")

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        paper_bgcolor="white", plot_bgcolor="white", height=420
    )
    return fig



def hmm_formula_block(meta: dict, mv: int, mp: int):
    """Show compact HMM math and transition matrices; no dependence on the graph's 'nodes'."""
    import numpy as np
    import pandas as pd
    import streamlit as st

    A = np.array(meta["A"], dtype=float)
    PD1 = float(meta.get("PD1", np.nan))
    PD2c = float(meta.get("PD2_cond", np.nan))

    st.markdown(r"""
**Formulación (resumen)**  
Estados: \( \{P, S1, S2\} \); \(S2\) absorbente.  
Matriz de transición **mensual** \(A\). Con \(\pi_0=[1,0,0]\),

- \( \pi_{t_1}=\pi_0 A^{mv} \Rightarrow PD_1 = \pi_{t_1}[S1] \)  
- \( q_{mp} = e_{S1}^\top A^{mp} \Rightarrow PD_{2|1} = q_{mp}[S2] \)
""")

    # A (mensual)
    dfA = pd.DataFrame(A, index=["P", "S1", "S2"], columns=["P", "S1", "S2"])
    st.write(dfA.style.format("{:.3f}"))

    # A^mv and A^mp quick check (optional but helpful)
    try:
        A_mv = np.linalg.matrix_power(A, int(mv))
        A_mp = np.linalg.matrix_power(A, int(mp))
        pi0 = np.array([1.0, 0.0, 0.0])
        pi_t1 = pi0 @ A_mv
        q_from_S1 = np.array([0.0, 1.0, 0.0]) @ A_mp
        st.caption(
            f"A^mv π₀ → πₜ₁ = {np.round(pi_t1, 4).tolist()}  ·  "
            f"e_S1^T A^mp → {np.round(q_from_S1, 4).tolist()}"
        )
    except Exception:
        pass  # keep silent if user enters non-integer mv/mp

    st.markdown(f"**PD₁:** {PD1:.2%} &nbsp;&nbsp; **PD₂|₁:** {PD2c:.2%}  &nbsp;&nbsp; *(mv={mv}, mp={mp})*")


@st.cache_data(show_spinner=False)
def simulate_paths(PD1: float, PD2: float, n:int=10000, seed:int=42):
    rng = np.random.default_rng(int(seed))
    u1 = rng.random(n)
    t1_def = u1 < PD1
    u2 = rng.random(n)
    def2 = t1_def & (u2 < PD2)
    cura = t1_def & (~def2)
    pay1 = ~t1_def
    return dict(
        pay1=int(pay1.sum()),
        cura=int(cura.sum()),
        def2=int(def2.sum()),
        n=n
    )

# ───────────────────── 3) CARGA CARTERA (defensivo + caché) ─────────────
@st.cache_data(show_spinner=False)
def load_cartera(uploaded):
    df = pd.read_excel(uploaded, sheet_name="CARTERA")
    df = df.rename(columns=lambda c: str(c).strip())
    return df

def demo_portfolio() -> pd.DataFrame:
    return pd.DataFrame({
        "Cliente": ["DemoCo A", "DemoCo B", "DemoCo C"],
        "Exposición USD": [1_000_000, 750_000, 1_250_000],
        "GARANTIAS": [600_000, 300_000, 900_000],
        "Peso de la Garantía": ["80%", "60%", "75%"],
        "TIPO DE GARANTIA": [
            "Hipoteca predio urbano; Cheque representante (medio patrimonio)",
            "Letra/Pagaré +/- Aval (medio patrimonio)",
            "Fideicomiso planta de packing"
        ],
    })

def sha256_file(f) -> str:
    try:
        pos = f.tell()
        f.seek(0); h = hashlib.sha256(f.read()).hexdigest(); f.seek(pos)
        return h
    except Exception:
        return ""

st.sidebar.header("📂 Cargar cartera (opcional)")
uploaded = st.sidebar.file_uploader("Sube OPINT.xlsx", type=["xlsx"])
file_sha = ""
if uploaded:
    try:
        file_sha = sha256_file(uploaded)
        df_cart = load_cartera(uploaded)
        needed = ["Exposición USD", "GARANTIAS"]
        missing = [c for c in needed if c not in df_cart.columns]
        if missing:
            raise ValueError(f"Faltan columnas: {', '.join(missing)}")
        st.sidebar.success("Cartera cargada ✅")
    except Exception as e:
        st.sidebar.error(f"❌ Error leyendo Excel: {e}")
        df_cart = demo_portfolio()
else:
    df_cart = demo_portfolio()
    st.sidebar.info("Usando cartera demo (sin archivo).")

cliente = st.sidebar.selectbox("Cliente:", df_cart.iloc[:, 0])
st.session_state["cliente"] = cliente
row = df_cart[df_cart.iloc[:, 0] == cliente].iloc[0]
EAD0 = float(row.get("Exposición USD", 1_000_000.0))
garantia_val0 = float(row.get("GARANTIAS", 0.0))

# w_G desde la COLUMNA M (con fallback por nombre)
try:
    peso_total0 = peso_garantia_col_M(row)
except Exception:
    peso_total0 = 1.0

df_g_inicial = construir_garantias_desde_CARTERA(row, garantia_val0)
st.session_state.update({
    "df_g_inicial": df_g_inicial,
    "EAD0": EAD0, "garantia_val0": garantia_val0,
    "peso_total0": float(peso_total0),
    "w_G": float(peso_total0)
})

st.sidebar.markdown("### 🧮 Calculadora: mensual → anual")
tasa_m_maybe = st.sidebar.number_input(
    "Tasa mensual (decimal, ej. 0.025 ↔ 2.50%)",
    min_value=-0.99, max_value=10.0, value=0.025, step=0.0005, format="%.4f"
)
tasa_anual_eq = to_annual_from_monthly(tasa_m_maybe)
st.sidebar.write(f"**Anual equivalente:** {tasa_anual_eq:.6f}  ({tasa_anual_eq*100:.2f}%)")

# (Opcional) botón para copiarla a tu tc_ann global
if st.sidebar.button("Usar como tc anual (global)"):
    st.session_state["tc_ann"] = float(tasa_anual_eq)
    st.sidebar.success(f"tc_ann actualizado a {tasa_anual_eq*100:.2f}%")


# ───────────────────── 4) NAVEGACIÓN ────────────────────────────────────
page = st.sidebar.radio("📌 Menú",
    ["Decidir (simple)", "Comité (avanzado)", "Simple (Q&A)", "Inputs",
     "Árbol probabilístico", "PD Learner (beta)", "Policy Lab (RL beta)", "AquilaEigenPD"],
    index=0)

if page == "AquilaEigenPD":
    render_aquilaeigen_tab()

    

# ——— Umbral de retorno (RE simple) ———
RET_THRESHOLD_DEFAULT = 0.12
RET_THRESHOLD = st.sidebar.number_input(
    "Umbral de retorno (RE simple)",
    min_value=0.0, max_value=1.0,
    value=float(RET_THRESHOLD_DEFAULT),
    step=0.01, format="%.2f"
)
st.session_state["ret_threshold"] = RET_THRESHOLD

# --- Glosario en barra lateral (solo si la sidebar está visible) ---
if not PRESENT:  # ya tienes este flag
    with st.sidebar:
        st.divider()  # opcional: separador visual
        with st.popover("📚 Glosario rápido"):
            st.markdown("""
**PD(12m):** prob. de default en 12 meses.  
**LGD:** pérdida en caso de default (1 − recuperación).  
**RE (simple):** `tc·(1−PD12) − LGD·PD12`.  
**EV (VP):** valor esperado descontado del árbol t₀→t₁→t₂.  
**mv/mp:** meses a t₁ / meses t₁→t₂.  
**HMM:** 3 estados {P,S1,S2} para refinar timing de default/cura.
""")


# ───────────────────── 5) VALIDACIONES BLANDAS ──────────────────────────
def soft_checks(tc_ann, co_ann, garantia_total, gastos_usd):
    if tc_ann < co_ann:
        st.warning("⚠️ La tasa compensatoria (tc) es menor al costo de oportunidad (CO).", icon="⚠️")
    if gastos_usd > garantia_total:
        st.warning("⚠️ Los gastos superan la recuperación bruta esperada.", icon="⚠️")

# ───────────────────── 5.1) Manifiesto (auditoría) helpers ──────────────
def build_manifest(extra: dict) -> dict:
    manifest = {
        "run_id": secrets.token_hex(4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "app_version": APP_VERSION,
        "file_sha256": file_sha or "demo",
        "anchors": {"PD12_score1": PD12_ANCLA_1, "PD12_score5": PD12_ANCLA_5},
    
        "user": "analyst",
        "tm_ann_fija": TM_ANUAL_FIJO,
    }
    manifest.update(extra or {})
    st.session_state["manifest"] = manifest
    return manifest
def mini_manifest_ribbon(mani: dict | None):
    """Small run badge for auditability."""
    if not mani:
        return
    rid = str(mani.get("run_id", "—"))
    ver = str(mani.get("app_version", "—"))
    sha = str(mani.get("file_sha256", "—"))
    st.markdown(f"""
    <div class="card" style="padding:10px 12px; display:flex; gap:12px; align-items:center;">
      <span class="badge">Run ID: <b>{rid}</b></span>
      <span class="small">v{ver}</span>
      <span class="small">file <code style="font-size:.78rem">{sha[:8]}</code></span>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────── 6) PÁGINAS ────────────────────────────────────────
# ───────────────────── 6) PÁGINAS ────────────────────────────────────────

# 6.1 Decidir (simple)
if page == "Decidir (simple)":
    st.markdown('<div class="card"><h3 style="margin:0;color:#e8eef8">✅ Decidir — Interfaz simple</h3></div>', unsafe_allow_html=True)

    # --- Inputs (simple) ---
    EAD0 = float(st.session_state.get("EAD0", 1_000_000.0))
    garantia_val0 = float(st.session_state.get("garantia_val0", 0.0))

    colA, colB, colC = st.columns(3)
    with colA:
        score  = st.slider("Score (1=alto riesgo, 5=bajo)", 1.0, 5.0, float(st.session_state.get("score", 3.0)), 0.01)
        tc_ann = st.number_input("Tasa compensatoria anual (tc)", 0.0, 2.0,
                                 float(st.session_state.get("tc_ann", 0.25)), 0.005, format="%.3f")
    with colB:
        EAD = st.number_input("EAD (USD)", 0.0, 1e12, float(EAD0), 10_000.0, format="%.0f")
        garantia_val0 = st.number_input("Garantías (USD)", 0.0, 1e12, float(garantia_val0), 10_000.0, format="%.0f")
    with colC:
        gastos_usd = st.number_input("Gastos (USD)", 0.0, 1e9, float(st.session_state.get("gastos_usd", 0.0)), 1000.0, format="%.0f")

    # Política simple: CO fijo + horizontes por defecto
    co_ann = CO_ANUAL_FIJO
    mv, mp = 12, 3

    # Garantías
    df_g = pd.DataFrame([{"Tipo":"Garantía", "Valor USD": float(garantia_val0)}], columns=["Tipo","Valor USD"])

    soft_checks(tc_ann, co_ann, garantia_val0, gastos_usd)

    # Cálculo principal
    res = calcular_resultados_curay(
        score, EAD, tc_ann, co_ann, mv, mp, df_g, gastos_usd,
        w_guarantee=float(st.session_state.get("peso_total0", 1.0))
    )

    manifest = build_manifest({
        "page": "Decidir",
        "cliente": str(cliente),
        "inputs": {"score": score, "EAD": EAD, "tc_ann": tc_ann, "co_ann": co_ann,
                   "tm_ann_fija": TM_ANUAL_FIJO, "mv": mv, "mp": mp,
                   "garantias": garantia_val0, "gastos": gastos_usd}
    })

    # --- Botón y veredicto (sin duplicar RE en KPIs) ---
    col_btn = st.columns([1,3])[0]
    with col_btn:
        if st.button("🎯 Determinar aceptación", type="primary", key="accept_btn"):
            st.session_state["accept_run_id"] = secrets.token_hex(4)
            st.session_state["accept_clicked"] = True

    if st.session_state.get("accept_clicked"):
        decision_ok = roulette_acceptance(res, ret_threshold=RET_THRESHOLD)   # animación
        decision_ribbon(decision_ok, manifest["run_id"])                      # pequeña cinta
        acceptance_panel(res, ret_threshold=RET_THRESHOLD)                    # RE mostrado UNA vez

    # KPIs (sin RE para evitar duplicación)
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("PD (12m)", fmt_pct(res['PD_12m']))
    with k2: kpi_card("LGD (política)", fmt_pct(res['LGD']))
    with k3: kpi_card("Recovery rate", fmt_pct(res['RECOVERY_RATE']))
    with k4: kpi_card("ECL (política)", fmt_usd(res['ECL']))

    # Gráfico PD(t)
    with st.expander("📈 Riesgo — PD (Poisson) por cliente", expanded=False):
        st.subheader(f"Curva PD(t) — {cliente}")
        plot_poisson_pd_for_client(res["lam"], horizon_months=36, show_linear=True, zoom=True)
        st.caption(f"Hazard λ = {res['lam']:.4f}. PD(12m) = {res['PD_12m']:.2%}.")

    # Estado
    st.session_state.update({
        "RE_anual_simple": res["RE_anual_simple"], "PD_12m": res["PD_12m"], "LGD": res["LGD"], "ECL": res["ECL"],
        "EAD": EAD, "score": score, "tc_ann": tc_ann, "co_ann": co_ann, "gastos_usd": gastos_usd,
        "garantia_val0": garantia_val0, "manifest": manifest
    })

# 6.2 Comité (avanzado)
elif page == "Comité (avanzado)":
    st.markdown('<div class="card"><h3 style="margin:0;color:#e8eef8">🧭 Comité (avanzado) — VP y RE vs CO (tm fija)</h3></div>', unsafe_allow_html=True)

    EAD = float(st.session_state.get("EAD0", 1_000_000.0))
    garantia_val0 = float(st.session_state.get("garantia_val0", 0.0))

    colA, colB, colC = st.columns(3)
    with colA:
        score = st.slider("Score (1=alto riesgo, 5=bajo)", 1.0, 5.0, float(st.session_state.get("score", 3.0)), 0.01)
        tc_ann = st.number_input("Tasa compensatoria anual (tc)", 0.0, 2.0, float(st.session_state.get("tc_ann", 0.25)), 0.005, format="%.3f")
    with colB:
        EAD = st.number_input("EAD (USD)", 0.0, 1e12, float(EAD), 10_000.0, format="%.0f")
        co_ann = st.number_input("Costo de oportunidad anual (CO)", min_value=float(CO_ANUAL_FIJO),
                                 max_value=float(CO_ANUAL_FIJO), value=float(CO_ANUAL_FIJO),
                                 step=0.0001, format="%.4f")
    with colC:
        meses_venc = st.number_input("Meses a vencimiento (mv)", 1, 60, int(st.session_state.get("mv", 12)))
        meses_post = st.number_input("Meses post-default (mp)", 1, 24, int(st.session_state.get("mp", 3)))

    with st.expander("⚙️ Parámetros de PD (anclas)"):
        pd12_s1, pd12_s5 = st.columns(2)
        with pd12_s1:
            pd1 = st.number_input("Ancla PD12 (score 1)", 0.0, 0.999, float(PD12_ANCLA_1), 0.01, format="%.3f")
        with pd12_s5:
            pd5 = st.number_input("Ancla PD12 (score 5)", 0.0, 0.999, float(PD12_ANCLA_5), 0.01, format="%.3f")
        st.caption("tm anual fija en 0.75% • CO anual fijo en 1.50%")

    df_g = st.session_state.get("df_g_inicial")
    if df_g is None or df_g.empty:
        df_g = pd.DataFrame([{"Tipo":"Garantía", "Valor USD": float(garantia_val0)}], columns=["Tipo","Valor USD"])

    soft_checks(tc_ann, CO_ANUAL_FIJO, garantia_val0, float(st.session_state.get("gastos_usd", 0.0)))

    res = calcular_resultados_curay(
        score, EAD, tc_ann, co_ann, meses_venc, meses_post, df_g, float(st.session_state.get("gastos_usd", 0.0)),
        w_guarantee=float(st.session_state.get("peso_total0", 1.0)),
        pd12_anchor_1=pd1, pd12_anchor_5=pd5
    )
    st.session_state["pd12_anchor_1"] = pd1
    st.session_state["pd12_anchor_5"] = pd5

    manifest = build_manifest({
        "page": "Comité",
        "cliente": str(cliente),
        "inputs": {
            "score": score, "EAD": EAD, "tc_ann": tc_ann, "co_ann": co_ann,
            "tm_ann_fija": TM_ANUAL_FIJO, "mv": meses_venc, "mp": meses_post,
            "garantias": garantia_val0, "gastos": float(st.session_state.get("gastos_usd", 0.0)),
            "anchors": {"pd12_s1": pd1, "pd12_s5": pd5}
        },
        "outputs": {"ret_ann_vp": res["ret_ann_vp"], "RE_anual_simple": res["RE_anual_simple"]}
    })

    # Criterio de decisión (RE simple)
    ret_val = float(res.get("RE_anual_simple", float("nan")))
    decision_ok = np.isfinite(ret_val) and (ret_val >= float(RET_THRESHOLD))

    decision_ribbon(decision_ok, manifest["run_id"])
    acceptance_panel(res, ret_threshold=RET_THRESHOLD)

    # KPIs (avanzado): aquí sí mostramos métricas VP
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Retorno anualizado (VP)", fmt_pct(res['ret_ann_vp']), "Incluye timing (Texp)")
    with c2: kpi_card("EV/EAD (VP)", "—" if not np.isfinite(res['multiplo_vp']) else f"{res['multiplo_vp']:.3f}", "Multiplicador esperado")
    with c3: kpi_card("RE (anual, simple)", fmt_pct(res['RE_anual_simple']), "Sin timing/curas")
    with c4: kpi_card("ECL (política)", fmt_usd(res['ECL']), "IFRS-9 (12m)")

    with st.expander("📈 Riesgo — PD (Poisson) por cliente", expanded=False):
        st.subheader(f"Curva PD(t) · {cliente}")
        plot_poisson_pd_for_client(res["lam"], horizon_months=36, show_linear=True, zoom=True)
        st.caption(f"Hazard λ = {res['lam']:.4f}. PD(12m) = {res['PD_12m']:.2%}.")

    st.session_state.update({
        "df_g_calc": df_g.copy(),
        "PD_12m": res["PD_12m"], "LGD": res["LGD"],
        "RE_anual_simple": res["RE_anual_simple"],
        "EAD": EAD, "score": score, "tc_ann": tc_ann, "co_ann": co_ann,
        "garantia_val0": garantia_val0, "manifest": manifest
    })

# 6.3 Simple (Q&A)
elif page == "Simple (Q&A)":
    st.markdown('<div class="card"><h3 style="margin:0;color:#e8eef8">🧩 Simple (Q&A) — Score 50/20/30 (tm fija)</h3></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        q_emp = st.radio("Riesgo de la empresa", ["Bajo", "Medio", "Alto"], index=1)

    with col2:
        q_duenio = st.radio("Riesgo del dueño/gerente", ["Bajo", "Medio", "Alto"], index=1)
    with col3:
        q_tipo = st.radio("Riesgo por tipo de operación", ["Bajo", "Medio", "Alto"], index=1)

    def map_risk(val): return {"Bajo": 1.0, "Medio": 0.5}.get(val, 0.0)
    idx = map_risk(q_emp) * 0.50 + map_risk(q_duenio) * 0.20 + map_risk(q_tipo) * 0.30
    score_auto = 1.0 + 4.0 * clamp(idx, 0.0, 1.0)

    colm = st.columns(2)
    with colm[0]:
        usar_manual = st.checkbox("Usar score manual")
    with colm[1]:
        score_manual = st.slider("Score manual (1–5)", 1.0, 5.0, float(round(score_auto, 2)), 0.01)
    score = score_manual if usar_manual else score_auto
    st.caption(f"Auto: **{score_auto:.2f}** · Usado: **{score:.2f}**  · Pesos: Empresa 50% · Dueño/GG 20% · Operación 30%")

    EAD = st.number_input("EAD (USD)", 0.0, 1e12, float(st.session_state.get("EAD0", 1_000_000.0)), 10_000.0, format="%.0f")
    garantia_val0 = st.number_input("Garantías (USD)", 0.0, 1e12, float(st.session_state.get("garantia_val0", 0.0)), 10_000.0, format="%.0f")
    tc_ann = st.number_input("Tasa compensatoria anual (tc)", 0.0, 2.0, 0.25, 0.005, format="%.3f")
    gastos_usd = st.number_input("Gastos (USD)", 0.0, 1e9, 0.0, 1000.0, format="%.0f")

    df_g = pd.DataFrame([{"Tipo": "Garantía", "Valor USD": float(garantia_val0)}], columns=["Tipo", "Valor USD"])

    soft_checks(tc_ann, CO_ANUAL_FIJO, garantia_val0, gastos_usd)

    res = calcular_resultados_curay(
        score, EAD, tc_ann, CO_ANUAL_FIJO, 12, 3, df_g, gastos_usd,
        w_guarantee=float(st.session_state.get("peso_total0", 1.0))
    )

    manifest = build_manifest({"page":"Q&A","cliente":str(cliente)})

    # Pequeño veredicto (opcional)
    decision_ok = (np.isfinite(res["RE_anual_simple"]) and res["RE_anual_simple"] >= float(RET_THRESHOLD))
    decision_ribbon(decision_ok, manifest["run_id"])

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("PD (12m)", fmt_pct(res['PD_12m']))
    with c2: kpi_card("Retorno (simple)", fmt_pct(res['RE_anual_simple']))
    with c3: kpi_card("LGD (política)", fmt_pct(res['LGD']))

    st.session_state.update({
        "RE_anual_simple": res["RE_anual_simple"],
        "PD_12m": res["PD_12m"], "LGD": res["LGD"],
        "EAD": EAD, "score": score, "tc_ann": tc_ann, "co_ann": CO_ANUAL_FIJO, "gastos_usd": gastos_usd,
        "garantia_val0": garantia_val0
    })

# 6.4 Inputs
elif page == "Inputs":
    st.markdown('<div class="card"><h3 style="margin:0;color:#e8eef8">📝 Inputs técnicos (para modeler) — tm fija</h3></div>', unsafe_allow_html=True)

    EAD0 = st.session_state.get("EAD0", 1_000_000.0)
    garantia_val0 = st.session_state.get("garantia_val0", 0.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        score = st.slider("Score", 1.0, 5.0, float(st.session_state.get("score", 3.0)), 0.01)
        gastos_usd = st.number_input("Gastos (USD)", 0.0, 1e9, float(st.session_state.get("gastos_usd", 0.0)), 1000.0, format="%.0f")
    with col2:
        EAD = st.number_input("EAD (USD)", 0.0, 1e12, float(EAD0), 10_000.0, format="%.0f")
        meses_venc = st.number_input("Meses t₀→t₁ (mv)", 1, 60, int(st.session_state.get("mv", 12)))
        meses_post = st.number_input("Meses t₁→t₂ (mp)", 1, 24, int(st.session_state.get("mp", 3)))
    with col3:
        tc_ann = st.number_input("tc (anual)", 0.0, 2.0, float(st.session_state.get("tc_ann", 0.25)), 0.005, format="%.3f")
        co_ann = st.number_input("CO (anual)", min_value=float(CO_ANUAL_FIJO), max_value=float(CO_ANUAL_FIJO),
                                 value=float(CO_ANUAL_FIJO), step=0.0001, format="%.4f")

    st.subheader("🛡️ Garantías")
    df_g_prefill = st.session_state.get("df_g_inicial")
    if df_g_prefill is None or df_g_prefill.empty:
        df_g_prefill = pd.DataFrame([{"Tipo": "Hipoteca predio urbano", "Valor USD": float(garantia_val0)}], columns=["Tipo","Valor USD"])

    df_g = st.data_editor(
        df_g_prefill, hide_index=True, num_rows="dynamic", key="garantias_edit",
        column_config={
            "Tipo": st.column_config.TextColumn("Tipo", help="Clase de garantía"),
            "Valor USD": st.column_config.NumberColumn("Valor USD", help="Monto asociado", step=10_000, format="$ %,d"),
        }
    )

    soft_checks(tc_ann, co_ann, garantia_val0, gastos_usd)

    res = calcular_resultados_curay(
        score, EAD, tc_ann, co_ann, meses_venc, meses_post, df_g, gastos_usd,
        w_guarantee=float(st.session_state.get("peso_total0", 1.0))
    )

    mini_manifest_ribbon(st.session_state.get("manifest", {"app_version": APP_VERSION}))
    k1, k2 = st.columns(2)
    with k1: kpi_card("PD (12m)", fmt_pct(res['PD_12m']))
    with k2: kpi_card("RE (anual, simple)", fmt_pct(res['RE_anual_simple']))

    st.session_state.update({
        "RE_anual_simple": res["RE_anual_simple"],
        "PD_12m": res["PD_12m"], "LGD": res["LGD"],
        "EAD": EAD, "score": score, "tc_ann": tc_ann, "co_ann": co_ann,
        "garantia_val0": garantia_val0, "df_g_calc": df_g.copy(), "gastos_usd": gastos_usd,
        "mv": meses_venc, "mp": meses_post
    })

# 6.5 Árbol probabilístico (neural + simulación)
elif page == "Árbol probabilístico":
    st.markdown('<div class="card"><h3 style="margin:0;color:#e8eef8">🌐 Árbol probabilístico (Neural) & proyecciones de PD</h3></div>', unsafe_allow_html=True)
with st.expander("🎛️ Parámetros (sandbox para explorar)", expanded=True):
    colA, colB, colC, colD = st.columns(4)
    with colA:
        score_loc = st.slider(
            "Score (1–5)", 1.0, 5.0,
            float(st.session_state.get("score", 3.0)),
            0.01, key="score_tree"
        )
        EAD_loc = st.number_input(
            "EAD (USD)", 0.0, 1e12,
            float(st.session_state.get("EAD", st.session_state.get("EAD0", 1_000_000.0))),
            10_000.0, format="%.0f", key="ead_tree"
        )

    with colB:
        tc_m_loc = st.number_input(
            "tc mensual (ej. 0.025 ↔ 2.50%)", 0.0, 1.0, 0.025, 0.0005,
            format="%.4f", key="tc_m_tree"
        )
        tc_ann_eff_loc = annualize_from_monthly(tc_m_loc, 12)
        st.caption(f"tc anual equivalente: {tc_ann_eff_loc:.6f}  ({tc_ann_eff_loc*100:.2f}%)")
        co_ann_loc = st.number_input(
            "CO anual (política)", 0.0, 0.50, float(CO_ANUAL_FIJO), 0.0005,
            format="%.4f", key="co_ann_tree"
        )

    with colC:
        mv_loc = st.number_input(
            "Meses t₀→t₁ (mv)", 1, 60,
            int(st.session_state.get("mv", 12)),
            key="mv_tree"
        )
        mp_loc = st.number_input(
            "Meses t₁→t₂ (mp)", 1, 24,
            int(st.session_state.get("mp", 3)),
            key="mp_tree"
        )

    with colD:
        garantia_loc = st.number_input(
            "Garantías (USD)", 0.0, 1e12,
            float(st.session_state.get("garantia_val0", 0.0)),
            10_000.0, format="%.0f", key="garantias_tree"
        )
        gastos_loc = st.number_input(
            "Gastos (USD)", 0.0, 1e9,
            float(st.session_state.get("gastos_usd", 0.0)),
            1000.0, format="%.0f", key="gastos_tree"
        )

    tweak_cols = st.columns(5)
    with tweak_cols[0]:
        vis_mode = st.radio("Escalar", ["Probabilidad", "Aporte EV"],
                            horizontal=True, key="vis_mode_tree")
    with tweak_cols[1]:
        scale_strength = st.slider("Intensidad del grosor", 0.5, 2.0, 1.0, 0.1,
                                   key="scale_strength_tree")
    with tweak_cols[2]:
        show_labels = st.checkbox("Etiquetas en aristas", value=True,
                                  key="show_labels_tree")
    with tweak_cols[3]:
        show_contrib = st.checkbox("Tabla de aportes EV", value=True,
                                   key="show_contrib_tree")
    with tweak_cols[4]:
        sync_state = st.checkbox("Sincronizar con estado global", value=False,
                                 key="sync_state_tree")

    tweak_cols2 = st.columns(2)
    with tweak_cols2[0]:
        show_time_titles_nn = st.checkbox("Mostrar títulos T = 0/1/2",
                                          value=True, key="show_time_titles_nn")
    with tweak_cols2[1]:
        time_title_style = st.radio("Ubicación de títulos",
                                    ["node", "column", "both"],
                                    index=0, horizontal=True,
                                    key="time_title_style_tree")

    highlight_branch = st.selectbox(
        "Resaltar rama/nodo",
        ["(ninguno)", "t₁: paga", "t₂: cura", "t₂: default final"],
        index=0, key="highlight_tree"
    )
    hi_map = {"(ninguno)": None, "t₁: paga": "pay1", "t₂: cura": "cura2", "t₂: default final": "def2"}
    hi_key = hi_map.get(highlight_branch)


# --- HMM controls (UI) ---
with st.expander("🧩 HMM (3 estados) — usar para PD₁ y PD₂|₁", expanded=False):
    st.checkbox("Activar HMM", value=False, key="use_hmm_nn")
    st.checkbox("α (P→S1) desde λ (hazard mensual)", value=True, key="auto_alpha_nn")
    st.number_input(
        "α mensual (si no auto)", 0.0, 1.0, 0.05, 0.001,
        disabled=st.session_state.get("auto_alpha_nn", True),
        key="alpha_manual_nn"
    )
    st.number_input("γ mensual: S1→P (cura)", 0.0, 1.0, 0.40, 0.01, key="gamma_nn")
    st.number_input("δ mensual: S1→S2 (default final)", 0.0, 1.0, 0.60, 0.01, key="delta_nn")
    st.checkbox(
        "Resolver binario en t₂ (sin quedarse en S1)",
        value=True,
        help="Si ON, γ+δ=1 ⇒ a t₂ solo hay cura o default final.",
        key="force_res_nn"
    )

# --- Right before you use it ---
use_hmm   = bool(st.session_state.get("use_hmm_nn", False))
auto_alpha = bool(st.session_state.get("auto_alpha_nn", True))
alpha_manual = float(st.session_state.get("alpha_manual_nn", 0.05))
gamma = float(st.session_state.get("gamma_nn", 0.40))
delta = float(st.session_state.get("delta_nn", 0.60))
force_res = bool(st.session_state.get("force_res_nn", True))

# Fallbacks (in case these were defined earlier in the page; these keep it crash-proof)
score_loc = float(st.session_state.get("score", 3.0))
EAD_loc   = float(st.session_state.get("EAD", st.session_state.get("EAD0", 1_000_000.0)))
tc_ann_eff_loc = float(st.session_state.get("tc_ann", 0.25))   # use your computed annual tc if you have one
co_ann_loc = float(st.session_state.get("co_ann", 0.015))
gastos_loc = float(st.session_state.get("gastos_usd", 0.0))
mv_loc  = int(st.session_state.get("mv", 12))
mp_loc  = int(st.session_state.get("mp", 3))

# Garantías (asegura df_g_loc ANTES de calcular)
garantia_loc = float(st.session_state.get("garantia_val0", 0.0))
df_g_loc = st.session_state.get("df_g_calc", st.session_state.get("df_g_inicial"))
if df_g_loc is None or df_g_loc.empty:
    df_g_loc = pd.DataFrame(
        [{"Tipo": "Garantía", "Valor USD": float(garantia_loc)}],
        columns=["Tipo", "Valor USD"]
    )
else:
    total = float(pd.to_numeric(df_g_loc["Valor USD"], errors="coerce").fillna(0.0).sum())
    if total > 0 and abs(total - garantia_loc) > 1e-6:
        df_g_loc = df_g_loc.copy()
        df_g_loc["Valor USD"] = df_g_loc["Valor USD"] * (garantia_loc / total)

# HMM config → cálculo
hmm_cfg = dict(
    use=use_hmm,
    alpha=None if auto_alpha else float(alpha_manual),
    gamma=float(gamma),
    delta=float(delta),
    beta=0.0,
    force_resolution_at_t2=bool(force_res)
)

# (Re)calcular resultados (con o sin HMM)
res_loc = calcular_resultados_curay(
    score_loc, EAD_loc, tc_ann_eff_loc, co_ann_loc, mv_loc, mp_loc, df_g_loc, gastos_loc,
    w_guarantee=float(st.session_state.get("peso_total0", 1.0)),
    hmm=hmm_cfg
)

st.session_state["res_loc"] = res_loc
# λ desde tu motor; si no existe, usa 0.10
lam_val = float(st.session_state.get("res_loc", {}).get("lam", 0.10))
# === Árbol probabilístico (dibujar) ===
st.markdown("#### 🌳 Árbol probabilístico")

# Fallbacks seguros (evitan NameError si el usuario reordena secciones)
vis_mode            = st.session_state.get("vis_mode", "Probabilidad")
scale_strength      = float(st.session_state.get("scale_strength_tree", 1.0))
show_labels         = bool(st.session_state.get("show_labels_tree", True))
show_time_titles_nn = bool(st.session_state.get("show_time_titles_nn", True))
time_title_style    = st.session_state.get("time_title_style", "node")
hi_key              = st.session_state.get("hi_key", None)
show_contrib = bool(st.session_state.get("show_contrib_tree", True))


st.markdown("#### 🌳 Árbol probabilístico")
plot_neural_probability_tree(
    res=res_loc,
    mv=int(st.session_state.get("mv", mv_loc)),
    mp=int(st.session_state.get("mp", mp_loc)),
    mode=vis_mode,                      # "Probabilidad" | "Aporte EV"
    show_labels=show_labels,
    scale_strength=scale_strength,
    show_time_titles=show_time_titles_nn,
    time_title_style=time_title_style,  # "node" | "column" | "both"
    highlight=hi_key,                   # None | "pay1" | "cura2" | "def2"
    palette=None
)
st.divider()


# Tabla de aportes EV (si activaste el checkbox)
if show_contrib:
    show_ev_contributions(res_loc)


# Marcas para saber si pasamos por aquí (debug rápido)
st.markdown("### ✅ Sección Poisson (debug)")
st.caption(f"λ leído = {lam_val:.4f}")

with st.expander("📈 PD(12m) — rectángulos bajo la curva (Poisson)", expanded=True):
    months = st.slider("Horizonte (meses)", 12, 60, 12, 1, key="h_pd_curve_main")
    esquema = st.selectbox("Esquema de Riemann", ["left", "mid", "right"], index=0, key="riemann_main")
    show_labels = st.checkbox("Mostrar ΔPD_m por mes", value=False, key="labels_main")

    # 🔔 AQUÍ LLAMAS A TU FUNCIÓN
    plot_pd12_accum_area(
        lam=lam_val,
        months=months,
        riemann=esquema,
        show_delta_labels=show_labels,
        show_integral=True,
    )

    st.latex(r"\mathrm{PD}(12\text{m})=\int_{0}^{12}\frac{\lambda}{12}e^{-\lambda u/12}\,du=1-e^{-\lambda}")



with st.expander("📋 Detalle mensual (PD, supervivencia, ΔPD)", expanded=False):
    # 🔔 Y AQUÍ LA TABLA
    show_pd_monthly_table(lam=lam_val, months=12, riemann="left")

# HMM caption + visualización (solo si está activado y presente)
hmm_meta = res_loc.get("HMM")
if use_hmm and hmm_meta:
    st.caption(
        f"HMM → α={hmm_meta['alpha']:.4f}, γ={hmm_meta['gamma']:.2f}, "
        f"δ={hmm_meta['delta']:.2f} · PD₁={res_loc.get('PD1', 0.0):.2%} · PD₂|₁={res_loc.get('PD2_cond', 0.0):.2%}"
    )
    fig = plot_hmm_3state_graph(hmm_meta, mv_loc, mp_loc)
    st.plotly_chart(fig, use_container_width=True)
    hmm_formula_block(hmm_meta, mv_loc, mp_loc)
else:
    st.caption(
        f"PD₁={float(res_loc.get('PD1', 0.0)):.2%} · PD₂|₁={float(res_loc.get('PD2_cond', 0.0)):.2%}"
    )

# ================== Policy Lab (RL beta) — Hybrid Neural-HMM ==================
from types import SimpleNamespace

# ==== Easy Guide / Guía fácil — on-demand summarizer (ES/EN) ====

# Reuse language if you already defined it; otherwise default to ES
_lang_val = st.session_state.get("expl_lang_math", "Español")

# Buttons to show/hide the summary (separate from the math explainer)
cS1, cS2 = st.columns([1,1])
with cS1:
    _show_sum = st.button("📘 Summarize / Resumir", key="btn_show_math_summary")
with cS2:
    _hide_sum = st.button("🙈 Hide / Ocultar", key="btn_hide_math_summary")

if _show_sum:
    st.session_state["show_math_summary"] = True
if _hide_sum:
    st.session_state["show_math_summary"] = False

# ---------- Renderers ----------
def render_es_summary():
    st.markdown("### 🧩 Guía fácil: Neural-HMM + EV")
    st.markdown("Explicación paso a paso de todas las fórmulas y símbolos usados en el objetivo.")

    st.markdown("#### 1) Rasgos (contexto + acción)")
    st.latex(r"z=\phi(x,a)\in\mathbb{R}^d")
    st.markdown(
        "- **\(z\)**: vector de características que resume al cliente (contexto **x**) y la acción de términos **a**.\n"
        "- **\(\\phi(x,a)\)**: función que construye esas características.\n"
        "- Componentes típicos: sesgo \(1\), **score**, **\\(\\ln(EAD)\\)** (exposición), **\\(\\ln(G)\\)** (garantía), **CO** (costo), **score²**, "
        "**\\(tc_{ann}\\)** (precio anual), **\\(mv/12\\)** y **\\(mp/12\\)** (meses a los nodos), **\\(w_G\\)** (peso de garantía)."
    )

    st.markdown("#### 2) Transiciones mensuales (Neural)")
    st.latex(r"h=\rho(W_1 z+b_1),\quad \rho(u)=\max(u,0)")
    st.markdown(
        "- **\(h\)**: representación no lineal del caso. **ReLU** deja pasar valores positivos, pone 0 a negativos.\n"
        "- Con **h**, cada fila de la matriz de transición mensual **\(A(z)\)** se obtiene con *softmax* (probabilidades que suman 1):"
    )
    st.latex(r"A_{P,\cdot}(z)=[\tilde p_1,\tilde p_2,\tilde p_3],\quad A_{S1,\cdot}(z)=[\tilde q_1,\tilde q_2,\tilde q_3],\quad A_{S2,\cdot}(z)=[0,0,1]")
    st.markdown(
        "- Estados: **P** (vigente), **S1** (estrés), **S2** (default final, absorbente: no se sale de S2).\n"
        "- Variante alternativa (no estricta): parámetros sigmoides \(\alpha,\beta,\gamma,\delta\) con ‘clamp’ para casi-estocástica."
    )

    st.markdown("#### 3) Probabilidades de ramas del árbol")
    st.latex(r"\pi_{t_1}(z)=\pi_0 A(z)^{mv},\quad \pi_0=[1,0,0]")
    st.markdown(
        "- **\(mv\)**: meses desde hoy hasta el primer nodo \(t_1\). **\(A^{mv}\)** compone \(mv\) pasos mensuales.\n"
        "- **\(\pi_{t_1}\)**: distribución de estados al llegar a \(t_1\)."
    )
    st.latex(r"PD_1(z)=\pi_{t_1}(z)[S1]")
    st.markdown("- **\(PD_1\)**: prob. de estar en **S1** en \(t_1\) (o incluir **S2** si consideras salto directo a default).")

    st.latex(r"q_{mp}(z)=e_{S1}^\top A(z)^{mp}")
    st.markdown(
        "- **\(mp\)**: meses entre \(t_1\) y \(t_2\). **\(q_{mp}\)**: distribución tras \(mp\) meses empezando en S1."
    )
    st.latex(r"""
PD_{2\mid 1}(z)=
\begin{cases}
q_{mp}(z)[S2], & \text{si S1 se fuerza a resolverse en } t_2,\\
\dfrac{q_{mp}(z)[S2]}{q_{mp}(z)[P]+q_{mp}(z)[S2]}, & \text{en otro caso.}
\end{cases}
""")
    st.markdown(
        "- **\(PD_{2|1}\)**: prob. condicional de caer en default en el segundo tramo dado que llegaste a S1."
    )
    st.latex(r"p_{pay1}=1-PD_1,\quad p_{cura}=PD_1(1-PD_{2\mid1}),\quad p_{def2}=PD_1\,PD_{2\mid1}")
    st.markdown(
        "- **Ramas**: *paga en \(t_1\)*, *cura a \(t_2\)*, *default final en \(t_2\)*."
    )
    st.latex(r"PD_{12m}^{HMM}(z)=(\pi_0 A(z)^{12})[S2]")
    st.markdown("- **PD 12m HMM**: prob. de default a 12 meses consistente con la cadena.")

    st.markdown("#### 4) Flujos y VP")
    st.latex(r"i_m=(1+tc_{ann})^{1/12}-1,\quad c_m=(1+co_{ann})^{1/12}-1,\quad m_m=(1+tm_{ann,fijo})^{1/12}-1")
    st.markdown("- **\(i_m,c_m,m_m\)**: tasas **mensuales** de precio, costo y descuento.")
    st.latex(r"PV_{t_1}=\dfrac{EAD(1+i_m)^{mv}}{(1+c_m)^{mv}}")
    st.latex(r"PV_{t_2}^{\mathrm{cura}}=\dfrac{EAD(1+i_m)^{mv+mp}(1+m_m)^{mp}}{(1+c_m)^{mv+mp}}")
    st.latex(r"PV_{t_2}^{\mathrm{def}}=w_G\,G_{\mathrm{total}}")
    st.latex(r"EV_{VP}=p_{pay1}PV_{t_1}+p_{cura}PV_{t_2}^{\mathrm{cura}}+p_{def2}PV_{t_2}^{\mathrm{def}}")
    st.latex(r"T_{exp}=mv+PD_1\,mp")
    st.latex(r"ret_{VP,ann}=\left(\dfrac{EV_{VP}}{EAD}\right)^{\frac{12}{T_{exp}}}-1,\quad ev_{mult}=\dfrac{EV_{VP}}{EAD}-1")
    st.markdown(
        "- **\(EV_{VP}\)**: valor esperado en VP mezclando ramas con sus probabilidades.\n"
        "- **\(T_{exp}\)**: meses ‘expuestos’ promedio (más largo si entras a S1).\n"
        "- **ret** / **ev_mult**: métricas de recompensa para el bandit."
    )

    st.markdown("#### 5) Bandit (LinUCB)")
    st.latex(r"\theta_a=A_a^{-1}b_a,\quad \hat\mu_a(x)=\theta_a^\top x,\quad UCB_a=\hat\mu_a+\alpha\,x^\top A_a^{-1}x")
    st.latex(r"A_a\leftarrow A_a+xx^\top,\quad b_a\leftarrow b_a+r(a)\,x")
    st.markdown(
        "- Mantiene una regresión por acción (**\(A_a,b_a\)**). Selecciona por **optimismo** (UCB). Actualiza con el reward observado."
    )

    st.markdown("#### 6) (Opc.) Verosimilitud para entrenar la parte HMM")
    st.latex(r"""
\log L_i(\Theta)=
\begin{cases}
\log(1-PD_1(z_i)), & y_i=\mathrm{pay1},\\
\log(PD_1(z_i)(1-PD_{2\mid1}(z_i))), & y_i=\mathrm{cura},\\
\log(PD_1(z_i)PD_{2\mid1}(z_i)), & y_i=\mathrm{def2}.
\end{cases}
""")
    st.markdown(
        "- Sumas en \(i\) y regularizas. Con trayectorias mensuales completas, usa **forward–backward** y **EM** (variacional) para ajustar las cabeceras."
    )

def render_en_summary():
    st.markdown("### 🧩 Easy Guide: Neural-HMM + EV")
    st.markdown("Plain-language walkthrough of all formulas and symbols in the objective.")

    st.markdown("#### 1) Features (context + action)")
    st.latex(r"z=\phi(x,a)\in\mathbb{R}^d")
    st.markdown(
        "- **\(z\)**: feature vector summarizing borrower context **x** and terms/action **a**.\n"
        "- **\(\\phi(x,a)\)**: mapping that builds those features.\n"
        "- Typical components: bias \(1\), **score**, **\\(\\ln(EAD)\\)**, **\\(\\ln(G)\\)**, **CO**, **score²**, "
        "**\\(tc_{ann}\\)**, **\\(mv/12\\)**, **\\(mp/12\\)**, **\\(w_G\\)**."
    )

    st.markdown("#### 2) Monthly transitions (Neural)")
    st.latex(r"h=\rho(W_1 z+b_1),\quad \rho(u)=\max(u,0)")
    st.markdown(
        "- **\(h\)**: non-linear representation via **ReLU**.\n"
        "- Each row of the monthly transition matrix **\(A(z)\)** is a **softmax** (probabilities sum to 1):"
    )
    st.latex(r"A_{P,\cdot}(z)=[\tilde p_1,\tilde p_2,\tilde p_3],\quad A_{S1,\cdot}(z)=[\tilde q_1,\tilde q_2,\tilde q_3],\quad A_{S2,\cdot}(z)=[0,0,1]")
    st.markdown(
        "- States: **P** (performing), **S1** (distressed), **S2** (final default; absorbing).\n"
        "- Alternative (non-strict) variant: sigmoid heads \(\alpha,\beta,\gamma,\delta\) with a clamp."
    )

    st.markdown("#### 3) Branch probabilities")
    st.latex(r"\pi_{t_1}(z)=\pi_0 A(z)^{mv},\quad \pi_0=[1,0,0]")
    st.markdown(
        "- **\(mv\)**: months to the first decision point \(t_1\). **\(A^{mv}\)** composes monthly steps.\n"
        "- **\(\pi_{t_1}\)**: state distribution at \(t_1\)."
    )
    st.latex(r"PD_1(z)=\pi_{t_1}(z)[S1]")
    st.markdown("- **\(PD_1\)**: prob. of being in **S1** at \(t_1\) (optionally include direct S2 if you model P→S2 jumps).")

    st.latex(r"q_{mp}(z)=e_{S1}^\top A(z)^{mp}")
    st.markdown("- **\(mp\)**: months from \(t_1\) to \(t_2\). **\(q_{mp}\)**: distribution after \(mp\) months starting in S1.")
    st.latex(r"""
PD_{2\mid 1}(z)=
\begin{cases}
q_{mp}(z)[S2], & \text{if S1 is forced to resolve by } t_2,\\
\dfrac{q_{mp}(z)[S2]}{q_{mp}(z)[P]+q_{mp}(z)[S2]}, & \text{otherwise.}
\end{cases}
""")
    st.markdown("- **\(PD_{2|1}\)**: conditional second-leg default probability given you were in S1 at \(t_1\).")
    st.latex(r"p_{pay1}=1-PD_1,\quad p_{cura}=PD_1(1-PD_{2\mid1}),\quad p_{def2}=PD_1\,PD_{2\mid1}")
    st.markdown("- **Branches**: pay at \(t_1\), cure by \(t_2\), or final default at \(t_2\).")
    st.latex(r"PD_{12m}^{HMM}(z)=(\pi_0 A(z)^{12})[S2]")
    st.markdown("- **12-month PD (HMM)**: default probability at 12 months consistent with the chain.")

    st.markdown("#### 4) Cashflows & PV")
    st.latex(r"i_m=(1+tc_{ann})^{1/12}-1,\quad c_m=(1+co_{ann})^{1/12}-1,\quad m_m=(1+tm_{ann,fijo})^{1/12}-1")
    st.markdown("- **\(i_m,c_m,m_m\)**: monthly price, cost, and discount rates.")
    st.latex(r"PV_{t_1}=\dfrac{EAD(1+i_m)^{mv}}{(1+c_m)^{mv}}")
    st.latex(r"PV_{t_2}^{\mathrm{cura}}=\dfrac{EAD(1+i_m)^{mv+mp}(1+m_m)^{mp}}{(1+c_m)^{mv+mp}}")
    st.latex(r"PV_{t_2}^{\mathrm{def}}=w_G\,G_{\mathrm{total}}")
    st.latex(r"EV_{VP}=p_{pay1}PV_{t_1}+p_{cura}PV_{t_2}^{\mathrm{cura}}+p_{def2}PV_{t_2}^{\mathrm{def}}")
    st.latex(r"T_{exp}=mv+PD_1\,mp")
    st.latex(r"ret_{VP,ann}=\left(\dfrac{EV_{VP}}{EAD}\right)^{\frac{12}{T_{exp}}}-1,\quad ev_{mult}=\dfrac{EV_{VP}}{EAD}-1")
    st.markdown(
        "- **\(EV_{VP}\)**: expected present value over branches.\n"
        "- **\(T_{exp}\)**: average time at risk; longer if S1 likely.\n"
        "- **ret** / **ev_mult**: reward choices for the bandit."
    )

    st.markdown("#### 5) Bandit (LinUCB)")
    st.latex(r"\theta_a=A_a^{-1}b_a,\quad \hat\mu_a(x)=\theta_a^\top x,\quad UCB_a=\hat\mu_a+\alpha\,x^\top A_a^{-1}x")
    st.latex(r"A_a\leftarrow A_a+xx^\top,\quad b_a\leftarrow b_a+r(a)\,x")
    st.markdown("- Per-action regression state (**\(A_a,b_a\)**), optimistic selection via UCB, then update with observed reward.")

    st.markdown("#### 6) (Opt.) Likelihood to train the HMM part")
    st.latex(r"""
\log L_i(\Theta)=
\begin{cases}
\log(1-PD_1(z_i)), & y_i=\mathrm{pay1},\\
\log(PD_1(z_i)(1-PD_{2\mid1}(z_i))), & y_i=\mathrm{cura},\\
\log(PD_1(z_i)PD_{2\mid1}(z_i)), & y_i=\mathrm{def2}.
\end{cases}
""")
    st.markdown("- Sum over \(i\), add regularization; with monthly paths use **forward–backward** and (variational) **EM** on the heads.")

# ---------- Show the summary in its own (non-nested) expander ----------
if st.session_state.get("show_math_summary", False):
    with st.expander("🧩 Easy Guide / Guía fácil — Neural-HMM + EV", expanded=True):
        if _lang_val == "Español":
            render_es_summary()
        else:
            render_en_summary()

# ==== Pretty math explainer: Neural-HMM + EV (ES/EN) ====

# 1) Slightly larger KaTeX (textbook feel)
st.markdown("""
<style>
.katex-display { font-size: 1.18em; }      /* bigger display math */
.katex .base { line-height: 1.25; }        /* tighter lines */
</style>
""", unsafe_allow_html=True)

# 2) Language + toggle
colE1, colE2, colE3 = st.columns([2,1,1])
with colE1:
    _lang_expl = st.selectbox("📘 Explicar / Explain:", ["Español","English"], key="expl_lang_math")
with colE2:
    _show_math = st.button("Mostrar / Show", key="btn_show_math")
with colE3:
    _hide_math = st.button("Ocultar / Hide", key="btn_hide_math")

if _show_math: st.session_state["show_math_explainer"] = True
if _hide_math: st.session_state["show_math_explainer"] = False

def render_es():
    st.markdown("### 📘 Objetivo Neural-HMM + EV (compacto y autocontenido)")
    st.markdown("A continuación va la formulación **Neural-HMM + EV** que puntúa una acción \\(a\\) bajo un contexto \\(x\\).")

    st.markdown("#### 1) Rasgos (contexto + acción)")
    st.latex(r"z=\phi(x,a)\in\mathbb{R}^d")
    st.latex(r"\phi(x,a)=[\,1,\; score,\; \ln(EAD),\; \ln(G),\; CO,\; score^2,\; tc_{ann},\; \tfrac{mv}{12},\; \tfrac{mp}{12},\; w_G\,]")

    st.markdown("#### 2) Parametrización neuronal de **transiciones mensuales**")
    st.markdown("Cadena de 3 estados \\(S=\\{P,S1,S2\\}\\) (vigente, en estrés, default final).")
    st.latex(r"h=\rho(W_1 z+b_1)\in\mathbb{R}^H,\qquad \rho(u)=\max(u,0)")
    st.markdown("**Cabeceras → probabilidades por fila (softmax)**")
    st.latex(r"\tilde p=\mathrm{softmax}(W_P h+b_P),\qquad A_{P,\cdot}(z)=[\tilde p_1,\tilde p_2,\tilde p_3]")
    st.latex(r"\tilde q=\mathrm{softmax}(W_{S1} h+b_{S1}),\qquad A_{S1,\cdot}(z)=[\tilde q_1,\tilde q_2,\tilde q_3]")
    st.latex(r"A_{S2,\cdot}(z)=[0,0,1]")
    st.caption("Esta app usa la parametrización estricta (softmax) por filas para P y S1; S2 es absorbente.")
    st.markdown("*(Variante “sigmoid+clamp”)*")
    st.latex(r"""
A(z)=
\begin{bmatrix}
1-\alpha-\beta & \alpha & \beta\\
\gamma & 1-\gamma-\delta & \delta\\
0 & 0 & 1
\end{bmatrix},
\qquad \alpha,\beta,\gamma,\delta\in(0,1)
""")

    st.markdown("#### 3) De transiciones mensuales a probabilidades de ramas")
    st.markdown("\\(mv\\) = meses hasta \\(t_1\\), \\(mp\\) = meses desde \\(t_1\\) hasta \\(t_2\\).  \\(\\pi_0=[1,0,0]\\).")
    st.latex(r"\pi_{t_1}(z)=\pi_0\,A(z)^{mv},\qquad PD_1(z)=\pi_{t_1}(z)[S1]")
    st.latex(r"q_{mp}(z)=e_{S1}^\top A(z)^{mp}")
    st.latex(r"""
PD_{2\mid 1}(z)=
\begin{cases}
q_{mp}(z)[S2], & \text{si S1 se fuerza a resolverse en } t_2,\\
\dfrac{q_{mp}(z)[S2]}{q_{mp}(z)[P]+q_{mp}(z)[S2]}, & \text{en otro caso.}
\end{cases}
""")
    st.caption("Si fuerzas que S1 se resuelva a t₂, entonces \(PD_{2\mid 1}=q_{mp}(z)[S2]\).")
    st.latex(r"p_{pay1}=1-PD_1,\qquad p_{cura}=PD_1(1-PD_{2\mid1}),\qquad p_{def2}=PD_1\,PD_{2\mid1}")
    st.latex(r"PD_{12m}^{HMM}(z)=(\pi_0A(z)^{12})[S2],\qquad \lambda=-\ln\!\big(1-PD_{12m}^{HMM}\big)")

    st.markdown("#### 4) Flujos y valores presentes")
    st.latex(r"i_m=(1+tc_{ann})^{1/12}-1,\qquad c_m=(1+co_{ann})^{1/12}-1,\qquad m_m=(1+tm_{ann,fijo})^{1/12}-1")
    st.latex(r"PV_{t_1}=\dfrac{EAD\,(1+i_m)^{mv}}{(1+c_m)^{mv}}")
    st.latex(r"PV_{t_2}^{\mathrm{cura}}=\dfrac{EAD\,(1+i_m)^{mv+mp}(1+m_m)^{mp}}{(1+c_m)^{mv+mp}}")
    st.latex(r"PV_{t_2}^{\mathrm{def}}=w_G\cdot G_{\mathrm{total}}")
    st.latex(r"EV_{VP}(z)=p_{pay1}\,PV_{t_1}+p_{cura}\,PV_{t_2}^{\mathrm{cura}}+p_{def2}\,PV_{t_2}^{\mathrm{def}}")
    st.latex(r"T_{exp}=mv+PD_1\,mp")
    st.latex(r"ret_{VP,ann}(z)=\left(\dfrac{EV_{VP}(z)}{EAD}\right)^{\frac{12}{T_{exp}}}-1")
    st.latex(r"ev_{mult}(z)=\dfrac{EV_{VP}(z)}{EAD}-1")

    st.markdown("#### 5) Objetivo bandit (LinUCB)")
    st.latex(r"\theta_a=A_a^{-1}b_a,\qquad \hat\mu_a(x)=\theta_a^\top x")
    st.latex(r"UCB_a(x)=\hat\mu_a(x)+\alpha\,x^\top A_a^{-1}x")
    st.latex(r"A_a\leftarrow A_a+xx^\top,\qquad b_a\leftarrow b_a+r(a)\,x")

    st.markdown("#### 6) (Opcional) Verosimilitud para entrenar el Neural-HMM")
    st.latex(r"""
\log L_i(\Theta)=
\begin{cases}
\log\!\big(1-PD_1(z_i)\big), & y_i=\mathrm{pay1},\\
\log\!\big(PD_1(z_i)\,(1-PD_{2\mid1}(z_i))\big), & y_i=\mathrm{cura},\\
\log\!\big(PD_1(z_i)\,PD_{2\mid1}(z_i)\big), & y_i=\mathrm{def2}.
\end{cases}
""")
    st.caption("Sumar en \(i\) + regularización; con trayectorias mensuales, usar forward–backward y EM (variacional) en las cabeceras.")

def render_en():
    st.markdown("### 📘 Neural-HMM + EV objective (compact & self-contained)")
    st.markdown("Below is the **Neural-HMM + EV** formulation that scores an action \\(a\\) under a context \\(x\\).")

    st.markdown("#### 1) Features (context + action)")
    st.latex(r"z=\phi(x,a)\in\mathbb{R}^d")
    st.latex(r"\phi(x,a)=[\,1,\; score,\; \ln(EAD),\; \ln(G),\; CO,\; score^2,\; tc_{ann},\; \tfrac{mv}{12},\; \tfrac{mp}{12},\; w_G\,]")

    st.markdown("#### 2) Neural parameterization of **monthly** transitions")
    st.markdown("Three states \(S=\{P,S1,S2\}\) (performing, distressed, final default).")
    st.latex(r"h=\rho(W_1 z+b_1)\in\mathbb{R}^H,\qquad \rho(u)=\max(u,0)")
    st.markdown("**Heads → row probabilities (softmax)**")
    st.latex(r"\tilde p=\mathrm{softmax}(W_P h+b_P),\qquad A_{P,\cdot}(z)=[\tilde p_1,\tilde p_2,\tilde p_3]")
    st.latex(r"\tilde q=\mathrm{softmax}(W_{S1} h+b_{S1}),\qquad A_{S1,\cdot}(z)=[\tilde q_1,\tilde q_2,\tilde q_3]")
    st.latex(r"A_{S2,\cdot}(z)=[0,0,1]")
    st.caption("This app uses the strict row-stochastic (softmax) parameterization for P and S1; S2 is absorbing.")
    st.markdown("*(Alternative “sigmoid+clamp”)*")
    st.latex(r"""
A(z)=
\begin{bmatrix}
1-\alpha-\beta & \alpha & \beta\\
\gamma & 1-\gamma-\delta & \delta\\
0 & 0 & 1
\end{bmatrix},
\qquad \alpha,\beta,\gamma,\delta\in(0,1)
""")

    st.markdown("#### 3) From monthly transitions to path probabilities")
    st.markdown("\\(mv\\) = months to \\(t_1\\), \\(mp\\) = months from \\(t_1\\) to \\(t_2\\).  \\(\\pi_0=[1,0,0]\\).")
    st.latex(r"\pi_{t_1}(z)=\pi_0\,A(z)^{mv},\qquad PD_1(z)=\pi_{t_1}(z)[S1]")
    st.latex(r"q_{mp}(z)=e_{S1}^\top A(z)^{mp}")
    st.latex(r"""
PD_{2\mid 1}(z)=
\begin{cases}
q_{mp}(z)[S2], & \text{if S1 is forced to resolve by } t_2,\\
\dfrac{q_{mp}(z)[S2]}{q_{mp}(z)[P]+q_{mp}(z)[S2]}, & \text{otherwise.}
\end{cases}
""")
    st.caption("If you force S1 to resolve by \(t_2\), then \(PD_{2\mid 1}=q_{mp}(z)[S2]\).")
    st.latex(r"p_{pay1}=1-PD_1,\qquad p_{cura}=PD_1(1-PD_{2\mid1}),\qquad p_{def2}=PD_1\,PD_{2\mid1}")
    st.latex(r"PD_{12m}^{HMM}(z)=(\pi_0A(z)^{12})[S2],\qquad \lambda=-\ln\!\big(1-PD_{12m}^{HMM}\big)")

    st.markdown("#### 4) Cashflows and present values")
    st.latex(r"i_m=(1+tc_{ann})^{1/12}-1,\qquad c_m=(1+co_{ann})^{1/12}-1,\qquad m_m=(1+tm_{ann,fijo})^{1/12}-1")
    st.latex(r"PV_{t_1}=\dfrac{EAD\,(1+i_m)^{mv}}{(1+c_m)^{mv}}")
    st.latex(r"PV_{t_2}^{\mathrm{cura}}=\dfrac{EAD\,(1+i_m)^{mv+mp}(1+m_m)^{mp}}{(1+c_m)^{mv+mp}}")
    st.latex(r"PV_{t_2}^{\mathrm{def}}=w_G\cdot G_{\mathrm{total}}")
    st.latex(r"EV_{VP}(z)=p_{pay1}\,PV_{t_1}+p_{cura}\,PV_{t_2}^{\mathrm{cura}}+p_{def2}\,PV_{t_2}^{\mathrm{def}}")
    st.latex(r"T_{exp}=mv+PD_1\,mp")
    st.latex(r"ret_{VP,ann}(z)=\left(\dfrac{EV_{VP}(z)}{EAD}\right)^{\frac{12}{T_{exp}}}-1")
    st.latex(r"ev_{mult}(z)=\dfrac{EV_{VP}(z)}{EAD}-1")

    st.markdown("#### 5) Bandit objective (LinUCB)")
    st.latex(r"\theta_a=A_a^{-1}b_a,\qquad \hat\mu_a(x)=\theta_a^\top x")
    st.latex(r"UCB_a(x)=\hat\mu_a(x)+\alpha\,x^\top A_a^{-1}x")
    st.latex(r"A_a\leftarrow A_a+xx^\top,\qquad b_a\leftarrow b_a+r(a)\,x")

    st.markdown("#### 6) (Optional) Likelihood to train the Neural-HMM")
    st.latex(r"""
\log L_i(\Theta)=
\begin{cases}
\log\!\big(1-PD_1(z_i)\big), & y_i=\mathrm{pay1},\\
\log\!\big(PD_1(z_i)\,(1-PD_{2\mid1}(z_i))\big), & y_i=\mathrm{cura},\\
\log\!\big(PD_1(z_i)\,PD_{2\mid1}(z_i)\big), & y_i=\mathrm{def2}.
\end{cases}
""")
    st.caption("Sum over \(i\) + regularization; with monthly paths, use forward–backward and (variational) EM on the heads.")

if st.session_state.get("show_math_explainer", False):
    with st.expander("📘 Neural-HMM + EV — explicación / explainer", expanded=True):
        if st.session_state.get("expl_lang_math","Español") == "Español":
            render_es()
        else:
            render_en()

# (Optional) Example to match the screenshot style exactly:
# st.latex(r"\nu = \dfrac{2\pi}{\lambda}\,d\,\sqrt{n^2-n_0^2}")


if page == "Policy Lab (RL beta)":
    st.markdown('<div class="card"><h3 style="margin:0;color:#e8eef8">🧭 Policy Lab (RL beta) — Aprendizaje de política (Neural-HMM)</h3></div>', unsafe_allow_html=True)
    st.caption("Optimiza precio (tc), tenor (mv/mp) y requisito de garantía (w_G) usando bandits (LinUCB) y un motor de PD híbrido: cadena de Markov mensual con transiciones parametrizadas por una MLP.")

    # ---------- Context from current session ----------
    ctx = {
        "score": float(st.session_state.get("score", 3.0)),
        "EAD": float(st.session_state.get("EAD", st.session_state.get("EAD0", 1_000_000.0))),
        "co_ann": float(st.session_state.get("co_ann", CO_ANUAL_FIJO)),
        "gastos_usd": float(st.session_state.get("gastos_usd", 0.0)),
        "garantia_val0": float(st.session_state.get("garantia_val0", 0.0)),
        "df_g": st.session_state.get("df_g_calc", st.session_state.get("df_g_inicial")),
    }
    if ctx["df_g"] is None or getattr(ctx["df_g"], "empty", True):
        ctx["df_g"] = pd.DataFrame([{"Tipo":"Garantía", "Valor USD": ctx["garantia_val0"]}])

    # ---------- Controls ----------
    c1, c2, c3 = st.columns(3)
    with c1:
        reward_metric = st.selectbox(
            "Métrica objetivo", ["ret_ann_vp", "ev_mult"], index=0,
            help="ret_ann_vp = retorno anualizado (VP). ev_mult = (EV/EAD) - 1."
        )
        alpha = st.slider("Exploración (α, LinUCB)", 0.1, 3.0, 1.0, 0.1)
    with c2:
        steps = st.number_input("Pasos de entrenamiento", 50, 2000, 300, 50)
        seed = st.number_input("Semilla", 1, 10_000, 42, 1)
    with c3:
        use_forced_resolution = st.checkbox("S1 se resuelve a t₂ (PD₂|₁ = q[S2])", value=True)
        st.caption("Si desmarcas, PD₂|₁ = q[S2] / (q[P]+q[S2]).")

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        tc_min = st.number_input("tc mín (anual)", 0.05, 0.80, float(st.session_state.get("tc_ann", 0.25)), 0.01, format="%.3f")
        tc_max = st.number_input("tc máx (anual)", tc_min+0.01, 1.00, max(0.35, float(st.session_state.get("tc_ann", 0.25))+0.10), 0.01, format="%.3f")
        tc_n   = st.slider("N precios", 2, 12, 6)
    with cc2:
        mv_min = st.number_input("mv mín", 3, 60, int(st.session_state.get("mv", 12)))
        mv_max = st.number_input("mv máx", mv_min, 60, max(int(st.session_state.get("mv", 12)), 24))
        mv_n   = st.slider("N mv", 1, 10, 4)
    with cc3:
        mp_min = st.number_input("mp mín", 1, 24, int(st.session_state.get("mp", 3)))
        mp_max = st.number_input("mp máx", mp_min, 24, max(int(st.session_state.get("mp", 3)), 6))
        mp_n   = st.slider("N mp", 1, 8, 3)
    with cc4:
        w_min = st.number_input("w_G mín", 0.0, 1.0, float(st.session_state.get("peso_total0", 1.0)), 0.05)
        w_max = st.number_input("w_G máx", w_min, 1.0, 1.0, 0.05)
        w_n   = st.slider("N w_G", 1, 8, 3)

    # ---------- Action grid ----------
    np.random.default_rng(int(seed))
    tc_grid = list(np.linspace(tc_min, tc_max, tc_n))
    mv_grid = list(np.linspace(mv_min, mv_max, mv_n, dtype=int))
    mp_grid = list(np.linspace(mp_min, mp_max, mp_n, dtype=int))
    wG_grid = list(np.linspace(w_min, w_max, w_n))
    actions = build_action_grid(tc_grid, mv_grid, mp_grid, wG_grid)   # assumes your helper returns objects with tc_ann, mv, mp, w_G
    nA = len(actions)
    st.markdown(f"**Acciones totales:** {nA} combinaciones")

    # ---------- Policy constraints ----------
    max_pd12 = st.slider("Máx PD(12m) aceptable", 0.05, 0.99, 0.90, 0.01, help="Penaliza acciones que excedan este límite.")
    penalty = st.slider("Penalización por violar política", 0.0, 1.0, 0.25, 0.05)

    # ---------- Hybrid model (Neural-HMM): helpers ----------
    def _softmax(v):
        v = np.asarray(v, dtype=float)
        v = v - np.max(v)
        ev = np.exp(v)
        s = ev.sum()
        return ev/s if s>0 else np.ones_like(v)/len(v)

    def _relu(u):
        return np.maximum(u, 0.0)

    def _phi(ctx_, act_):
        # Feature map φ(x,a): keep simple & auditable; extend freely
        EAD = float(ctx_["EAD"]); G0 = float(ctx_["garantia_val0"])
        co = float(ctx_["co_ann"]); sc = float(ctx_["score"])
        tc = float(act_.tc_ann); mv = float(act_.mv); mp = float(act_.mp); wG = float(act_.w_G)
        lnE = np.log(max(EAD, 1.0))
        lnG = np.log(max(G0, 1.0))
        return np.array([
            1.0, sc, lnE, lnG, co, tc, mv/12.0, mp/12.0, wG,
            sc*tc, sc*wG, tc*wG, (mv/12.0)*(mp/12.0)
        ], dtype=float)

    def _ensure_hybrid_params(d_phi, H=16, seed_=12345):
        key = "hyb_params"
        if key in st.session_state:
            P = st.session_state[key]
            # reshape if dimension changed
            if P["W1"].shape[1] != d_phi:
                del st.session_state[key]
        if key not in st.session_state:
            rng = np.random.default_rng(seed_)
            W1 = 0.05 * rng.standard_normal((H, d_phi))
            b1 = np.zeros(H)
            Wp = 0.05 * rng.standard_normal((3, H))   # row for state P
            bp = np.zeros(3)
            Ws = 0.05 * rng.standard_normal((3, H))   # row for state S1
            bs = np.zeros(3)
            st.session_state[key] = {"W1":W1,"b1":b1,"Wp":Wp,"bp":bp,"Ws":Ws,"bs":bs}
        return st.session_state[key]

    def _A_of(ctx_, act_):
        z = _phi(ctx_, act_)
        P = _ensure_hybrid_params(len(z), H=16, seed_=int(seed))
        h = _relu(P["W1"].dot(z) + P["b1"])
        # Row for P = softmax of logits
        rowP = _softmax(P["Wp"].dot(h) + P["bp"])
        # Row for S1 = softmax of logits
        rowS1 = _softmax(P["Ws"].dot(h) + P["bs"])
        # Absorbing S2
        A = np.array([
            [rowP[0],  rowP[1],  rowP[2]],
            [rowS1[0], rowS1[1], rowS1[2]],
            [0.0,      0.0,      1.0],
        ], dtype=float)
        # Numerical guard to keep rows stochastic
        A[0] /= A[0].sum()
        A[1] /= A[1].sum()
        return A

    def _powM(A, n):
        n = int(max(0, n))
        return np.linalg.matrix_power(A, n)

    def _total_garantia(df_g):
        try:
            col = None
            for c in df_g.columns:
                if str(c).strip().lower() in ("valor usd","valor_usd","valor","monto","monto usd"):
                    col = c; break
            if col is None and "Valor USD" in df_g.columns: col = "Valor USD"
            return float(pd.to_numeric(df_g[col], errors="coerce").fillna(0.0).sum()) if col is not None else 0.0
        except Exception:
            return 0.0

    def _pv_values(ctx_, act_):
        EAD = float(ctx_["EAD"])
        mv = int(act_.mv); mp = int(act_.mp)
        tc_ann = float(act_.tc_ann)
        co_ann = float(ctx_["co_ann"])
        tm_ann_fix = float(st.session_state.get("tm_ann_fija", TM_ANUAL_FIJO))

        i_m = (1.0 + tc_ann)**(1/12.0) - 1.0
        c_m = (1.0 + co_ann)**(1/12.0) - 1.0
        m_m = (1.0 + tm_ann_fix)**(1/12.0) - 1.0

        PV_t1 = (EAD * (1+i_m)**mv) / ((1+c_m)**mv)
        PV_t2_cura = (EAD * (1+i_m)**(mv+mp) * (1+m_m)**mp) / ((1+c_m)**(mv+mp))

        G_total = _total_garantia(ctx_["df_g"])
        PV_t2_def = float(act_.w_G) * float(G_total)  # política de recuperación
        return PV_t1, PV_t2_cura, PV_t2_def

    def evaluate_action_hybrid(ctx_, act_, reward_metric="ret_ann_vp"):
        # Transition matrix and path probabilities
        A = _A_of(ctx_, act_)
        mv = int(act_.mv); mp = int(act_.mp)

        pi0 = np.array([1.0, 0.0, 0.0])
        pi_t1 = pi0.dot(_powM(A, mv))
        PD1 = float(pi_t1[1])  # prob en S1 a t1 (rama "paga" vs "cura/default")

        eS1 = np.array([0.0, 1.0, 0.0])
        q = eS1.dot(_powM(A, mp))
        if use_forced_resolution:
            PD2_cond = float(q[2])
        else:
            denom = float(q[0] + q[2])
            PD2_cond = float(q[2] / denom) if denom > 0 else 0.0

        p_pay1 = 1.0 - PD1
        p_cura = PD1 * (1.0 - PD2_cond)
        p_def2 = PD1 * PD2_cond

        # 12m PD from HMM
        PD_12m = float((pi0.dot(_powM(A, 12)))[2])
        lam = -np.log(max(1e-12, 1.0 - PD_12m))  # hazard-equivalent for your plots

        # Cashflow PVs and EV
        PV_t1, PV_t2_cura, PV_t2_def = _pv_values(ctx_, act_)
        EV_VP = p_pay1*PV_t1 + p_cura*PV_t2_cura + p_def2*PV_t2_def

        # Rewards
        EAD = float(ctx_["EAD"])
        T_exp = float(mv) + float(PD1) * float(mp)
        ret_ann_vp = (EV_VP / EAD)**(12.0 / max(T_exp, 1e-9)) - 1.0 if EAD > 0 else 0.0
        ev_mult = (EV_VP / EAD) - 1.0 if EAD > 0 else 0.0
        reward = float(ret_ann_vp if reward_metric == "ret_ann_vp" else ev_mult)

        # Policy LGD (solo para KPI visual)
        G_total = _total_garantia(ctx_["df_g"])
        recov_ratio = float(act_.w_G) * (G_total / EAD) if EAD > 0 else 0.0
        LGD = max(0.0, 1.0 - min(1.0, recov_ratio))

        res = {
            "PD1": PD1, "PD2_cond": PD2_cond,
            "p_pay1": p_pay1, "p_cura": p_cura, "p_def2": p_def2,
            "PV_t1": PV_t1, "PV_t2_cura": PV_t2_cura, "PV_t2_def": PV_t2_def,
            "EV_VP": EV_VP, "ret_ann_vp": ret_ann_vp, "ev_mult": ev_mult,
            "PD_12m": PD_12m, "lam": lam, "LGD": LGD
        }
        return reward, res

    # ---------- Bandit context (still your original context_features) ----------
    x = context_features(ctx["score"], ctx["EAD"], ctx["garantia_val0"], ctx["co_ann"])
    d = x.shape[0]
    agent = LinUCB(n_actions=nA, d=d, alpha=float(alpha), reward_clip=2.0)

    # ---------- RL training ----------
    hist = []
    best = {"reward": -1e18, "a": None, "res": None}
    prog = st.progress(0, text="Entrenando política (Neural-HMM)...")
    for t in range(int(steps)):
        a_idx = agent.select(x)
        act = actions[a_idx]
        r, res = evaluate_action_hybrid(ctx, act, reward_metric=reward_metric)

        # Policy guardrail: penaliza si PD12 supera máximo
        if float(res.get("PD_12m", 0.0)) > float(max_pd12):
            r -= float(penalty)

        agent.update(a_idx, x, r)
        hist.append((t, a_idx, r, act, res))

        if r > best["reward"]:
            best = {"reward": r, "a": a_idx, "res": res}

        if (t+1) % max(1, int(steps//100)) == 0:
            prog.progress(min(1.0, (t+1)/steps), text=f"Entrenando política (Neural-HMM)... {t+1}/{steps}")

    prog.progress(1.0, text="Entrenamiento completado")

    # ---------- Results ----------
    if best["a"] is not None:
        best_act = actions[best["a"]]
        st.success(f"Mejor acción ⇒ tc={best_act.tc_ann:.3f}, mv={best_act.mv}, mp={best_act.mp}, w_G={best_act.w_G:.2f} · Recompensa≈{best['reward']:.4f}")

        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card("tc* (anual)", f"{best_act.tc_ann:.3f}")
        with k2: kpi_card("mv*, mp*", f"{best_act.mv}, {best_act.mp}")
        with k3: kpi_card("w_G*", f"{best_act.w_G:.2f}")
        with k4: kpi_card("PD(12m)", fmt_pct(best["res"]["PD_12m"]))

        # Árbol EV (reusa tus gráficas)
        st.markdown("**Árbol VP (acción óptima)**")
        try:
            plot_decision_tree_clean(best["res"], mv=best_act.mv, mp=best_act.mp, cliente=str(st.session_state.get("cliente","Cliente")))
        except Exception:
            pass
        try:
            show_ev_contributions(best["res"])
        except Exception:
            pass

        # Top-k actions
        topk = 12
        ranked = sorted(hist, key=lambda z: z[2], reverse=True)[:topk]
        rows = []
        for _, _, r, act, res in ranked:
            rows.append({
                "tc_ann": act.tc_ann, "mv": act.mv, "mp": act.mp, "w_G": act.w_G,
                "Reward": r, "ret_ann_vp": res.get("ret_ann_vp"),
                "EV/EAD-1": (res["EV_VP"]/ctx["EAD"] - 1.0) if ctx["EAD"]>0 else np.nan,
                "PD12": res.get("PD_12m"), "LGD": res.get("LGD")
            })
        df_top = pd.DataFrame(rows)

    # ---------- PD equations (no nested expander) ----------
    with st.expander("📐 Ecuaciones de la PD (Neural-HMM)", expanded=False):
        if best["a"] is not None:
            st.latex(r"A(z)=\begin{bmatrix} p_{PP}&p_{PS1}&p_{PS2}\\ p_{S1P}&p_{S1S1}&p_{S1S2}\\ 0&0&1\end{bmatrix},\quad p_{\cdot}=\mathrm{softmax}(W h + b)")
            st.latex(r"\pi_{t_1}=\pi_0 A^{mv},\quad PD_1=\big[\pi_{t_1}\big]_{S1},\quad q=e_{S1}^\top A^{mp},\quad PD_{2|1}=q_{S2}")
            st.latex(r"PD_{12m}=\big[\pi_0 A^{12}\big]_{S2},\quad \lambda=-\ln(1-PD_{12m})")
            st.dataframe(
                df_top.style.format({
                    "tc_ann": "{:.3f}", "w_G": "{:.2f}", "Reward": "{:.4f}",
                    "ret_ann_vp": "{:.2%}", "EV/EAD-1": "{:.2%}", "PD12": "{:.2%}", "LGD": "{:.2%}"
                }),
                use_container_width=True, hide_index=True
            )

# ===================== ✉️ Enviar PDF (LaTeX) por email =====================
import os, shutil, tempfile, subprocess, textwrap
import smtplib, ssl
from email.message import EmailMessage
import streamlit as st  # <- NECESARIO

# ---------- Utilidades ----------
def _latex_escape(s: str) -> str:
    repl = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
    }
    return ''.join(repl.get(c, c) for c in str(s))

def _pct_ltx(x) -> str:
    """0.1234 -> '12.34\\%' (en LaTeX % comenta líneas, así que hay que escaparlo)."""
    try:
        return f"{float(x):.2%}".replace("%", r"\%")
    except Exception:
        return r"0.00\%"

def _has_pdflatex() -> bool:
    if shutil.which("pdflatex"):
        return True
    # macOS: intenta agregar MacTeX al PATH
    texbin = "/Library/TeX/texbin"
    if os.path.isdir(texbin):
        os.environ["PATH"] = texbin + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("pdflatex") is not None

# ---------- Generación del .tex ----------
def _build_latex_summary(lang: str) -> str:
    """
    Construye el documento LaTeX completo desde st.session_state.
    lang: 'es' o 'en'
    """
    is_es = (lang == "es")
    babel = r"\usepackage[spanish]{babel}" if is_es else r"\usepackage[english]{babel}"
    titulo = "Policy Lab — Resumen técnico" if is_es else "Policy Lab — Technical Summary"

    # Datos (con defaults seguros)
    cliente = _latex_escape(st.session_state.get("cliente", "Cliente"))
    score   = float(st.session_state.get("score", 3.0))
    EAD     = float(st.session_state.get("EAD", 1_000_000.0))
    co_ann  = float(st.session_state.get("co_ann", 0.015))
    mv      = int(st.session_state.get("mv", 12))
    mp      = int(st.session_state.get("mp", 3))
    tc_ann  = float(st.session_state.get("tc_ann", 0.25))
    wG      = float(st.session_state.get("peso_total0", 1.0))

    res     = st.session_state.get("res_loc", {}) or {}
    PD12    = float(res.get("PD_12m", 0.10))
    lam     = float(res.get("lam", 0.10))
    pv_t1   = float(res.get("PV_t1", EAD))
    pv_t2c  = float(res.get("PV_t2_cura", EAD))
    pv_t2d  = float(res.get("PV_t2_def", 0.0))
    ev_vp   = float(res.get("EV_VP", EAD))
    retann  = float(res.get("ret_ann_vp", 0.0))
    lgd     = float(res.get("LGD", 0.45))

    # % seguros para LaTeX
    PD12_s = _pct_ltx(PD12)
    LGD_s  = _pct_ltx(lgd)
    RET_s  = _pct_ltx(retann)

    # Bullets (ES / EN) — OJO: duplicamos llaves { -> {{ y } -> }} para f-strings
    bullets_es = rf"""
\begin{{itemize}}
  \item Cliente: {cliente}.
  \item Score: {score:.2f} \quad EAD: {EAD:,.0f}\,USD \quad CO$_{{ann}}$: {co_ann:.3f}.
  \item Términos: $tc_{{ann}}={tc_ann:.3f}$, $mv={mv}$, $mp={mp}$, $w_G={wG:.2f}$.
  \item KPI: $PD(12m)={PD12_s}$, $\lambda={lam:.4f}$, $LGD={LGD_s}$.
  \item VP: $PV_{{t1}}={pv_t1:,.0f}$, $PV_{{t2}}^{{\mathrm{{cura}}}}={pv_t2c:,.0f}$, $PV_{{t2}}^{{\mathrm{{def}}}}={pv_t2d:,.0f}$, $EV_{{VP}}={ev_vp:,.0f}$, $ret_{{ann}}={RET_s}$.
\end{{itemize}}
""".strip()

    bullets_en = rf"""
\begin{{itemize}}
  \item Client: {cliente}.
  \item Score: {score:.2f} \quad EAD: {EAD:,.0f}\,USD \quad CO$_{{ann}}$: {co_ann:.3f}.
  \item Terms: $tc_{{ann}}={tc_ann:.3f}$, $mv={mv}$, $mp={mp}$, $w_G={wG:.2f}$.
  \item KPIs: $PD(12m)={PD12_s}$, $\lambda={lam:.4f}$, $LGD={LGD_s}$.
  \item PV: $PV_{{t1}}={pv_t1:,.0f}$, $PV_{{t2}}^{{\mathrm{{cure}}}}={pv_t2c:,.0f}$, $PV_{{t2}}^{{\mathrm{{def}}}}={pv_t2d:,.0f}$, $EV_{{VP}}={ev_vp:,.0f}$, $ret_{{ann}}={RET_s}$.
\end{{itemize}}
""".strip()

    intro_es = r"""
\section*{Resumen}
Optimiza precio, tenores y garantía con un bandit contextual (LinUCB)
alimentado por un motor de riesgo híbrido (Neural-HMM). Este documento resume
los principales KPIs y fórmulas de la evaluación.
""".strip()

    intro_en = r"""
\section*{Summary}
Optimizes price, tenors and collateral with a contextual bandit (LinUCB)
powered by a hybrid risk engine (Neural-HMM). This document summarizes
the key KPIs and formulas.
""".strip()

    math_es = r"""
\section*{Esquema}
\[
EV_{VP}=p_{pay1}\,PV_{t_1}+p_{cura}\,PV_{t_2}^{\mathrm{cura}}+p_{def2}\,PV_{t_2}^{\mathrm{def}},\qquad
ret_{VP,ann}=\left(\frac{EV_{VP}}{EAD}\right)^{\frac{12}{T_{exp}}}-1,\quad
T_{exp}=mv+PD_1\,mp.
\]
""".strip()

    math_en = r"""
\section*{Scheme}
\[
EV_{VP}=p_{pay1}\,PV_{t_1}+p_{cure}\,PV_{t_2}^{\mathrm{cure}}+p_{def2}\,PV_{t_2}^{\mathrm{def}},\qquad
ret_{VP,ann}=\left(\frac{EV_{VP}}{EAD}\right)^{\frac{12}{T_{exp}}}-1,\quad
T_{exp}=mv+PD_1\,mp.
\]
""".strip()

    body = (intro_es + "\n\n" + bullets_es + "\n\n" + math_es) if is_es else (intro_en + "\n\n" + bullets_en + "\n\n" + math_en)

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{amsmath,amssymb}}
\usepackage{{microtype}}
{babel}
\usepackage[margin=1in]{{geometry}}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{{hyperref}}
\setlength{{\parskip}}{{6pt}}
\begin{{document}}
\begin{{center}}
{{\LARGE \textbf{{{titulo}}}}}\\[2mm]
{{\large {cliente}}}
\end{{center}}

{body}

\end{{document}}
""".strip()
    return tex

def _compile_tex_to_pdf(tex_source: str):
    """LaTeX -> PDF. Devuelve (pdf_bytes, log) o (None, log)."""
    with tempfile.TemporaryDirectory() as td:
        tex_path = os.path.join(td, "summary.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_source)
        try:
            run = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "summary.tex"],
                cwd=td, capture_output=True, text=True, timeout=90
            )
            log = run.stdout + "\n" + run.stderr
            pdf_path = os.path.join(td, "summary.pdf")
            if run.returncode == 0 and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    return f.read(), log
            return None, log
        except Exception as e:
            return None, str(e)

def _send_email_with_attachment_gmail(from_email: str, app_password: str,
                                      to_email: str, subject: str, body: str,
                                      attachment: bytes, filename: str):
    """SMTP Gmail (STARTTLS). Usa App Password (16 chars, sin espacios)."""
    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment:
        msg.add_attachment(attachment, maintype="application", subtype="pdf", filename=filename)
    try:
        ctx_ssl = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls(context=ctx_ssl)
            s.login(from_email, (app_password or "").replace(" ", ""))
            s.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)

# ---------------- UI ----------------
with st.expander("✉️ Enviar PDF (LaTeX) por email", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        pdf_lang_ui = st.selectbox("Idioma / Language", ["Español", "English"], key="pdf_mail_lang")
        to_email = st.text_input("Destinatario", key="pdf_to_mail", placeholder="nombre@dominio.com")
    with col2:
        from_email = st.text_input("Tu Gmail (remitente)", key="pdf_from_mail", placeholder="tuusuario@gmail.com")
        app_pwd = st.text_input("App Password (16 caracteres)", type="password", key="pdf_app_pwd")

    cA, cB = st.columns(2)
    with cA:
        gen_only = st.button("🧾 Generar PDF (descargar)", key="btn_gen_pdf_only")
    with cB:
        gen_and_send = st.button("🚀 Generar y enviar PDF", key="btn_gen_send_pdf")

    if gen_only or gen_and_send:
        lang_code = "es" if pdf_lang_ui == "Español" else "en"
        tex = _build_latex_summary(lang_code)

        # Vista previa segura del .tex (opcional, NO usar st.latex aquí)
        if st.checkbox("👀 Ver .tex (solo código)", key="preview_tex_mail"):
            st.code(tex, language="latex")

        if not _has_pdflatex():
            st.warning("No encontré `pdflatex`. Instala TeX Live/MacTeX (macOS: `brew install --cask mactex-no-gui`) y reinicia la app.")
            st.download_button("⬇️ Descargar .tex", data=tex.encode("utf-8"),
                               file_name="PolicyLab_Resumen.tex", mime="application/x-tex")
        else:
            pdf_bytes, log = _compile_tex_to_pdf(tex)
            if not pdf_bytes:
                st.error("No se pudo compilar el PDF.")
                with st.expander("Ver log LaTeX"):
                    st.code(log or "(sin log)")
                st.download_button("⬇️ Descargar .tex", data=tex.encode("utf-8"),
                                   file_name="PolicyLab_Resumen.tex", mime="application/x-tex")
            else:
                st.success("✅ PDF generado.")
                st.session_state["latest_summary_pdf"] = pdf_bytes
                st.download_button("⬇️ Descargar PDF",
                    data=pdf_bytes, file_name=f"PolicyLab_Resumen_{lang_code}.pdf", mime="application/pdf")

                if gen_and_send:
                    if not (from_email and app_pwd and to_email):
                        st.error("Completa remitente, App Password y destinatario.")
                    else:
                        subject = "Policy Lab — Resumen PDF" if lang_code == "es" else "Policy Lab — PDF Summary"
                        body_es = "Adjunto encontrarás el resumen en PDF de Policy Lab (Neural-HMM + EV)."
                        body_en = "Attached is the PDF summary of Policy Lab (Neural-HMM + EV)."
                        ok, err = _send_email_with_attachment_gmail(
                            from_email, app_pwd, to_email, subject,
                            body_es if lang_code == "es" else body_en,
                            pdf_bytes, f"PolicyLab_Resumen_{lang_code}.pdf"
                        )
                        if ok:
                            st.success("📨 Enviado correctamente.")
                        else:
                            st.error(f"❌ No se pudo enviar: {err}")
                            st.caption("Verifica: Gmail habilitado, App Password (sin espacios) y STARTTLS en 587.")
# ===================== fin bloque =====================
