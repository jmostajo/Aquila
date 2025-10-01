from pathlib import Path

p = Path("Aquila.py")
src = p.read_text(encoding="utf-8")

changed = False
css_anchor = 'st.markdown(css, unsafe_allow_html=True)'
kpi_css_block = r'''

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
'''
if "kpi-card" not in src:
    if css_anchor in src:
        src = src.replace(css_anchor, css_anchor + kpi_css_block)
        changed = True
heading = 'st.markdown("### 📈 Métricas Clave del Análisis")'
kpi_cards_block = r'''
st.markdown('<div class="kpi-wrap">', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-title">Probabilidad de Default</div>
                <div class="kpi-val">{resultado['PD_12m']*100:.2f}%</div>
                <div class="kpi-sub">Próximos 12 meses</div>
            </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-title">Pérdida por Default (LGD)</div>
                <div class="kpi-val">{resultado['LGD']*100:.2f}%</div>
                <div class="kpi-sub">Loss Given Default</div>
            </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-title">Pérdida Esperada (ECL)</div>
                <div class="kpi-val">{fmt_usd(resultado["ECL"], 2)}</div>
                <div class="kpi-sub">Expected Credit Loss</div>
            </div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-title">Retorno Esperado</div>
                <div class="kpi-val">{ret_pct*100:.2f}%</div>
                <div class="kpi-sub">Umbral: <span class="kpi-delta {'up' if decision_ok else 'down'}">{umbral*100:.0f}%</span></div>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)
'''
if heading in src and "kpi-card" not in src:
    insert_pos = src.index(heading) + len(heading)
    src = src[:insert_pos] + kpi_cards_block + src[insert_pos:]
    changed = True

if changed:
    p.write_text(src, encoding="utf-8")
    print("PATCH OK: KPI CSS + KPI cards inserted.")
else:
    print("PATCH: No changes (either already present or anchor not found).")
