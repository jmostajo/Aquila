import streamlit as st
import pandas as pd
import datetime
import altair as alt
import io

# --------------------- CORE CLASSES ---------------------

class FechaHelper:
    @staticmethod
    def dias_entre(f1, f2):
        return (f2 - f1).days

class InteresFinanciero:
    def __init__(self, monto, tasa_compensatoria, tasa_moratoria, igv):
        self.monto = monto
        self.tasa_c = tasa_compensatoria
        self.tasa_m = tasa_moratoria
        self.igv = igv

    def calcular_interes_I(self, fecha_inicio, fecha_vencimiento):
        dias = FechaHelper.dias_entre(fecha_inicio, fecha_vencimiento)
        interes = ((1 + self.tasa_c) ** (dias / 30) - 1) * self.monto if dias > 0 else 0.0
        return interes, dias

    def calcular_interes_II_y_mora(self, fecha_vencimiento, fecha_pago, interes_I):
        if fecha_pago <= fecha_vencimiento:
            return 0.0, 0.0, 0
        dias = FechaHelper.dias_entre(fecha_vencimiento, fecha_pago)
        base = self.monto + interes_I
        interes_II = ((1 + self.tasa_c) ** (dias / 30) - 1) * base
        interes_mora = ((1 + self.tasa_m) ** (dias / 30) - 1) * base
        return interes_II, interes_mora, dias

    def calcular_totales(self, interes_I, interes_II, interes_mora):
        total_sin_igv = interes_I + interes_II
        igv_calc = total_sin_igv * self.igv
        total_con_igv = total_sin_igv + igv_calc
        total = total_con_igv + interes_mora
        return total_sin_igv, igv_calc, total_con_igv, total

class ReporteDinamico:
    def __init__(self, fecha_inicio, fecha_vencimiento, fecha_pago, interes_I, interes_II, interes_mora, dias_I, dias_II):
        self.inicio = fecha_inicio
        self.vencimiento = fecha_vencimiento
        self.pago = fecha_pago
        self.i_I = interes_I
        self.i_II = interes_II
        self.i_mora = interes_mora
        self.d_I = dias_I
        self.d_II = dias_II

    def generar(self):
        i_dia_I = round(self.i_I / self.d_I, 6) if self.d_I > 0 else 0.0
        i_dia_II = round(self.i_II / self.d_II, 6) if self.d_II > 0 else 0.0
        i_dia_mora = round(self.i_mora / self.d_II, 6) if self.d_II > 0 else 0.0
        reporte = []
        fecha_actual = self.inicio

        while fecha_actual < self.vencimiento:
            reporte.append({
                "Fecha": fecha_actual.strftime("%d/%m/%Y"),
                "Interés Diario I": i_dia_I,
                "Interés Diario II": 0.0,
                "Interés Moratorio": 0.0,
                "Total Diario": i_dia_I
            })
            fecha_actual += datetime.timedelta(days=1)

        while fecha_actual < self.pago:
            total_diario = round(i_dia_II + i_dia_mora, 6)
            reporte.append({
                "Fecha": fecha_actual.strftime("%d/%m/%Y"),
                "Interés Diario I": 0.0,
                "Interés Diario II": i_dia_II,
                "Interés Moratorio": i_dia_mora,
                "Total Diario": total_diario
            })
            fecha_actual += datetime.timedelta(days=1)

        return pd.DataFrame(reporte), i_dia_I, i_dia_II, i_dia_mora

class OvernightCalculator:
    def __init__(self, monto, tasa_anual, fecha_inicio, fecha_vencimiento):
        self.monto = monto
        self.tasa = tasa_anual
        self.inicio = fecha_inicio
        self.vencimiento = fecha_vencimiento

    def calcular(self):
        dias = FechaHelper.dias_entre(self.inicio, self.vencimiento)
        if dias <= 0:
            return 0.0, 0.0, dias
        interes_total = ((1 + self.tasa) ** (dias / 360) - 1) * self.monto
        interes_diario = interes_total / dias
        return interes_total, interes_diario, dias

# --------------------- STREAMLIT UI ---------------------

st.set_page_config(page_title="📈 Simulador de Intereses Financieros", layout="wide")
st.title("📊 Simulador de Intereses Financieros Diarios")

with st.sidebar:
    st.header("🧾 Parámetros del cálculo")

    contrato = st.text_input("📄 # Contrato Cliente (CE)", placeholder="Ej.: POST 114")

    # 👉 NUEVO dropdown agregado aquí:
    tipo_general_operacion = st.selectbox(
        "🏷️ Tipo de operación",
        ["Operación Comex (comercio exterior)", 
         "Operación de capital de trabajo estándar", 
         "Operaciones de cronograma escalonadas"]
    )

    tipo_operacion = st.selectbox("📋 Código de documento", ["FE01", "FE02", "FE03", "ND01", "ND02", "NC01", "OTRO"])
    codigo_operacion = st.text_input("🔢 Código de operación", placeholder="Ej.: 1423")
    descripcion = f"{tipo_operacion} {codigo_operacion}" if codigo_operacion else tipo_operacion

    fecha_inicio = st.date_input("📅 Fecha de inicio", datetime.date.today() - datetime.timedelta(days=60))
    fecha_vencimiento = st.date_input("📅 Fecha de vencimiento", datetime.date.today() - datetime.timedelta(days=30))
    fecha_pago = st.date_input("📅 Fecha de pago", datetime.date.today())

    monto = st.number_input("💵 Monto financiado (USD)", min_value=0.01, format="%.2f")
    tasa_c_pct = st.number_input("📈 Tasa compensatoria (% mensual)", min_value=0.01, step=0.01, format="%.4f")
    tasa_m_pct = st.number_input("⚠️ Tasa moratoria (% mensual)", min_value=0.01, step=0.01, format="%.4f")
    igv_pct = st.number_input("📊 IGV (%)", min_value=0.0, max_value=100.0, value=18.0, step=0.01)

    st.markdown("---")
    st.subheader("💤 Cotización Overnight")
    f_over_inicio = st.date_input("📅 Fecha desembolso Overnight", datetime.date.today())
    f_over_venc = st.date_input("📅 Fecha vencimiento Overnight", datetime.date.today() + datetime.timedelta(days=1))
    monto_over = st.number_input("💵 Base importe Overnight", min_value=0.01, format="%.2f")
    tasa_over_pct = st.number_input("📉 Tasa anual Overnight (%)", min_value=0.0, step=0.0001, format="%.6f")

# Validaciones
if fecha_inicio > fecha_vencimiento:
    st.sidebar.error("⚠️ La fecha de inicio debe ser menor o igual a la fecha de vencimiento.")
    st.stop()
if fecha_pago < fecha_inicio:
    st.sidebar.error("⚠️ La fecha de pago debe ser mayor o igual a la fecha de inicio.")
    st.stop()

# Cálculos principales
financiero = InteresFinanciero(monto, tasa_c_pct / 100, tasa_m_pct / 100, igv_pct / 100)
i_I, d_I = financiero.calcular_interes_I(fecha_inicio, fecha_vencimiento)
i_II, i_mora, d_II = financiero.calcular_interes_II_y_mora(fecha_vencimiento, fecha_pago, i_I)
total_sin_igv, igv_calc, total_con_igv, total_final = financiero.calcular_totales(i_I, i_II, i_mora)

reporte = ReporteDinamico(fecha_inicio, fecha_vencimiento, fecha_pago, i_I, i_II, i_mora, d_I, d_II)
df_reporte, i_dia_I, i_dia_II, i_dia_mora = reporte.generar()

# ------ Exportar reporte a Excel ------
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df_reporte.to_excel(writer, index=False, sheet_name='Reporte Intereses')
buffer.seek(0)

st.download_button(
    label="📥 Exportar reporte a Excel",
    data=buffer,
    file_name=f"reporte_intereses_{contrato.replace(' ', '_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# -------- Resultados principales --------
overnight = OvernightCalculator(monto_over, tasa_over_pct / 100, f_over_inicio, f_over_venc)
interes_over, interes_d_over, dias_over = overnight.calcular()

if dias_over <= 0:
    st.sidebar.error("⚠️ El plazo del Overnight debe ser mayor a 0 días.")
    st.stop()

# -------- Gráfico --------
st.markdown("### 📈 Evolución de Intereses Diarios")

df_reporte['Fecha'] = pd.to_datetime(df_reporte['Fecha'], format="%d/%m/%Y")
df_chart = df_reporte[['Fecha', 'Interés Diario I', 'Interés Diario II', 'Interés Moratorio', 'Total Diario']]
df_melted = df_chart.melt('Fecha', var_name='Tipo de Interés', value_name='Valor Diario')

# --- Agregar el Interés Diario Overnight ---
if interes_d_over > 0 and dias_over > 0:
    fechas_overnight = [f_over_inicio + datetime.timedelta(days=i) for i in range(dias_over)]
    overnight_df = pd.DataFrame({
        'Fecha': fechas_overnight,
        'Tipo de Interés': ['Interés Diario Overnight'] * dias_over,
        'Valor Diario': [interes_d_over] * dias_over
    })
    df_melted = pd.concat([df_melted, overnight_df], ignore_index=True)
else:
    st.warning("ℹ️ El interés diario del overnight es cero o negativo, no se mostrará en el gráfico.")
    st.dataframe(df_melted)

# --- Anotaciones ---
fechas_clave = {
    'Inicio Operación': fecha_inicio.strftime("%Y-%m-%d"),
    'Fecha Vencimiento': fecha_vencimiento.strftime("%Y-%m-%d"),
    'Inicio Plazo Moratorio': (fecha_vencimiento + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
}

annotations = pd.DataFrame({
    'Fecha': pd.to_datetime(list(fechas_clave.values())),
    'Evento': list(fechas_clave.keys())
})

lineas_ano = alt.Chart(annotations).mark_rule(color='gray').encode(
    x='Fecha:T',
    tooltip='Evento:N'
)

puntos_ano = alt.Chart(annotations).mark_point(color='red', size=100).encode(
    x='Fecha:T',
    tooltip='Evento:N'
)

base = alt.Chart(df_melted).mark_line().encode(
    x='Fecha:T',
    y='Valor Diario:Q',
    color='Tipo de Interés:N',
    tooltip=['Fecha:T', 'Tipo de Interés:N', 'Valor Diario:Q']
).interactive()

grafico = base + lineas_ano + puntos_ano

st.altair_chart(grafico, use_container_width=True)

# -------- Tabla resumen --------
st.markdown("### 🧾 Resumen de Resultados")
col1, col2 = st.columns(2)

with col1:
    st.metric("Interés Compensatorio I", f"${i_I:,.2f}")
    st.metric("Interés Compensatorio II", f"${i_II:,.2f}")
    st.metric("Interés Moratorio", f"${i_mora:,.2f}")
    st.metric("Total Intereses (sin IGV)", f"${total_sin_igv:,.2f}")

with col2:
    st.metric("IGV calculado", f"${igv_calc:,.2f}")
    st.metric("Total con IGV (sin mora)", f"${total_con_igv:,.2f}")
    st.metric("Total Final (con mora)", f"${total_final:,.2f}")
    st.metric("Interés Diario Overnight", f"${interes_d_over:,.6f}")

# -------- Mostrar tabla detallada --------
with st.expander("Ver detalle diario de intereses"):
    st.dataframe(df_reporte.style.format({
        "Interés Diario I": "${:,.6f}",
        "Interés Diario II": "${:,.6f}",
        "Interés Moratorio": "${:,.6f}",
        "Total Diario": "${:,.6f}"
    }), height=350)
