import streamlit as st
import pandas as pd
import datetime

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

# Sidebar inputs
with st.sidebar:
    st.header("🧾 Parámetros del cálculo")

    contrato = st.text_input("📄 # Contrato Cliente (CE)", placeholder="Ej.: POST 114")
    descripcion = st.text_input("📝 Descripción de la operación", placeholder="Ej.: FE01 1423")

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

# Validations for dates
if fecha_inicio > fecha_vencimiento:
    st.sidebar.error("⚠️ La fecha de inicio debe ser menor o igual a la fecha de vencimiento.")
    st.stop()

if fecha_pago < fecha_inicio:
    st.sidebar.error("⚠️ La fecha de pago debe ser mayor o igual a la fecha de inicio.")
    st.stop()

# Calculos financieros
financiero = InteresFinanciero(monto, tasa_c_pct / 100, tasa_m_pct / 100, igv_pct / 100)
i_I, d_I = financiero.calcular_interes_I(fecha_inicio, fecha_vencimiento)
i_II, i_mora, d_II = financiero.calcular_interes_II_y_mora(fecha_vencimiento, fecha_pago, i_I)
total_sin_igv, igv_calc, total_con_igv, total_final = financiero.calcular_totales(i_I, i_II, i_mora)

# Reporte dinámico
reporte = ReporteDinamico(fecha_inicio, fecha_vencimiento, fecha_pago, i_I, i_II, i_mora, d_I, d_II)
df_reporte, i_dia_I, i_dia_II, i_dia_mora = reporte.generar()

import altair as alt
import pandas as pd

st.markdown("### 📈 Evolución de Intereses Diarios")

df_reporte['Fecha'] = pd.to_datetime(df_reporte['Fecha'], format="%d/%m/%Y")

df_chart = df_reporte[['Fecha', 'Interés Diario I', 'Interés Diario II', 'Interés Moratorio', 'Total Diario']]
df_melted = df_chart.melt('Fecha', var_name='Tipo de Interés', value_name='Valor Diario')

# Definir fechas clave (ajusta estas fechas según tus datos reales)
fechas_clave = {
    'Inicio Operación': '2024-11-15',
    'Fecha Vencimiento': '2025-01-10',
    'Inicio Plazo Moratorio': '2025-01-11',
    'Inicio Interés Diario I': '2024-11-16',
    'Inicio Interés Diario II': '2024-11-20',
}

# Construir DataFrame anotaciones con valor vertical automático (max valor de ese día)
annotations = pd.DataFrame({
    'Fecha': pd.to_datetime(list(fechas_clave.values())),
    'Evento': list(fechas_clave.keys())
})

# Para el Y, obtener máximo valor diario ese día para posicionar texto justo encima
def obtener_valor_y(fecha):
    valores = df_melted.loc[df_melted['Fecha'] == fecha, 'Valor Diario']
    if not valores.empty:
        return valores.max() * 1.1  # un poco más arriba para no tapar la línea
    else:
        return 0

annotations['Valor Diario'] = annotations['Fecha'].apply(obtener_valor_y)

# Gráfico base
line_chart = alt.Chart(df_melted).mark_line(point=True).encode(
    x=alt.X('Fecha:T', title='Fecha'),
    y=alt.Y('Valor Diario:Q', title='Valor (USD)', scale=alt.Scale(zero=True)),
    color=alt.Color('Tipo de Interés:N', title='Tipo de Interés'),
    tooltip=[
        alt.Tooltip('Fecha:T', title='Fecha'),
        alt.Tooltip('Tipo de Interés:N', title='Tipo'),
        alt.Tooltip('Valor Diario:Q', title='Valor (USD)', format=',.2f')
    ]
)

# Capa de anotaciones
annotation_layer = alt.Chart(annotations).mark_text(
    align='left',
    baseline='middle',
    fontSize=13,
    font='Helvetica Neue',
    dx=5,
    dy=-10,
    color='black',
    angle=0,
    opacity=0.9
).encode(
    x='Fecha:T',
    y='Valor Diario:Q',
    text='Evento:N'
)

# Combinar capas
final_chart = (line_chart + annotation_layer).properties(
    width='container',
    height=450,
    title=alt.TitleParams(
        text='🧠 Evolución de Intereses Diarios con Anotaciones Clave',
        subtitle='Fechas importantes en el ciclo financiero',
        anchor='start',
        fontSize=20,
        subtitleFontSize=14
    )
).interactive()

st.altair_chart(final_chart, use_container_width=True)



# Overnight
overnight = OvernightCalculator(monto_over, tasa_over_pct / 100, f_over_inicio, f_over_venc)
interes_over, interes_d_over, dias_over = overnight.calcular()
if dias_over <= 0:
    st.sidebar.error("⚠️ El plazo del Overnight debe ser mayor a 0 días.")
    st.stop()

# --------- Visualización de Resultados ---------

st.success("✅ Cálculo exitoso. Revisa los resultados a continuación:")

st.markdown(f"### 🧾 Contrato Cliente (CE): {contrato}")
st.markdown(f"### 📝 Descripción: {descripcion}")

st.markdown("---")

st.markdown("### ⏳ Plazos y Tasas Diarias")
col1, col2, col3 = st.columns(3)
col1.metric("📆 Plazo I (Días)", f"{d_I} días", f"${i_dia_I:.6f}/día")
col2.metric("📆 Plazo II (Días)", f"{d_II} días", f"${i_dia_II:.6f}/día")
col3.metric("📆 Plazo Mora (Días)", f"{d_II} días", f"${i_dia_mora:.6f}/día")

st.markdown("### 📑 Reporte Diario Detallado de Intereses")
st.dataframe(
    df_reporte.style.format({
        "Interés Diario I": "{:.6f}",
        "Interés Diario II": "{:.6f}",
        "Interés Moratorio": "{:.6f}",
        "Total Diario": "{:.6f}"
    }),
    use_container_width=True
)

st.markdown("### 💰 Detalle Final de Intereses")
col_a, col_b = st.columns([2,1])
with col_a:
    st.markdown(f"""
- **Interés I (sin IGV):** USD {i_I:,.6f}
- **Interés II (sin IGV):** USD {i_II:,.6f}
- **Total Compensatorio sin IGV:** USD {total_sin_igv:,.6f}
- **IGV ({igv_pct:.2f}%):** USD {igv_calc:,.6f}
- **Total con IGV:** USD {total_con_igv:,.6f}
- **Interés Moratorio:** USD {i_mora:,.6f}
    """)

with col_b:
    st.markdown(f"### ✅ **Total Final a Pagar: USD {total_final:,.2f}**")

st.markdown("---")

st.markdown("### 💤 Detalle Cotización Overnight")
st.info(f"""
- **Interés total Overnight:** USD {interes_over:,.6f}
- **Interés diario Overnight:** USD {interes_d_over:,.6f}
- **Días del período:** {dias_over} días
""")

import streamlit as st
import pandas as pd

def calcular_resultado_y_valor_cuota(total_ingresos, total_gastos, patrimonio_inicial, aporte_participe, cantidad_cuotas):
    resultado_diario = total_ingresos - total_gastos
    patrimonio_final = patrimonio_inicial + resultado_diario + aporte_participe
    valor_cuota = patrimonio_final / cantidad_cuotas if cantidad_cuotas > 0 else 0.0
    return resultado_diario, patrimonio_final, valor_cuota

def main():
    st.title("🧮 Cálculo del Valor Cuota Diario Neto (USD) - Múltiples Fechas")

    if "fechas_data" not in st.session_state:
        st.session_state.fechas_data = []

    with st.form(key="formulario_fecha"):
        st.subheader("📅 Ingreso de datos para una fecha específica")
        fecha = st.date_input("Selecciona la Fecha")
        
        # Paso 1: Ingresos y gastos
        total_ingresos = st.number_input("💰 Total Ingresos para la fecha (USD)", min_value=0.0, format="%.2f")
        total_gastos = st.number_input("💸 Total Gastos para la fecha (USD)", min_value=0.0, format="%.2f")

        # Paso 2: Resultado diario
        resultado_diario = total_ingresos - total_gastos
        st.write(f"➡️ **Resultado Diario (Ingresos - Gastos): USD {resultado_diario:,.2f}**")

        # Paso 3: Patrimonio y aporte
        patrimonio_inicial = st.number_input("🏦 Patrimonio Inicial hasta esta fecha (USD)", min_value=0.0, format="%.2f")
        aporte_participe = st.number_input("👤 Aporte del Partícipe (USD, si no hay, escribir 0)", min_value=0.0, format="%.2f")

        # Paso 4: Cálculo de patrimonio final
        patrimonio_final = patrimonio_inicial + resultado_diario + aporte_participe
        st.write(f"➡️ **Patrimonio Final = {patrimonio_inicial:,.2f} + {resultado_diario:,.2f} + {aporte_participe:,.2f} = USD {patrimonio_final:,.2f}**")

        # Paso 5: Cuotas y valor cuota neto diario
        cantidad_cuotas = st.number_input("🔢 Cantidad de Cuotas Finales para la fecha", min_value=0.000001, format="%.6f")
        valor_cuota = patrimonio_final / cantidad_cuotas if cantidad_cuotas > 0 else 0.0

        st.write(f"✅ **Valor Cuota Diario Neto para {fecha.strftime('%d/%m/%Y')}: USD {valor_cuota:,.6f}**")

        submitted = st.form_submit_button("✅ Agregar esta fecha a la tabla")

        if submitted:
            st.session_state.fechas_data.append({
                "Fecha": fecha.strftime("%d/%m/%Y"),
                "Total Ingresos (USD)": total_ingresos,
                "Total Gastos (USD)": total_gastos,
                "Resultado Diario (USD)": resultado_diario,
                "Patrimonio Inicial (USD)": patrimonio_inicial,
                "Aporte Partícipe (USD)": aporte_participe,
                "Patrimonio Final (USD)": patrimonio_final,
                "Cantidad Cuotas": cantidad_cuotas,
                "Valor Cuota Diario Neto (USD)": valor_cuota
            })
            st.success("✔️ Datos agregados correctamente.")

    if st.session_state.fechas_data:
        df = pd.DataFrame(st.session_state.fechas_data)
        st.markdown("### 📊 Tabla Consolidada por Fecha")
        st.dataframe(df.style.format({"Valor Cuota Diario Neto (USD)": "{:.6f}"}))

if __name__ == "__main__":
    main()
