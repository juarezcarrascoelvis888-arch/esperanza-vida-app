import streamlit as st
import numpy as np
import pickle

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
st.set_page_config(
    page_title="Esperanza de Vida",
    page_icon="🌍",
    layout="centered"
)

# --------------------------------------------------
# ESTILOS (DISEÑO CLARO)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}

h1 {
    color: #1f4e79;
    text-align: center;
}

h3 {
    color: #2c3e50;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-top: 20px;
}

.resultado {
    font-size: 26px;
    font-weight: bold;
    color: #1e8449;
    text-align: center;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CARGA DEL MODELO Y SCALER
# --------------------------------------------------
with open("modelo_esperanza_vida.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --------------------------------------------------
# INTERFAZ
# --------------------------------------------------
st.title("🌍 Predicción de la Esperanza de Vida")
st.write(
    "Esta aplicación utiliza un **modelo de regresión basado en "
    "Máquinas de Soporte Vectorial (SVR)** para estimar la esperanza "
    "de vida de un país."
)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Ingrese los datos del país")

    year = st.number_input("📅 Año", min_value=1900, max_value=2100, value=2015)
    adult_mortality = st.number_input("🧓 Mortalidad adulta", min_value=0.0)
    infant_deaths = st.number_input("👶 Muertes infantiles", min_value=0.0)
    alcohol = st.number_input("🍷 Consumo de alcohol", min_value=0.0)
    percentage_expenditure = st.number_input("💸 Gasto en salud (%)", min_value=0.0)
    hepatitis_b = st.number_input("💉 Cobertura Hepatitis B", min_value=0.0)
    measles = st.number_input("🦠 Casos de sarampión", min_value=0.0)
    bmi = st.number_input("⚖️ Índice de masa corporal (BMI)", min_value=0.0)
    diphtheria = st.number_input("💉 Cobertura difteria", min_value=0.0)
    population = st.number_input("👥 Población", min_value=0.0)
    thinness = st.number_input("📉 Delgadez (1-19 años)", min_value=0.0)
    schooling = st.number_input("🎓 Años de escolaridad", min_value=0.0)
    gdp = st.number_input("💰 PIB per cápita", min_value=0.0)

    if st.button("🔮 Calcular esperanza de vida"):
        datos = np.array([[
            year,
            adult_mortality,
            infant_deaths,
            alcohol,
            percentage_expenditure,
            hepatitis_b,
            measles,
            bmi,
            diphtheria,
            population,
            thinness,
            schooling,
            gdp
        ]])

        datos_scaled = scaler.transform(datos)
        prediccion = modelo.predict(datos_scaled)

        st.markdown(
            f'<div class="resultado">🧬 Esperanza de vida estimada: '
            f'{prediccion[0]:.2f} años</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)