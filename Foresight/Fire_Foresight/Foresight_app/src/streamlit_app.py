import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from huggingface_hub import hf_hub_download
import lightgbm as lgb
from model_interface import run_h3_processing
import tempfile
# Model


@st.cache_resource
def load_lightgbm_model():
    model_path = hf_hub_download(
        repo_id="AchG/Fire_Foresight",
        filename="lightgbm_fire_model.txt"
    )
    return lgb.Booster(model_file=model_path)

model = load_lightgbm_model()


st.set_page_config(page_title="Morocco Fire Risk", page_icon="../assets/Group 1.svg", layout="wide")

st.title("🔥 Morocco Wildfire Risk Predictor")
st.write("Provide your weather conditions and see a fire-risk map at the same time.")


left, right = st.columns([1.2, 1.1])  # Adjust ratio for balance


with left:
    st.subheader("📊 Prediction Inputs")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            temperature_max = st.number_input("Max Temperature (°C)", 0.0, 50.0, 30.0)
            wind_speed_max = st.number_input("Max Wind Speed (km/h)", 0.0, 150.0, 20.0)
            precipitation_total = st.number_input("Total Precipitation (mm)", 0.0, 200.0, 0.0)
            relative_humidity = st.number_input("Humidity (%)", 0.0, 100.0, 40.0)

        with col2:
            soil_moisture = st.number_input("Soil Moisture (0–1)", 0.0, 1.0, 0.2)
            evapotranspiration = st.number_input("Evapotranspiration", 0.0, 10.0, 3.0)
            shortwave_radiation = st.number_input("Radiation (W/m²)", 0.0, 500.0, 250.0)

        day_of_year = st.slider("Day of Year", 1, 365, 150)
        day_of_week = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 3)
        is_weekend = st.selectbox("Weekend?", ["No", "Yes"])
        is_weekend = 1 if is_weekend == "Yes" else 0

        longitude = st.number_input("Longitude", -17.0, -1.0, -7.0)
        latitude = st.number_input("Latitude", 20.0, 36.0, 32.0)
        sea_distance = st.number_input("Distance from Sea (km)", 0.0, 50000.0, 50.0)

        submitted = st.form_submit_button("Predict Fire Risk")


    if submitted:
        features = pd.DataFrame([{
            "temperature_max": temperature_max,
            "wind_speed_max": wind_speed_max,
            "precipitation_total": precipitation_total,
            "relative_humidity": relative_humidity,
            "soil_moisture": soil_moisture,
            "evapotranspiration": evapotranspiration,
            "shortwave_radiation": shortwave_radiation,
            "day_of_year": day_of_year,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "longitude": longitude,
            "latitude": latitude,
            "sea_distance": sea_distance
        }])

        prob = float(model.predict(features)[0])
        st.metric("🔥 Fire Probability", f"{prob:.2%}")

        if prob > 0.7:
            st.error("🔥 HIGH RISK — Fire likely!")
        elif prob > 0.4:
            st.warning("⚠️ MODERATE RISK — Be cautious.")
        else:
            st.success("✅ LOW RISK — Conditions safe.")


with right:
    st.markdown("### 🗺️ Morocco Fire Risk Map")

    st.markdown("### 📂 Upload CSV")

    uploaded_csv = st.file_uploader("Choose CSV", type=["csv"])

    h3_res = st.slider("H3 Resolution", 1, 9, 6)

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.success("CSV uploaded!")

        with st.spinner("Processing H3 grid and building map..."):

            # Create a temporary file for the map output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                temp_map_path = tmp.name

            # Run your H3 processing using the dataframe directly
            run_h3_processing(
                df,                         # dataframe instead of path
                output_map=temp_map_path,   # temporary file path
                resolution=h3_res
            )

        st.success("Map ready!")

        # Display the generated map
        with open(temp_map_path, "r", encoding="utf-8") as f:
            html = f.read()

        st.components.v1.html(html, height=650, scrolling=True)






