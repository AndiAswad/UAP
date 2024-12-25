from sklearn.calibration import LabelEncoder
import streamlit as st
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model  # Use load_model instead of joblib

# Title and Description
st.set_page_config(page_title="Air Quality Prediction", layout="centered")
st.title("Air Quality Classification")
st.markdown("""
### Selamat datang di aplikasi prediksi kualitas udara!
Masukkan nilai untuk setiap fitur di bawah ini, lalu klik tombol **Prediksi** untuk mengetahui Kondisi udara di wilayah anda.
""")

# Input fields for 9 features
st.sidebar.header("Input Features")
with st.sidebar:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=30.0, step=0.1)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=200.0, value=15.0, step=0.1)
    co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
    proximity = st.number_input("Proximity to Industrial Areas (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    population_density = st.number_input("Population Density (people/km²)", min_value=0.0, max_value=10000.0, value=1000.0, step=1.0)

# Prediction function
def predict_air_quality(features):
    # Define paths for model, normalizer, scaler, and label encoder
    model_path = Path(__file__).parent / "model_FFNN/ffnn_model.h5"  # Use load_model for Keras models
    normalizer_path = Path(__file__).parent / "model_FFNN/normalizer.joblib"
    scaler_path = Path(__file__).parent / "model_FFNN/scaler.joblib"
    encoder_path = Path(__file__).parent / "model_FFNN/label_encoder.joblib"

    # Load the model using Keras' load_model method
    model = load_model(model_path)  # Changed this line to use load_model
    normalizer = joblib.load(normalizer_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    
    # Prepare input data
    input_data = pd.DataFrame([features], columns=[
        "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
        "Proximity_to_Industrial_Areas", "Population_Density"
    ])
    input_data[["Population_Density"]] = normalizer.transform(input_data[["Population_Density"]])
    input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]] = scaler.transform(
        input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]])

    # Predict using the model
    probabilities = model.predict(input_data)[0]  # Changed this line to use predict()
    predicted_class = np.argmax(probabilities)
    
    # Decode the predicted class back to its original label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, probabilities

# Predict button
if st.button("Prediksi", type="primary"):
    st.subheader("Hasil Prediksi")
    features = [
        temperature, humidity, pm25, pm10, no2, so2, co, proximity, population_density
    ]
    with st.spinner('Memproses data untuk prediksi...'):
        predicted_label, probabilities = predict_air_quality(features)

    st.write(f"Prediksi Kualitas Udara: **{predicted_label}**")

    # Visualize probabilities
    model_path = Path(__file__).parent / "model_FFNN/label_encoder.joblib"
    label_encoder = joblib.load(model_path)
    classes = label_encoder.classes_  # Get original class labels
    prob_df = pd.DataFrame({"Class": classes, "Probability": probabilities})
    fig = px.pie(prob_df, names="Class", values="Probability", title="Distribusi Probabilitas Kelas")
    st.plotly_chart(fig)

# Optional: Add model interpretation (e.g., SHAP or Grad-CAM)
st.sidebar.markdown("---")
st.sidebar.subheader("Fitur Lanjutan")
st.sidebar.markdown("Gunakan fitur ini untuk memahami model lebih baik (masih dalam pengembangan).")
