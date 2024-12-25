# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from pathlib import Path
# from tensorflow.keras.models import load_model

# # Set page title and layout
# st.set_page_config(page_title="Air Quality Prediction", layout="centered")
# st.title("Air Quality Classification")
# # Title and welcome message
# st.title("Selamat Datang di Aplikasi Penilaian Kualitas Udara!")
# st.markdown("""
# ### Aplikasi ini memungkinkan Anda untuk menilai kualitas udara menggunakan dua model pembelajaran mesin yang berbeda: **Random Forest** dan **FeedForward Neural Network (FFNN)**.
# Silakan pilih model yang ingin digunakan untuk memprediksi kualitas udara di wilayah Anda.
# """)


# # Sidebar for model selection
# st.sidebar.header("Model Selection")
# model_type = st.sidebar.radio("Pilih Model", ("FFNN", "Random Forest"))

# # Input fields for features
# st.sidebar.header("Input Features")
# temperature = st.sidebar.number_input("Temperature (\u00b0C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
# humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
# pm25 = st.sidebar.number_input("PM2.5 (\u00b5g/m\u00b3)", min_value=0.0, max_value=500.0, value=30.0, step=0.1)
# pm10 = st.sidebar.number_input("PM10 (\u00b5g/m\u00b3)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
# no2 = st.sidebar.number_input("NO2 (\u00b5g/m\u00b3)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
# so2 = st.sidebar.number_input("SO2 (\u00b5g/m\u00b3)", min_value=0.0, max_value=200.0, value=15.0, step=0.1)
# co = st.sidebar.number_input("CO (mg/m\u00b3)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
# proximity = st.sidebar.number_input("Proximity to Industrial Areas (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
# population_density = st.sidebar.number_input("Population Density (people/km\u00b2)", min_value=0.0, max_value=10000.0, value=1000.0, step=1.0)

# # Function for FFNN prediction
# def predict_ffnn(features):
#     model_path = Path(__file__).parent / "model_FFNN/ffnn_model.h5"
#     normalizer_path = Path(__file__).parent / "model_FFNN/normalizer.joblib"
#     scaler_path = Path(__file__).parent / "model_FFNN/scaler.joblib"
#     encoder_path = Path(__file__).parent / "model_FFNN/label_encoder.joblib"

#     model = load_model(model_path)
#     normalizer = joblib.load(normalizer_path)
#     scaler = joblib.load(scaler_path)
#     label_encoder = joblib.load(encoder_path)

#     input_data = pd.DataFrame([features], columns=[
#         "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
#         "Proximity_to_Industrial_Areas", "Population_Density"
#     ])
#     input_data[["Population_Density"]] = normalizer.transform(input_data[["Population_Density"]])
#     input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]] = scaler.transform(
#         input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]])

#     probabilities = model.predict(input_data)[0]
#     predicted_class = np.argmax(probabilities)
#     predicted_label = label_encoder.inverse_transform([predicted_class])[0]

#     return predicted_label, probabilities

# # Function for Random Forest prediction
# def predict_rf(features):
#     model_path = Path(__file__).parent / "model_RF/random_forest_model.joblib"
#     normalizer_path = Path(__file__).parent / "model_RF/normalizer.joblib"
#     scaler_path = Path(__file__).parent / "model_RF/scaler.joblib"
#     encoder_path = Path(__file__).parent / "model_RF/label_encoder.joblib"

#     model = joblib.load(model_path)
#     normalizer = joblib.load(normalizer_path)
#     scaler = joblib.load(scaler_path)
#     label_encoder = joblib.load(encoder_path)

#     input_data = pd.DataFrame([features], columns=[
#         "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
#         "Proximity_to_Industrial_Areas", "Population_Density"
#     ])
#     input_data[["Population_Density"]] = normalizer.transform(input_data[["Population_Density"]])
#     input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]] = scaler.transform(
#         input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]])

#     probabilities = model.predict_proba(input_data)[0]
#     predicted_class = np.argmax(probabilities)
#     predicted_label = label_encoder.inverse_transform([predicted_class])[0]

#     return predicted_label, probabilities

# # Handle predictions
# if st.button("Prediksi", type="primary"):
#     st.subheader("Hasil Prediksi")
#     features = [
#         temperature, humidity, pm25, pm10, no2, so2, co, proximity, population_density
#     ]

#     with st.spinner('Memproses data untuk prediksi...'):
#         if model_type == "FFNN":
#             predicted_label, probabilities = predict_ffnn(features)
#         else:
#             predicted_label, probabilities = predict_rf(features)

#     st.write(f"Prediksi Kualitas Udara: **{predicted_label}**")

#     # Visualize probabilities
#     label_path = Path(__file__).parent / "model_RF/label_encoder.joblib"
#     label_encoder = joblib.load(label_path)
#     classes = label_encoder.classes_
#     prob_df = pd.DataFrame({"Class": classes, "Probability": probabilities})
#     fig = px.pie(prob_df, names="Class", values="Probability", title="Distribusi Probabilitas Kelas")
#     st.plotly_chart(fig)

# # CSS for styling
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #f8f9fa;
#         font-family: Arial, sans-serif;
#     }
#     .stButton > button {
#         background-color: #007bff;
#         color: white;
#         font-size: 16px;
#         padding: 10px 20px;
#         border-radius: 8px;
#     }
#     .stSidebar {
#         background-color: #e9ecef;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from tensorflow.keras.models import load_model

# Set page title and layout
st.set_page_config(page_title="Air Quality Prediction", layout="centered")
st.title("Air Quality Classification")
# Title and welcome message
st.title("Selamat Datang di Aplikasi Penilaian Kualitas Udara!")
st.markdown("""
### Aplikasi ini memungkinkan Anda untuk menilai kualitas udara menggunakan dua model pembelajaran mesin yang berbeda: **Random Forest** dan **FeedForward Neural Network (FFNN)**.
Silakan pilih model yang ingin digunakan untuk memprediksi kualitas udara di wilayah Anda.
""")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.radio("Pilih Model", ("FFNN", "Random Forest"))

# Layout for input features (in the main body)
st.header("Input Fitur untuk Prediksi Kualitas Udara")

# Create columns for input fields
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (\u00b0C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    pm25 = st.number_input("PM2.5 (\u00b5g/m\u00b3)", min_value=0.0, max_value=500.0, value=30.0, step=0.1)
    pm10 = st.number_input("PM10 (\u00b5g/m\u00b3)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)

with col2:
    no2 = st.number_input("NO2 (\u00b5g/m\u00b3)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    so2 = st.number_input("SO2 (\u00b5g/m\u00b3)", min_value=0.0, max_value=200.0, value=15.0, step=0.1)
    co = st.number_input("CO (mg/m\u00b3)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
    proximity = st.number_input("Proximity to Industrial Areas (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    population_density = st.number_input("Population Density (people/km\u00b2)", min_value=0.0, max_value=10000.0, value=1000.0, step=1.0)

# Function for FFNN prediction
def predict_ffnn(features):
    model_path = Path(__file__).parent / "model_FFNN/ffnn_model.h5"
    normalizer_path = Path(__file__).parent / "model_FFNN/normalizer.joblib"
    scaler_path = Path(__file__).parent / "model_FFNN/scaler.joblib"
    encoder_path = Path(__file__).parent / "model_FFNN/label_encoder.joblib"

    model = load_model(model_path)
    normalizer = joblib.load(normalizer_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    input_data = pd.DataFrame([features], columns=[
        "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
        "Proximity_to_Industrial_Areas", "Population_Density"
    ])
    input_data[["Population_Density"]] = normalizer.transform(input_data[["Population_Density"]])
    input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]] = scaler.transform(
        input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]])

    probabilities = model.predict(input_data)[0]
    predicted_class = np.argmax(probabilities)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label, probabilities

# Function for Random Forest prediction
def predict_rf(features):
    model_path = Path(__file__).parent / "model_RF/random_forest_model.joblib"
    normalizer_path = Path(__file__).parent / "model_RF/normalizer.joblib"
    scaler_path = Path(__file__).parent / "model_RF/scaler.joblib"
    encoder_path = Path(__file__).parent / "model_RF/label_encoder.joblib"

    model = joblib.load(model_path)
    normalizer = joblib.load(normalizer_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    input_data = pd.DataFrame([features], columns=[
        "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
        "Proximity_to_Industrial_Areas", "Population_Density"
    ])
    input_data[["Population_Density"]] = normalizer.transform(input_data[["Population_Density"]])
    input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]] = scaler.transform(
        input_data[["Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO", "Proximity_to_Industrial_Areas"]])

    probabilities = model.predict_proba(input_data)[0]
    predicted_class = np.argmax(probabilities)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label, probabilities

# Handle predictions
if st.button("Prediksi", type="primary"):
    st.subheader("Hasil Prediksi")
    features = [
        temperature, humidity, pm25, pm10, no2, so2, co, proximity, population_density
    ]

    with st.spinner('Memproses data untuk prediksi...'):
        if model_type == "FFNN":
            predicted_label, probabilities = predict_ffnn(features)
        else:
            predicted_label, probabilities = predict_rf(features)

    st.write(f"Prediksi Kualitas Udara: **{predicted_label}**")

    # Visualize probabilities
    label_path = Path(__file__).parent / "model_RF/label_encoder.joblib"
    label_encoder = joblib.load(label_path)
    classes = label_encoder.classes_
    prob_df = pd.DataFrame({"Class": classes, "Probability": probabilities})
    fig = px.pie(prob_df, names="Class", values="Probability", title="Distribusi Probabilitas Kelas")
    st.plotly_chart(fig)

# CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stSidebar {
        background-color: #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True
)
