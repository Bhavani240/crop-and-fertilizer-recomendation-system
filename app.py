import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained crop prediction model
model = pickle.load(open("model.pkl", "rb"))

# Load the fertilizer dataset
fertilizer_df = pd.read_csv("Fertilizer Prediction.csv")

# Streamlit App Title
st.title("🌾 Crop and Fertilizer Recommendation System")

# Sidebar Inputs
st.sidebar.header("📝 Enter Soil & Weather Details")
nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=140, value=50)
potassium = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=140, value=50)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
pH = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

# On Button Click
if st.sidebar.button("🚀 Recommend"):
    # Crop Prediction
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
    crop_prediction = model.predict(input_data)[0]

    st.subheader("🌱 Recommended Crop")
    st.success(f"👉 {crop_prediction}")

    # Fertilizer Recommendation Logic (closest nutrient match)
    def recommend_fertilizer(N, P, K):
        df = fertilizer_df.copy()
        df["diff"] = abs(df["Nitrogen"] - N) + abs(df["Phosphorous"] - P) + abs(df["Potassium"] - K)
        best_match = df.loc[df["diff"].idxmin()]
        return best_match["Fertilizer Name"], best_match["Crop Type"]

    fert_name, for_crop = recommend_fertilizer(nitrogen, phosphorus, potassium)

    st.subheader("🧪 Recommended Fertilizer")
    st.info(f"👉 {fert_name} (usually recommended for {for_crop})")
