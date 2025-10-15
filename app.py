import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

file_path = os.path.join(os.path.dirname(__file__), "pipe.pkl")
try:
    pipe = joblib.load(file_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.header("🚗 Car Price Predictor")

year = st.number_input("Year of Purchase", min_value=1990, max_value=2024, step=1)
kms = st.number_input("Kilometers Driven", min_value=0)
fuel = st.selectbox("Fuel Type", ["Diesel", "Petrol"])
seller = st.selectbox("Seller Type", ["Individual", "Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner"])
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, format="%.2f")
engine = st.number_input("Engine (CC)", min_value=0)
power = st.number_input("Max Power (bhp)", min_value=0.0, format="%.2f")
seats = st.number_input("Seats", min_value=2, max_value=10, step=1)
brand = st.selectbox(
    "Brand",
    [
        "Maruti", "Hyundai", "Mahindra", "Tata", "Ford", 
        "Honda", "Toyota", "Renault", "Chevrolet", "Volkswagen"
    ]
)

if st.button("Predict Price"):
    # Create input DataFrame
    input_df = pd.DataFrame(
        [[year, kms, fuel, seller, transmission, owner, mileage, engine, power, seats, brand]],
        columns=[
            "year", "km_driven", "fuel", "seller_type", "transmission",
            "owner", "mileage", "engine", "max_power", "seats", "brand"
        ]
    )
    
    # Make prediction safely
    try:
        y_pred = pipe.predict(input_df)
        st.success(f"Estimated Car Price: ₹ {np.round(y_pred[0], 2)}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
