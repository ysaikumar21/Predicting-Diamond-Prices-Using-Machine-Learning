import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly
import sklearn
import tensorflow
import gdown
import os

# Function to download model if not found
def download_model(file_id, output):
    if not os.path.exists(output):
        st.info(f"Downloading {output} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
        st.success(f"{output} download complete!")

# Google Drive File ID for the trained model
model_file = "diamond_price_model.pkl"
model_id = "1QgY6hpRoDJcuF8waxx8Ril2nhw4CjkJV"

# Check and download model if necessary
download_model(model_id, model_file)

# Load the trained model
@st.cache_data
def load_model():
    with open(model_file, "rb") as model_file_obj:
        model = pickle.load(model_file_obj)
    return model

model = load_model()

# Streamlit UI
st.title("💎 Diamond Price Estimator")
st.markdown("### Predict the price of a diamond based on its characteristics.")

# User Inputs
carat = st.slider("Carat Weight", 0.1, 5.0, step=0.1)
cut = st.selectbox("Cut Quality", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
size = st.number_input("Size (x * y * z) in mm³", min_value=1.0, step=0.1)

# Encoding categorical variables
cut_mapping = {"Fair": 5, "Good": 3, "Very Good": 4, "Premium": 2, "Ideal": 1}
cut_encoded = cut_mapping[cut]

# Convert input into model format
input_features = np.array([[carat, cut_encoded, size]])

# Prediction Button
if st.button("💎 Predict Price"):
    predicted_price = model.predict(input_features)[0]
    st.success(f"Estimated Diamond Price: **${predicted_price:,.2f}**")

# Footer
st.markdown("---")
st.markdown("**Developed by YARAMALA SAIKUMAR** | Powered by Machine Learning & Streamlit 🚀")
