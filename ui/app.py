import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Heart Disease Prediction")

model = joblib.load("C:/Users/pc/Downloads/Heart_Disease_Project/models/final_model.pkl")
pca_pipeline = joblib.load("C:/Users/pc/Downloads/pca_pipeline.pkl")

features = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"]

# Collect user input
inputs = {}
for feat in features:
    if feat in ["cp", "restecg", "slope", "thal", "ca"]:  # categorical
        val = st.number_input(f"{feat}", min_value=0, max_value=10, value=0)
    else:  # continuous
        val = st.number_input(f"{feat}", value=0.0)
    inputs[feat] = val

# Prediction
if st.button("Predict"):
    X = pd.DataFrame([inputs])             # raw input
    X_pca = pca_pipeline.transform(X)      # apply PCA
    pred = model.predict(X_pca)[0]
    prob = model.predict_proba(X_pca)[0,1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result")
    st.write(f"**{int(pred)} (1 = Heart Disease, 0 = No Heart Disease)**")
    if prob is not None:
        st.write(f"Probability of disease: **{prob:.2f}**")