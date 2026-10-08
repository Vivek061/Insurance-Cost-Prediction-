import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)

insurance_pipeline = joblib.load("insurance_pipeline.pkl")

st.title("Insurance Cost Prediction")
st.write(
    "Enter a customer's information to estimate their medical insurance cost."
)