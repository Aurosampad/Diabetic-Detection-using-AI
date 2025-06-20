import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
with open('app2/model_logreg.pkl', 'rb') as f:
    model = pickle.load(f)

# Define class names
class_names = np.array(['0', '1'])  # 0: Not Diabetic, 1: Diabetic

# Streamlit App
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Detection App")

st.write("Enter the following health parameters to predict diabetes:")

# Input form
with st.form("diabetes_form"):
    Glucose = st.number_input("Glucose", min_value=0.0, value=199.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, value=122.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, value=99.0)
    Insulin = st.number_input("Insulin", min_value=0.0, value=846.0)
    BMI = st.number_input("BMI", min_value=0.0, value=67.1)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.078, value=2.42)
    Age = st.number_input("Age", min_value=21, value=81)

    submit = st.form_submit_button("Predict")

# Predict and display result
if submit:
    input_df = pd.DataFrame({
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    if class_names[prediction] == '1':
        st.error(f"🚨 The patient is **likely diabetic** (Probability: {round(probability, 2)})")
    else:
        st.success(f"✅ The patient is **not diabetic** (Probability: {round(probability, 2)})")


