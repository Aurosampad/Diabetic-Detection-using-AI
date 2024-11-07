from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('app/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define possible outcomes
class_names = np.array(['0', '1'])

# Initialize FastAPI app
app= FastAPI()

# Define a Pydantic model for the input data
class DiabetesInput(BaseModel):
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Root endpoint
@app.get('/')
def read_root():
    return {'message': 'Diabetes Detection API'}

# Prediction endpoint
@app.post('/predict')
def predict(input_data: DiabetesInput):
    # Convert input data to numpy array
    features = pd.DataFrame({
        'Glucose': [input_data.Glucose],
        'BloodPressure': [input_data.BloodPressure],
        'SkinThickness': [input_data.SkinThickness],
        'Insulin': [input_data.Insulin],
        'BMI': [input_data.BMI],
        'DiabetesPedigreeFunction': [input_data.DiabetesPedigreeFunction],
        'Age': [input_data.Age]
    })
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]  # Probability of being diabetic
    
    # Return the result
    return {
        'prediction': class_names[prediction[0]],
        'probability_of_diabetes': round(probability, 2)
    }
