import json
import requests
# Define the URL of the FastAPI server
BASE_URL = "http://0.0.0.0:8000"

# Define a function to send a POST request to the /predict endpoint
def predict_diabetes(glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    # Create a dictionary with the input data
    input_data = {
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    # Send a POST request to the /predict endpoint
    response = requests.post(f"{BASE_URL}/predict", json=input_data)

    # Check if the response was successful
    if response.status_code == 200:
        # Parse the JSON response
        prediction_data = response.json()
        return prediction_data
    else:
        return {"error": "Request failed", "status_code": response.status_code}