#main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris
from pyngrok import ngrok, conf
import nest_asyncio
import uvicorn

# Patch asyncio event loop for notebooks
nest_asyncio.apply()

# Set ngrok authtoken
conf.get_default().auth_token = "YOUR_AUTH_TOKEN"  # Replace this with your token

# Load trained model
model = joblib.load("iris_model.pkl")
iris = load_iris()

# Initialize FastAPI
app = FastAPI(title="Iris Species Prediction API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API!"}

@app.post("/predict")
def predict_species(data: IrisInput):
    input_data = np.array([
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]).reshape(1, -1)
    
    prediction = model.predict(input_data)[0]
    predicted_species = iris.target_names[prediction]
    return {"predicted_species": predicted_species}

# Start FastAPI + ngrok
public_url = ngrok.connect(8000)
print(f"🔗 Public URL: {public_url}")
uvicorn.run(app, host="0.0.0.0", port=8000)
