import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel

app = FastAPI(
    title="AquaGuard ML API",
    description="Water quality prediction API for ESP32 & Supabase",
)

# Load your trained .pkl models and encoder
try:
  reg_model = joblib.load("bangus_regression_model.pkl")
  clf_model = joblib.load("bangus_classification_model.pkl")
  label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
  print(f"Error loading models: {e}")


# Define expected incoming data structure from ESP32 / Supabase
class WaterSensorData(BaseModel):
  dissolved_oxygen: float
  temperature: float
  ph: float
  ammonia: float
  turbidity: float
  salinity: float


@app.get("/")
def home():
  return {"status": "AquaGuard ML API is running!"}


@app.post("/predict")
def predict_water_quality(data: WaterSensorData):
  try:
    # Convert incoming JSON payload to DataFrame matching training features
    input_df = pd.DataFrame(
        [{
            "dissolved_oxygen": data.dissolved_oxygen,
            "temperature": data.temperature,
            "ph": data.ph,
            "ammonia": data.ammonia,
            "turbidity": data.turbidity,
            "salinity": data.salinity,
        }]
    )

    # Make predictions using your .pkl models
    predicted_score = float(reg_model.predict(input_df)[0])
    predicted_encoded = clf_model.predict(input_df)[0]
    predicted_category = str(
        label_encoder.inverse_transform([predicted_encoded])[0]
    )

    return {
        "water_quality_score": round(predicted_score, 2),
        "water_quality_category": predicted_category,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))