from datetime import timedelta
from fastapi import FastAPI
import os, hopsworks, joblib
import pandas as pd

app = FastAPI(title="AQI Forecast API", version="1.0")

# Connect to Hopsworks
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
print("Connected to Hopsworks project.")

# Load Model & Scaler
model_registry = project.get_model_registry()
models = model_registry.get_models("aqi_forecast_model")
model_meta = max(models, key=lambda m: m.version)
model_dir = model_meta.download()
model = joblib.load(f"{model_dir}/model.pkl")
scaler = joblib.load(f"{model_dir}/scaler.pkl")
print("Model and scaler loaded successfully.")

@app.get("/predict")
def predict_aqi():
    # Load latest dataset
    feature_group = fs.get_or_create_feature_group(name="aqi_feature_pipeline", version=1)
    df = feature_group.read()
    df = df.sort_values(by="timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.ffill()

    # Recreate lag features
    for lag in range(1, 7):
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
    df = df.dropna().reset_index(drop=True)

    # Forecast 72 hours
    future_predictions = []
    recent_rows = df.tail(6).copy()

    for i in range(72):
        input_row = recent_rows.iloc[-1:].copy()

        for lag in range(1, 7):
            input_row[f"aqi_lag_{lag}"] = recent_rows["aqi"].iloc[-lag]

        input_row["hour"] = input_row["timestamp"].dt.hour
        input_row["dayofweek"] = input_row["timestamp"].dt.dayofweek

        # Scale input
        X_scaled = scaler.transform(input_row.drop(columns=["timestamp", "aqi"], errors="ignore"))

        # Predict
        predicted_aqi = model.predict(X_scaled)[0]

        # Next timestamp
        next_timestamp = input_row["timestamp"].iloc[0] + timedelta(hours=1)

        future_predictions.append({"timestamp": next_timestamp.isoformat(), "predicted_aqi": round(predicted_aqi, 3)})

        # Update recent rows for next iteration
        new_row = input_row.copy()
        new_row["timestamp"] = next_timestamp
        new_row["aqi"] = predicted_aqi
        recent_rows = pd.concat([recent_rows, new_row]).tail(6)

    return {"predictions": future_predictions}
