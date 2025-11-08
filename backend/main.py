from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import os, hopsworks, joblib
import pandas as pd

app = FastAPI(title="AQI Forecast API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Hopsworks
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
print("Connected to Hopsworks project.")

# Load Model & Scaler
model_registry = project.get_model_registry()
models = model_registry.get_models("xgboost_aqi_forecast_model")
model_meta = max(models, key=lambda m: m.version)
model_dir = model_meta.download()
model = joblib.load(f"{model_dir}/xgboost_model.pkl")
print("Model loaded successfully.")

@app.get("/predict")
def predict_aqi():
    fg = fs.get_feature_group("aqi_feature_pipeline", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    weather_cols = ['temp','wind_speed','wind_gusts','humidity_percent','dew_point','pressure',
                'cloud_cover','visibility']
    pollutant_cols = ['pm_10','pm_2_5','no_2','o_3','so_2','co','aqi']
    all_cols = pollutant_cols + weather_cols
    n_lags = 24

    # Feature engineering
    lag_arrays = []
    for lag in range(1, n_lags+1):
        lagged = df[all_cols].shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in all_cols]
        lag_arrays.append(lagged)
    lag_features = pd.concat(lag_arrays, axis=1)

    roll3 = df[all_cols].rolling(3).mean().add_suffix('_roll3')
    roll6 = df[all_cols].rolling(6).mean().add_suffix('_roll6')

    time_features = pd.DataFrame({
        'hour': df['timestamp'].dt.hour,
        'day_of_week': df['timestamp'].dt.dayofweek
    })
    time_features['is_weekend'] = time_features['day_of_week'].isin([5,6]).astype(int)

    df_full = pd.concat([df, lag_features, roll3, roll6, time_features], axis=1).dropna().reset_index(drop=True)

    # Prepare latest features
    feature_cols = [col for col in df_full.columns if col not in ['timestamp','aqi']]
    latest_features = df_full[feature_cols].iloc[-1].values.reshape(1, -1)

    # Predict next 72 hours
    future_aqi_pred = model.predict(latest_features).flatten()

    # Prepare timestamps
    forecast_horizon = 72
    future_timestamps = pd.date_range(
        start=df_full['timestamp'].iloc[-1] + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq='h'
    )

    predictions_list = [
        {"timestamp": str(ts), "aqi": float(aqi)}  # convert timestamp to string and aqi to float
        for ts, aqi in zip(future_timestamps, future_aqi_pred)
    ]

    return {"predictions": predictions_list}