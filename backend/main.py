from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from warnings import simplefilter
import os, hopsworks, joblib
import pandas as pd
import numpy as np

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

app = FastAPI(title="AQI Forecast API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    
    # Load model from registry
    model_registry = project.get_model_registry()
    models = model_registry.get_models("aqi_forecast_xgboost")
    model_meta = max(models, key=lambda m: m.version)
    model_dir = model_meta.download()
    
    # Load model artifacts
    model_artifacts = joblib.load(f"{model_dir}/model.pkl")
    feature_cols = model_artifacts['feature_cols']
    forecast_horizons = model_artifacts['forecast_horizons']
    metadata = model_artifacts['metadata']
    
    print(f"Model loaded successfully!")
    print(f"Training date: {metadata['training_date']}")
    print(f"Features: {metadata['feature_count']}")
    print(f"Horizons: {forecast_horizons}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def prepare_features(df):
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Handle missing timestamps (hourly frequency)
    full_index = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='h')
    df = df.set_index('timestamp').reindex(full_index).rename_axis('timestamp').reset_index()
    
    # Fill missing values
    weather_cols = ['temp','wind_speed','wind_gusts','humidity_percent','dew_point','pressure','cloud_cover','visibility']
    pollutant_cols = ['pm_10','pm_2_5','no_2','o_3','so_2','co','aqi']
    
    # Set timestamp as index temporarily
    df = df.set_index('timestamp')
    
    # Interpolate weather features
    df[weather_cols] = df[weather_cols].interpolate(method='time')
    
    # Reset index
    df = df.reset_index()
    
    # Fill pollutants using forward fill + rolling mean
    df[pollutant_cols] = df[pollutant_cols].ffill().rolling(3, min_periods=1).mean()
    
    # Detect gaps
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().div(3600)
    df['gap_flag'] = (df['time_diff'] > 1).astype(int)
    df['gap_flag'] = df['gap_flag'].fillna(0)
    
    # Create lag and rolling features
    n_lags = metadata['n_lags']
    all_cols = pollutant_cols + weather_cols
    
    for col in all_cols:
        for lag in range(1, n_lags+1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    for col in all_cols:
        df[f"{col}_roll3"] = df[col].rolling(window=3).mean()
        df[f"{col}_roll6"] = df[col].rolling(window=6).mean()
        df[f"{col}_roll12"] = df[col].rolling(window=12).mean()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Drop rows with NaNs
    df = df.dropna().reset_index(drop=True)
    
    return df

@app.get("/")
def root():
    return {
        "status": "online",
        "model": "aqi_forecast_xgboost"
    }

@app.get("/predict")
def predict_aqi():
    if model_artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Connect to Hopsworks and load data
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_feature_pipeline", version=1)
        df = fg.read()
        
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Prepare features
        df_processed = prepare_features(df)
        
        if len(df_processed) == 0:
            raise HTTPException(status_code=400, detail="Not enough data to generate features")
        
        # Get latest observation
        last_row = df_processed.iloc[-1]
        current_timestamp = last_row['timestamp']
        
        # Prepare feature row
        feature_row = last_row[feature_cols].values.reshape(1, -1)
        
        # Get predictions from each horizon model
        models = model_artifacts['models']
        horizon_predictions = {}
        
        for horizon in forecast_horizons:
            pred = models[horizon].predict(feature_row)[0]
            horizon_predictions[horizon] = float(pred)
        
        # Interpolate between horizons for full 72-hour forecast
        future_predictions = []
        for hour in range(1, 73):
            if hour in horizon_predictions:
                # Direct prediction available
                future_predictions.append(horizon_predictions[hour])
            else:
                # Interpolate between nearest horizons
                lower_h = max([h for h in forecast_horizons if h < hour])
                upper_h = min([h for h in forecast_horizons if h > hour])
                
                # Linear interpolation
                weight = (hour - lower_h) / (upper_h - lower_h)
                interpolated = (1 - weight) * horizon_predictions[lower_h] + weight * horizon_predictions[upper_h]
                future_predictions.append(float(interpolated))
        
        # Generate future timestamps
        future_timestamps = pd.date_range(
            start=current_timestamp + pd.Timedelta(hours=1),
            periods=72,
            freq='h'
        )
        
        # Create predictions list
        predictions_list = [
            {
                "timestamp": str(ts),
                "aqi": round(aqi, 2),
                "hour_ahead": i + 1
            }
            for i, (ts, aqi) in enumerate(zip(future_timestamps, future_predictions))
        ]
        
        return {
            "success": True,
            "predictions": predictions_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)