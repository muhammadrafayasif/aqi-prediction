from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from warnings import simplefilter
from datetime import datetime, timezone
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
    print(f'Loaded latest model: v{model_meta.version}')
    model_dir = model_meta.download()
    
    # Load model artifacts
    model_artifacts = joblib.load(f"{model_dir}/model.pkl")
    feature_cols = model_artifacts['feature_cols']
    forecast_horizons = model_artifacts['forecast_horizons']
    metadata = model_artifacts['metadata']
    
    print(f"Model loaded successfully!")
    print(f"Features: {metadata['feature_count']}")
    print(f"Horizons: {forecast_horizons}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def prepare_features(df, debug=False):
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Remove duplicate timestamps (keep the last one)
    df = df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
    
    if debug:
        print(f"\n[DEBUG] Initial rows: {len(df)}")
        print(f"[DEBUG] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Detect gaps
    orig_time_diff = df["timestamp"].diff().dt.total_seconds().div(3600)
    real_gaps = orig_time_diff > 2
    gap_start_timestamps = df.loc[real_gaps, "timestamp"].tolist()
    
    if debug and gap_start_timestamps:
        print(f"[DEBUG] Found {len(gap_start_timestamps)} real outages")

    # Reindex to a complete hourly range
    full_index = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="h")
    df_reindexed = df.set_index("timestamp").reindex(full_index).rename_axis("timestamp").reset_index()
    
    if debug:
        print(f"[DEBUG] After reindexing: {len(df_reindexed)} rows")

    # Mark where real gaps start
    df_reindexed["gap_flag"] = 0
    for ts in gap_start_timestamps:
        df_reindexed.loc[df_reindexed["timestamp"] == ts, "gap_flag"] = 1

    # Define columns
    weather_cols = ["temp", "wind_speed", "wind_gusts", "humidity_percent",
                    "dew_point", "pressure", "cloud_cover", "visibility"]
    pollutant_cols = ["pm_10", "pm_2_5", "no_2", "o_3", "so_2", "co", "aqi"]

    # Causal imputation
    df_reindexed = df_reindexed.set_index("timestamp")
    df_reindexed[weather_cols] = df_reindexed[weather_cols].interpolate(
        method="time", limit_direction="forward"
    )
    df_reindexed[weather_cols] = df_reindexed[weather_cols].ffill().bfill()
    df_reindexed = df_reindexed.reset_index()

    # Pollutants: forward-fill and lightly smooth
    df_reindexed[pollutant_cols] = df_reindexed[pollutant_cols].ffill()
    df_reindexed[pollutant_cols] = df_reindexed[pollutant_cols].rolling(3, min_periods=1).mean()

    # Clip pollutant values to safe range (0–500)
    for col in pollutant_cols:
        df_reindexed[col] = np.clip(df_reindexed[col], 0, 500)

    # Remove if gap is very large (> 24 hours)
    gap_threshold_hours = 24

    # Compute time differences in hours
    orig_time_diff = df_reindexed["timestamp"].diff().dt.total_seconds() / 3600
    orig_time_diff = orig_time_diff.fillna(0)

    # Detect large gaps
    large_gaps = orig_time_diff > gap_threshold_hours
    large_gap_timestamps = df_reindexed.loc[large_gaps, "timestamp"].tolist()

    # Mark and filter
    df_reindexed["skip_row"] = 0
    df_reindexed.loc[df_reindexed["timestamp"].isin(large_gap_timestamps), "skip_row"] = 1

    # Drop skipped rows
    df_reindexed = df_reindexed[df_reindexed["skip_row"] == 0].copy().reset_index(drop=True)
    df_reindexed = df_reindexed.drop("skip_row", axis=1)

    
    if debug:
        print(f"[DEBUG] After gap filtering (> {gap_threshold_hours}h only): {len(df_reindexed)} rows")

    df = df_reindexed

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month"] = df["timestamp"].dt.month
    df["season"] = df["timestamp"].dt.month % 12 // 3 + 1

    all_cols = pollutant_cols + weather_cols

    # Lag features
    for col in all_cols:
        for lag in range(1, 13):  # n_lags = 12
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling features (based on lags)
    for col in all_cols:
        df[f"{col}_roll3"] = df[[f"{col}_lag{i}" for i in range(1, 4)]].mean(axis=1)
        df[f"{col}_roll6"] = df[[f"{col}_lag{i}" for i in range(1, 7)]].mean(axis=1)
        df[f"{col}_roll12"] = df[[f"{col}_lag{i}" for i in range(1, 13)]].mean(axis=1)

    # Count NaNs before dropping
    nan_count = df.isnull().sum().sum()
    if debug:
        print(f"[DEBUG] Total NaN values before dropping: {nan_count}")
        print(f"[DEBUG] Rows with ANY NaN: {df.isnull().any(axis=1).sum()}")

    # Drop rows with any NaN from lag creation
    df_before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    
    # Forward-fill remaining NaN from lag creation (handles edge cases at boundaries)
    df = df.bfill().ffill()
    
    if debug:
        print(f"[DEBUG] Rows before dropna: {df_before_drop}, after: {len(df)}")
        if len(df) > 0:
            print(f"[DEBUG] Last row timestamp: {df['timestamp'].iloc[-1]}")
    
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
        fg = fs.get_feature_group("aqi_feature_pipeline", version=2)
        df = fg.read()
        
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Prepare features
        df_processed = prepare_features(df, debug=False)
        
        if len(df_processed) == 0:
            raise HTTPException(status_code=400, detail="Not enough valid data after processing")
        
        # Get latest observation
        last_row = df_processed.iloc[-1]
        current_timestamp = last_row['timestamp']
        
        # Check data freshness (with online store, data should be < 3 hours old)
        data_age = (datetime.now(timezone.utc) - pd.to_datetime(current_timestamp)).total_seconds() / 3600
        
        if data_age > 3:
            print(f"[WARNING] Data is {data_age:.1f}h old, predictions may be unreliable")
        
        # Check for NaN in feature row
        feature_row = last_row[feature_cols].values.reshape(1, -1)
        if pd.isna(feature_row).any():
            nan_indices = np.where(np.isnan(feature_row[0]))[0]
            nan_features = [feature_cols[i] for i in nan_indices]
            raise HTTPException(status_code=400, detail=f"NaN values in features: {nan_features}")
        
        print(f"[PREDICT] Using timestamp: {current_timestamp}")
        print(f"[PREDICT] Data age: {data_age:.2f}h")
        print(f"[PREDICT] Feature shape: {feature_row.shape}")
        
        # Get predictions from each horizon model
        models = model_artifacts['models']
        horizon_predictions = {}
        
        for horizon in forecast_horizons:
            pred = models[horizon].predict(feature_row)[0]
            pred = np.clip(pred, 0, 500)
            horizon_predictions[horizon] = float(pred)
            print(f"[PREDICT] Horizon {horizon}h: {pred:.2f}")
        
        # Interpolate between horizons for full 72-hour forecast
        future_predictions = []
        for hour in range(1, 73):
            if hour in horizon_predictions:
                future_predictions.append(horizon_predictions[hour])
            else:
                lower_h = max([h for h in forecast_horizons if h < hour])
                upper_h = min([h for h in forecast_horizons if h > hour])
                
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
        import traceback
        print(f"[ERROR] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)