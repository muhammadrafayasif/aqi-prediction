import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dotenv import load_dotenv
import hopsworks
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
load_dotenv()

# Load data from Hopsworks
project = hopsworks.login(api_key_value=os.getenv("API_KEY"))
fs = project.get_feature_store()
fg = fs.get_or_create_feature_group("aqi_feature_pipeline", version=1)
df = fg.read()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Original data shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Handle missing timestamps (hourly frequency)
full_index = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='h')
df = df.set_index('timestamp').reindex(full_index).rename_axis('timestamp').reset_index()

# Fill missing values and create gap flag
weather_cols = ['temp','wind_speed','wind_gusts','humidity_percent','dew_point','pressure','cloud_cover','visibility']
pollutant_cols = ['pm_10','pm_2_5','no_2','o_3','so_2','co','aqi']

# Set timestamp as index temporarily
df = df.set_index('timestamp')

# Interpolate weather features using time
df[weather_cols] = df[weather_cols].interpolate(method='time')

# Reset index to make timestamp a column again
df = df.reset_index()

# Fill pollutants using forward fill + rolling mean (to smooth across gaps)
df[pollutant_cols] = df[pollutant_cols].ffill().rolling(3, min_periods=1).mean()

# Detect gaps (more than 1 hour difference)
df['time_diff'] = df['timestamp'].diff().dt.total_seconds().div(3600)
df['gap_flag'] = (df['time_diff'] > 1).astype(int)
df['gap_flag'] = df['gap_flag'].fillna(0)  # first row has NaN

# Create lag and rolling features
n_lags = 12
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

# Drop rows with NaNs in lags
df = df.dropna().reset_index(drop=True)
print(f"Data shape after feature engineering: {df.shape}")

# Train separate models for different horizons

forecast_horizons = [1, 6, 12, 24, 48, 72]
models = {}

feature_cols = [col for col in df.columns if col not in ['timestamp', 'aqi', 'time_diff']]

# Train/validation/test split
n_samples = len(df)
train_end = int(0.7 * n_samples)
val_end = int(0.85 * n_samples)

print(f"\nTraining models for different horizons...")
for horizon in forecast_horizons:
    print(f"  Training model for {horizon}h ahead...")
    
    # Create dataset for this horizon
    X_list, y_list = [], []
    for i in range(len(df) - horizon):
        X_list.append(df[feature_cols].iloc[i].values)
        y_list.append(df['aqi'].iloc[i + horizon])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Split data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:gamma',
        random_state=42
    )
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              verbose=False)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    y_pred_test = np.clip(y_pred_test, 0, 500)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\nTest MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")
    
    models[horizon] = model

# Save models to Hopsworks Model Registry
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

print("\nSaving models to Hopsworks Model Registry...\n")

# Create a directory to save model artifacts
model_dir = "aqi_forecast_model"
os.makedirs(model_dir, exist_ok=True)

# Save all models in one pickle file
model_artifacts = {
    'models': models,
    'feature_cols': feature_cols,
    'forecast_horizons': forecast_horizons,
    'metadata': {
        'n_lags': n_lags,
        'train_samples': train_end,
        'val_samples': val_end - train_end,
        'test_samples': len(df) - val_end,
        'feature_count': len(feature_cols),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
}

joblib.dump(model_artifacts, f"{model_dir}/model.pkl")
print(f"Model artifacts saved to {model_dir}/model.pkl")

# Create input/output schema
input_schema = Schema([{"name": col, "type": "double"} for col in feature_cols])
output_schema = Schema([{"name": f"aqi_{h}h", "type": "double"} for h in forecast_horizons])
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

# Calculate average metrics across all horizons for model metadata
avg_metrics = {
    'avg_mae': np.mean([mean_absolute_error(
        df['aqi'].iloc[val_end:-h] if h < len(df) - val_end else df['aqi'].iloc[val_end:val_end+1],
        models[h].predict(X_val[:len(df['aqi'].iloc[val_end:-h]) if h < len(df) - val_end else 1])
    ) for h in forecast_horizons]),
}

# Get model registry
mr = project.get_model_registry()

# Create model in registry
aqi_model = mr.python.create_model(
    name="aqi_forecast_xgboost",
    metrics=avg_metrics,
    model_schema=model_schema,
    description="Multi-horizon XGBoost models for 72-hour AQI forecasting. Includes models for 1h, 6h, 12h, 24h, 48h, and 72h horizons."
)

# Save model to registry
aqi_model.save(model_dir)
print(f"Model saved to Hopsworks Model Registry: {aqi_model.name}, version {aqi_model.version}")