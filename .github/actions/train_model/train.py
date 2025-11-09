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

# Fill pollutants using forward fill + rolling mean
df[pollutant_cols] = df[pollutant_cols].ffill().rolling(3, min_periods=1).mean()

# Detect gaps
df['time_diff'] = df['timestamp'].diff().dt.total_seconds().div(3600)
df['gap_flag'] = (df['time_diff'] > 1).astype(int)
df['gap_flag'] = df['gap_flag'].fillna(0)

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

# Add month and season for distribution shift handling
df['month'] = df['timestamp'].dt.month
df['season'] = df['timestamp'].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

# Drop rows with NaNs in lags
df = df.dropna().reset_index(drop=True)
print(f"Data shape after feature engineering: {df.shape}")

# Use adaptive splits based on data size
n_samples = len(df)

# Calculate total days of data available
total_hours = n_samples
total_days = total_hours / 24

# Adaptive split strategy based on data size
# The dataset will use a different algorithm based on its size
# As of now (9/11/2025), the model will use proportianal time splits
if total_days < 90:
    # Small dataset - use proportional time splits
    test_days = max(3, int(total_days * 0.15))  # 15% or 3 days minimum
    val_days = max(3, int(total_days * 0.15))   # 15% or 3 days minimum
    
    HOURS_FOR_TEST = test_days * 24
    HOURS_FOR_VAL = val_days * 24
    
    test_start_idx = n_samples - HOURS_FOR_TEST
    val_start_idx = test_start_idx - HOURS_FOR_VAL
    train_end_idx = val_start_idx
    train_start_idx = 0
else:
    # Large dataset - use fixed time-based splits
    HOURS_FOR_TEST = 24 * 30   # Last 30 days for testing
    HOURS_FOR_VAL = 24 * 30    # 30 days before test for validation
    
    test_start_idx = n_samples - HOURS_FOR_TEST
    val_start_idx = test_start_idx - HOURS_FOR_VAL
    train_end_idx = val_start_idx
    
    # Use rolling window for very large datasets
    MAX_TRAIN_MONTHS = 12
    max_train_samples = MAX_TRAIN_MONTHS * 30 * 24
    
    if train_end_idx > max_train_samples:
        train_start_idx = train_end_idx - max_train_samples
    else:
        train_start_idx = 0

# Validate splits
if train_end_idx <= train_start_idx:
    raise ValueError(f"Not enough data for training! Need at least 2 days of data. Current: {total_days:.1f} days")
if val_start_idx >= test_start_idx:
    raise ValueError(f"Not enough data for validation! Need at least {total_days/7:.1f} more days of data.")
if test_start_idx >= n_samples:
    raise ValueError(f"Not enough data for testing! Need at least {total_days/7:.1f} more days of data.")

# Train separate models for different horizons
forecast_horizons = [1, 6, 12, 24, 48, 72]
models = {}
metrics_history = {}

feature_cols = [col for col in df.columns if col not in ['timestamp', 'aqi', 'time_diff']]

print(f"\nTraining models for different horizons...")
for horizon in forecast_horizons:
    # Create dataset for this horizon
    X_list, y_list = [], []
    for i in range(len(df) - horizon):
        X_list.append(df[feature_cols].iloc[i].values)
        y_list.append(df['aqi'].iloc[i + horizon])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Apply rolling window split
    X_train = X[train_start_idx:train_end_idx]
    y_train = y[train_start_idx:train_end_idx]
    X_val = X[val_start_idx:test_start_idx]
    y_val = y[val_start_idx:test_start_idx]
    X_test = X[test_start_idx:]
    y_test = y[test_start_idx:]
    
    # Validate that we have data
    if len(X_train) == 0:
        raise ValueError(f"Training set is empty for {horizon}h horizon!")
    if len(X_val) == 0:
        print(f"  WARNING: Validation set is empty for {horizon}h horizon. Skipping validation.")
        use_validation = False
    else:
        use_validation = True
    if len(X_test) == 0:
        print(f"  WARNING: Test set is empty for {horizon}h horizon. Skipping testing.")
        use_test = False
    else:
        use_test = True
    
    # Add sample weights to emphasize recent data ======
    # More recent data gets higher weight (only if enough training data)
    if len(y_train) > 24:  # Only use weights if we have at least 24 hours of data
        train_indices = np.arange(len(y_train))
        sample_weights = np.exp(train_indices / len(train_indices)) / np.e
        sample_weights = sample_weights / sample_weights.mean()
    else:
        sample_weights = None
    
    # Tuned XGBRegressor
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=30 if use_validation else None
    )
    
    # Prepare eval_set
    eval_set = [(X_val, y_val)] if use_validation else None
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=False
    )
    
    # Comprehensive evaluation
    if use_validation:
        y_pred_val = model.predict(X_val)
        y_pred_val = np.clip(y_pred_val, 0, 500)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    else:
        val_mae = None
        val_rmse = None
    
    if use_test:
        y_pred_test = model.predict(X_test)
        y_pred_test = np.clip(y_pred_test, 0, 500)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    else:
        test_mae = None
        test_rmse = None
    
    # Store metrics for comparison
    metrics_history[horizon] = {
        'val_mae': val_mae if val_mae is not None else float('nan'),
        'val_rmse': val_rmse if val_rmse is not None else float('nan'),
        'test_mae': test_mae if test_mae is not None else float('nan'),
        'test_rmse': test_rmse if test_rmse is not None else float('nan'),
        'n_train_samples': len(y_train),
        'n_val_samples': len(y_val) if use_validation else 0,
        'n_test_samples': len(y_test) if use_test else 0,
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None
    }
    
    models[horizon] = model

# Save models to Hopsworks Model Registry
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

print("\nSaving models to Hopsworks Model Registry...")

model_dir = "aqi_forecast_model"
os.makedirs(model_dir, exist_ok=True)

model_artifacts = {
    'models': models,
    'feature_cols': feature_cols,
    'forecast_horizons': forecast_horizons,
    'metrics_history': metrics_history,
    'metadata': {
        'n_lags': n_lags,
        'train_start_date': df['timestamp'].iloc[train_start_idx].strftime('%Y-%m-%d %H:%M:%S'),
        'train_end_date': df['timestamp'].iloc[train_end_idx-1].strftime('%Y-%m-%d %H:%M:%S'),
        'val_start_date': df['timestamp'].iloc[val_start_idx].strftime('%Y-%m-%d %H:%M:%S'),
        'test_start_date': df['timestamp'].iloc[test_start_idx].strftime('%Y-%m-%d %H:%M:%S'),
        'train_samples': train_end_idx - train_start_idx,
        'val_samples': test_start_idx - val_start_idx,
        'test_samples': n_samples - test_start_idx,
        'feature_count': len(feature_cols),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
}

joblib.dump(model_artifacts, f"{model_dir}/model.pkl")
print(f"Model artifacts saved locally")

# Create input/output schema
input_schema = Schema([{"name": col, "type": "double"} for col in feature_cols])
output_schema = Schema([{"name": f"aqi_{h}h", "type": "double"} for h in forecast_horizons])
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

# Calculate average metrics across all horizons for model metadata
test_maes = [m['test_mae'] for m in metrics_history.values() if not np.isnan(m['test_mae'])]
test_rmses = [m['test_rmse'] for m in metrics_history.values() if not np.isnan(m['test_rmse'])]

# Store dates for description (not metrics - those must be numeric)
train_start_date = df['timestamp'].iloc[train_start_idx].strftime('%Y-%m-%d')
train_end_date = df['timestamp'].iloc[train_end_idx-1].strftime('%Y-%m-%d')

# Only numeric values allowed in metrics
avg_metrics = {
    'avg_test_mae': float(np.mean(test_maes)) if test_maes else 0.0,
    'avg_test_rmse': float(np.mean(test_rmses)) if test_rmses else 0.0,
    'train_samples': int(train_end_idx - train_start_idx),
    'train_days': float((train_end_idx - train_start_idx) / 24),
    'val_samples': int(test_start_idx - val_start_idx),
    'test_samples': int(n_samples - test_start_idx),
}

# Get model registry
mr = project.get_model_registry()

# Create model in registry
aqi_model = mr.python.create_model(
    name="aqi_forecast_xgboost",
    metrics=avg_metrics,
    model_schema=model_schema,
    description=f"Multi-horizon XGBoost models for 72-hour AQI forecasting. Trained on {avg_metrics['train_days']:.1f} days from {train_start_date} to {train_end_date}."
)

aqi_model.save(model_dir)
print(f"\nModel saved to Hopsworks Model Registry")