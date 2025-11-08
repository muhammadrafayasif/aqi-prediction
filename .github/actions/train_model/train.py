import joblib
import hopsworks, os
import pandas as pd
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("API_KEY"))
fs = project.get_feature_store()
fg = fs.get_or_create_feature_group("aqi_feature_pipeline", version=1)
df = fg.read()
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

# Handle missing timestamps
full_index = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='H')
df = df.set_index('timestamp').reindex(full_index).rename_axis('timestamp').reset_index()

# Columns
weather_cols = ['temp','wind_speed','wind_gusts','humidity_percent','dew_point','pressure',
                'cloud_cover','visibility']
pollutant_cols = ['pm_10','pm_2_5','no_2','o_3','so_2','co','aqi']

# Interpolate weather features
df[weather_cols] = df[weather_cols].interpolate(method='linear')

# Fill pollutant features with forward fill
df[pollutant_cols] = df[pollutant_cols].fillna(method='ffill')

# Create lag features
n_lags = 24  # past 24 hours
for col in pollutant_cols + weather_cols:
    for lag in range(1, n_lags+1):
        df[f"{col}_lag{lag}"] = df[col].shift(lag)


# Create rolling features
for col in pollutant_cols + weather_cols:
    df[f"{col}_roll3"] = df[col].rolling(window=3).mean()
    df[f"{col}_roll6"] = df[col].rolling(window=6).mean()

# Add time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

# Drop rows with NaNs from lags
df = df.dropna().reset_index(drop=True)

# Prepare features and targets
forecast_horizon = 72  # 72-hour prediction

feature_cols = [col for col in df.columns if col not in ['timestamp','aqi']]

X = []
y = []

for i in range(len(df) - forecast_horizon):
    X.append(df[feature_cols].iloc[i].values)
    y.append(df['aqi'].iloc[i:i+forecast_horizon].values)

X = np.array(X)
y = np.array(y)

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train XGBoost MultiOutputRegressor
xgb_reg = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective='reg:squarederror',
    random_state=42
)

model = MultiOutputRegressor(xgb_reg)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Upload to Hopsworks Model Registry
joblib.dump(model, "xgboost_model.pkl")
mr = project.get_model_registry()

model_meta = mr.python.create_model(
    name="xgboost_aqi_forecast_model",
    metrics={"mae": mae, "mse": mse},
    description="XGBoost MultiOutputRegressor AQI forecasting model with lag features"
)
model_meta.save('xgboost_model.pkl')

print("Model uploaded to Hopsworks successfully.")