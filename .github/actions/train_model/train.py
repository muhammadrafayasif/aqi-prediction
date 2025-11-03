import hopsworks, joblib, os
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

load_dotenv()

# Connect to Hopsworks
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# Load your AQI feature group
feature_group = fs.get_or_create_feature_group(name="aqi_feature_pipeline", version=1)
df = feature_group.read()

print("✅ Data loaded from Hopsworks:", df.shape)

# Feature Engineering

# Ensure timestamp is datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Create AQI lag features (past 6 hours)
for lag in range(1, 7):
    df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

# Drop rows with missing lags
df = df.dropna().reset_index(drop=True)

# Extract time-based features
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek

# Define features and target
target_col = "aqi"
drop_cols = ["timestamp", target_col]

X = df.drop(columns=drop_cols)
y = df[target_col]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("Model training complete!")

# Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.3f}\n")

# Upload to Hopsworks Model Registry
mr = project.get_model_registry()
model_dir = "aqi_model_artifacts"
import os

os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, f"{model_dir}/model.pkl")
joblib.dump(scaler, f"{model_dir}/scaler.pkl")

# Save feature column names in the correct order
feature_columns = list(X.columns)  # same X used for training
joblib.dump(feature_columns, f"{model_dir}/feature_columns.pkl")
print("Saved feature column order for API predictions.")

model_meta = mr.python.create_model(
    name="aqi_forecast_model",
    metrics={"mae": mae, "mse": mse, "r2": r2},
    description="Random Forest AQI forecasting model with lag features"
)

model_meta.save(model_dir)
print("Model uploaded to Hopsworks successfully.")