import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings, hopsworks, os, joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# === 1. Load and prepare data ===
load_dotenv()

# Login to HopsWorks and get our feature store
project = hopsworks.login(
    api_key_value = os.getenv('API_KEY')
)
fs = project.get_feature_store()

# Get or create a feature group containing our features for AQI prediction
fg = fs.get_or_create_feature_group(
    name="aqi_feature_pipeline",
    version=1,
    primary_key=["timestamp"],
    description="A feature pipeline for storing AQI, environmental pollutants and weather data to the feature store."
)

df = fg.read()

print("Number of hours:", len(df))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# === 2. Fill the 5-day outage gap ===
full_range = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="H")
df = df.set_index("timestamp").reindex(full_range)
df.index.name = "timestamp"
df = df.ffill().bfill()

# === 3. Create lag features for time-series behavior ===
target_col = "aqi"  # target variable
num_lags = 6  # number of previous hours to use as predictors

for lag in range(1, num_lags + 1):
    df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

df = df.dropna()

feature_cols = [col for col in df.columns if col != target_col]
X = df[feature_cols]
y = df[target_col]

# === 4. Train-test split (chronological) ===
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

# === 5. Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. Train RandomForest model ===
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# === 7. Evaluate ===
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")

# === 8. Save to Model Registry ===
joblib.dump(model, "aqi_model.pkl")

mr = project.get_model_registry()


model_meta = mr.python.create_model(
    name="aqi_predictor",
    metrics={"mae": mae, "rmse": rmse},
    description="RandomForest model for predicting Air Quality Index based on pollutants and weather data."
)

# Save the model artifact to Hopsworks
model_meta.save("aqi_model.pkl")