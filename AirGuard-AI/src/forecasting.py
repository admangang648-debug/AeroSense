"""
================================================================================
AeroSense - Air Quality Machine Learning Forecasting Engine
================================================================================
Problem Statement: PS-1A (Air Quality Forecasting & Public Health Alert)

Key Steps in this Module:
1. Load historical dataset (data/air_quality.csv).
2. Chronological sorting and datetime parsing.
3. Feature Engineering:
   - Calendar / Time-of-day features (Hour, Day of Week, Sine/Cosine cycles)
   - Lag features for PM2.5 and PM10 (t-1, t-2, t-3, t-6, t-12, t-24)
   - Rolling window statistics (3h, 6h, 12h, 24h moving averages)
   - Meteorological variables (temperature, humidity, wind_speed, pressure, rainfall)
4. Chronological Train/Test Split (Preserving time-series order, NO shuffling).
5. Multi-target RandomForestRegressor model training for PM2.5 & PM10.
6. Model Evaluation (MAE, RMSE, R2 Score).
7. Recursive 24-Hour Horizon Future Forecasting.
8. Artifact Serialization:
   - Saved Model: models/air_quality_model.joblib
   - Metrics: models/evaluation_metrics.json
   - 24-Hour Forecast Results: data/forecast_24h.csv and data/forecast_24h.json
================================================================================
"""

import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------------------------
# 1. Feature Engineering Utilities
# ------------------------------------------------------------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs time-series features: temporal, lag, rolling window, and weather features.
    """
    df = df.copy()

    # Ensure timestamp is datetime and sorted chronologically
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # A. Time-based & Cyclical Features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month

    # Cyclical sine/cosine transformation for hour of day (ensures 23:00 is close to 00:00)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # B. Lag Features (Pollutant values from prior hours)
    lag_intervals = [1, 2, 3, 6, 12, 24]
    for target in ['PM2.5', 'PM10']:
        for lag in lag_intervals:
            df[f'{target}_lag_{lag}'] = df[target].shift(lag)

    # C. Rolling Statistics (Short-term and medium-term moving averages)
    rolling_windows = [3, 6, 12, 24]
    for target in ['PM2.5', 'PM10']:
        for window in rolling_windows:
            # Shift by 1 so the rolling window only uses strictly past observed values
            df[f'{target}_rolling_mean_{window}'] = (
                df[target].shift(1).rolling(window=window, min_periods=1).mean()
            )

    return df


def get_feature_columns():
    """
    Returns the exact list of feature columns used for model training and inference.
    """
    lag_cols = [f'{t}_lag_{l}' for t in ['PM2.5', 'PM10'] for l in [1, 2, 3, 6, 12, 24]]
    rolling_cols = [f'{t}_rolling_mean_{w}' for t in ['PM2.5', 'PM10'] for w in [3, 6, 12, 24]]
    weather_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 'rainfall']
    time_cols = ['hour', 'dayofweek', 'is_weekend', 'month', 'hour_sin', 'hour_cos']

    return weather_cols + time_cols + lag_cols + rolling_cols


# ------------------------------------------------------------------------------
# 2. Model Training & Evaluation Pipeline
# ------------------------------------------------------------------------------
def train_and_evaluate(data_path: str = None, models_dir: str = None):
    """
    Loads data, prepares features, splits chronologically, trains RandomForest,
    evaluates performance metrics, and saves the trained model.
    """
    # Resolve directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'air_quality.csv')
    if models_dir is None:
        models_dir = os.path.join(base_dir, 'models')

    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 STEP 1: Loading & Preprocessing Air Quality Data...")
    print("=" * 70)
    raw_df = pd.read_csv(data_path)
    print(f"Loaded {len(raw_df)} records from: {data_path}")

    # Build feature set
    featured_df = create_features(raw_df)

    # Drop early rows containing NaN values resulting from the 24-hour lag shift
    clean_df = featured_df.dropna().reset_index(drop=True)
    print(f"Cleaned dataset after lag/rolling creation: {len(clean_df)} records")

    feature_cols = get_feature_columns()
    target_cols = ['PM2.5', 'PM10']

    X = clean_df[feature_cols]
    y = clean_df[target_cols]

    # Chronological Train-Test Split (80% Train, 20% Test) - strictly NO random shuffle!
    split_index = int(len(clean_df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_timestamps = clean_df['timestamp'].iloc[split_index:]

    print(f"\n📊 Chronological Split (No random shuffling):")
    print(f"   • Training samples : {len(X_train)} ({clean_df['timestamp'].iloc[0]} to {clean_df['timestamp'].iloc[split_index - 1]})")
    print(f"   • Testing samples  : {len(X_test)} ({test_timestamps.iloc[0]} to {test_timestamps.iloc[-1]})")

    print("\n🌲 STEP 2: Training RandomForestRegressor Model...")
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Model training completed.")

    # Model Evaluation on Test Set
    print("\n📈 STEP 3: Evaluating Performance on Test Data...")
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=['PM2.5_pred', 'PM10_pred'], index=y_test.index)

    # Calculate metrics
    mae_pm25 = mean_absolute_error(y_test['PM2.5'], y_pred_df['PM2.5_pred'])
    rmse_pm25 = np.sqrt(mean_squared_error(y_test['PM2.5'], y_pred_df['PM2.5_pred']))
    r2_pm25 = r2_score(y_test['PM2.5'], y_pred_df['PM2.5_pred'])

    mae_pm10 = mean_absolute_error(y_test['PM10'], y_pred_df['PM10_pred'])
    rmse_pm10 = np.sqrt(mean_squared_error(y_test['PM10'], y_pred_df['PM10_pred']))
    r2_pm10 = r2_score(y_test['PM10'], y_pred_df['PM10_pred'])

    metrics = {
        "model_type": "RandomForestRegressor (Multi-target)",
        "features_used": feature_cols,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "PM2.5": {
                "MAE": round(float(mae_pm25), 3),
                "RMSE": round(float(rmse_pm25), 3),
                "R2": round(float(r2_pm25), 4)
            },
            "PM10": {
                "MAE": round(float(mae_pm10), 3),
                "RMSE": round(float(rmse_pm10), 3),
                "R2": round(float(r2_pm10), 4)
            }
        }
    }

    print("-" * 50)
    print(f"⭐ PM2.5 Forecast Performance:")
    print(f"   • MAE  : {mae_pm25:.3f} µg/m³")
    print(f"   • RMSE : {rmse_pm25:.3f} µg/m³")
    print(f"   • R²   : {r2_pm25:.4f}")
    print(f"⭐ PM10 Forecast Performance:")
    print(f"   • MAE  : {mae_pm10:.3f} µg/m³")
    print(f"   • RMSE : {rmse_pm10:.3f} µg/m³")
    print(f"   • R²   : {r2_pm10:.4f}")
    print("-" * 50)

    # Save Model & Metrics
    model_file = os.path.join(models_dir, 'air_quality_model.joblib')
    metrics_file = os.path.join(models_dir, 'evaluation_metrics.json')

    joblib.dump(model, model_file)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"💾 Model saved to   : {model_file}")
    print(f"💾 Metrics saved to : {metrics_file}")

    return model, clean_df, feature_cols, metrics


# ------------------------------------------------------------------------------
# 3. 24-Hour Horizon Future Forecasting
# ------------------------------------------------------------------------------
def generate_24h_forecast(model, historical_df: pd.DataFrame, feature_cols: list, output_dir: str = None):
    """
    Performs recursive multi-step forecasting for the next 24 hours starting
    from the last recorded timestamp.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'data')

    os.makedirs(output_dir, exist_ok=True)

    print("\n🔮 STEP 4: Generating Next 24-Hour Future Forecast...")

    # Working buffer containing historical records to dynamically compute lags & rolling averages
    working_df = historical_df[['timestamp', 'PM2.5', 'PM10', 'temperature', 'humidity', 'wind_speed', 'pressure', 'rainfall']].copy()
    working_df['timestamp'] = pd.to_datetime(working_df['timestamp'])

    last_ts = working_df['timestamp'].iloc[-1]
    forecast_results = []

    # Weather base values for future horizon (with realistic diurnal modulation)
    last_temp = working_df['temperature'].iloc[-1]
    last_humidity = working_df['humidity'].iloc[-1]
    last_wind = working_df['wind_speed'].iloc[-1]
    last_press = working_df['pressure'].iloc[-1]

    for step in range(1, 25):
        future_ts = last_ts + timedelta(hours=step)
        hour = future_ts.hour

        # Forecasted weather estimates for step hour
        future_temp = last_temp + 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        future_humidity = np.clip(last_humidity - 1.2 * (future_temp - last_temp), 25.0, 95.0)
        future_wind = np.clip(last_wind + 0.8 * np.sin(2 * np.pi * (hour - 12) / 24.0), 1.0, 12.0)
        future_pressure = last_press + 0.5 * np.cos(2 * np.pi * hour / 24.0)
        future_rainfall = 0.0  # Assumed dry baseline for forecast horizon

        # Dynamically calculate features on the updated historical + predicted sequence
        temp_df = create_features(working_df)
        latest_row = temp_df.iloc[-1]

        # Extract features for prediction
        input_data = {}
        for col in feature_cols:
            if col == 'temperature':
                input_data[col] = future_temp
            elif col == 'humidity':
                input_data[col] = future_humidity
            elif col == 'wind_speed':
                input_data[col] = future_wind
            elif col == 'pressure':
                input_data[col] = future_pressure
            elif col == 'rainfall':
                input_data[col] = future_rainfall
            elif col == 'hour':
                input_data[col] = hour
            elif col == 'dayofweek':
                input_data[col] = future_ts.dayofweek
            elif col == 'is_weekend':
                input_data[col] = 1 if future_ts.dayofweek in [5, 6] else 0
            elif col == 'month':
                input_data[col] = future_ts.month
            elif col == 'hour_sin':
                input_data[col] = np.sin(2 * np.pi * hour / 24.0)
            elif col == 'hour_cos':
                input_data[col] = np.cos(2 * np.pi * hour / 24.0)
            else:
                # Lag & rolling features come directly from the latest state in the buffer
                input_data[col] = latest_row[col]

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        pred_pm25 = max(1.0, round(float(prediction[0]), 2))
        pred_pm10 = max(pred_pm25 + 1.0, round(float(prediction[1]), 2))

        # Categorize AQI Status based on predicted PM2.5 levels (Standard AQI Bands)
        if pred_pm25 <= 30:
            aqi_category = "Good"
            alert_level = "Low Risk"
        elif pred_pm25 <= 60:
            aqi_category = "Satisfactory / Moderate"
            alert_level = "Moderate"
        elif pred_pm25 <= 90:
            aqi_category = "Poor / Sensitive Warning"
            alert_level = "Unhealthy for Sensitive Groups"
        elif pred_pm25 <= 150:
            aqi_category = "Very Poor"
            alert_level = "High Alert"
        else:
            aqi_category = "Severe / Hazardous"
            alert_level = "Severe Emergency"

        forecast_record = {
            "forecast_hour": step,
            "timestamp": future_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_PM2.5": pred_pm25,
            "predicted_PM10": pred_pm10,
            "temperature": round(float(future_temp), 2),
            "humidity": round(float(future_humidity), 2),
            "wind_speed": round(float(future_wind), 2),
            "pressure": round(float(future_pressure), 2),
            "rainfall": round(float(future_rainfall), 2),
            "aqi_category": aqi_category,
            "alert_level": alert_level
        }
        forecast_results.append(forecast_record)

        # Append new prediction to working buffer so subsequent steps use it as lag
        new_row = pd.DataFrame([{
            'timestamp': future_ts,
            'PM2.5': pred_pm25,
            'PM10': pred_pm10,
            'temperature': future_temp,
            'humidity': future_humidity,
            'wind_speed': future_wind,
            'pressure': future_pressure,
            'rainfall': future_rainfall
        }])
        working_df = pd.concat([working_df, new_row], ignore_index=True)

    forecast_df = pd.DataFrame(forecast_results)

    # Save to CSV and JSON for easy integration with Flask backend & dashboards
    forecast_csv = os.path.join(output_dir, 'forecast_24h.csv')
    forecast_json = os.path.join(output_dir, 'forecast_24h.json')

    forecast_df.to_csv(forecast_csv, index=False)
    with open(forecast_json, 'w') as f:
        json.dump(forecast_results, f, indent=4)

    print(f"💾 24h Forecast CSV  saved to : {forecast_csv}")
    print(f"💾 24h Forecast JSON saved to : {forecast_json}")
    print("\nPreview of 24-Hour Forecast:")
    print(forecast_df[['timestamp', 'predicted_PM2.5', 'predicted_PM10', 'aqi_category', 'alert_level']].head(6))
    print("=" * 70)

    return forecast_df


# ------------------------------------------------------------------------------
# 4. Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Train model, compute metrics, and save artifacts
    trained_model, clean_dataset, features, eval_metrics = train_and_evaluate()

    # Generate next 24-hour forecasts and health alert classifications
    generate_24h_forecast(trained_model, clean_dataset, features)
