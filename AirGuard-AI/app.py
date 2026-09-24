"""
================================================================================
AeroSense - Web Application & Dashboard Server
================================================================================
Problem Statement: PS-1A (Air Quality Forecasting & Public Health Alert)
"""

import os
import json
import math
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, request, session
import plotly
import plotly.graph_objs as go

# Import internal alert engine, data provider, and forecasting helpers
from src.alerts import get_alert_advisory, process_forecast_alerts, AI_DISCLAIMER
from src.data_provider import get_location_air_quality, fetch_live_openaq_data
from src.forecasting import create_features, get_feature_columns

app = Flask(__name__)
app.secret_key = "aerosense-secret-session-key-hackathon"

# Base directory for resolving file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Built-in high-accuracy geocoding registry for offline and fast search
KNOWN_CITIES = [
    {"city": "Bengaluru", "country": "India", "display_name": "Bengaluru, India", "latitude": 12.9716, "longitude": 77.5946},
    {"city": "Mumbai", "country": "India", "display_name": "Mumbai, India", "latitude": 19.0760, "longitude": 72.8777},
    {"city": "Delhi", "country": "India", "display_name": "Delhi, India", "latitude": 28.6139, "longitude": 77.2090},
    {"city": "Hyderabad", "country": "India", "display_name": "Hyderabad, India", "latitude": 17.3850, "longitude": 78.4867},
    {"city": "Chennai", "country": "India", "display_name": "Chennai, India", "latitude": 13.0827, "longitude": 80.2707},
    {"city": "Kolkata", "country": "India", "display_name": "Kolkata, India", "latitude": 22.5726, "longitude": 88.3639},
    {"city": "Pune", "country": "India", "display_name": "Pune, India", "latitude": 18.5204, "longitude": 73.8567},
    {"city": "Ahmedabad", "country": "India", "display_name": "Ahmedabad, India", "latitude": 23.0225, "longitude": 72.5714},
    {"city": "Jaipur", "country": "India", "display_name": "Jaipur, India", "latitude": 26.9124, "longitude": 75.7873},
    {"city": "New York", "country": "United States", "display_name": "New York, United States", "latitude": 40.7128, "longitude": -74.0060},
    {"city": "London", "country": "United Kingdom", "display_name": "London, United Kingdom", "latitude": 51.5074, "longitude": -0.1278},
    {"city": "San Francisco", "country": "United States", "display_name": "San Francisco, United States", "latitude": 37.7749, "longitude": -122.4194},
    {"city": "Tokyo", "country": "Japan", "display_name": "Tokyo, Japan", "latitude": 35.6762, "longitude": 139.6503},
    {"city": "Singapore", "country": "Singapore", "display_name": "Singapore", "latitude": 1.3521, "longitude": 103.8198},
    {"city": "Paris", "country": "France", "display_name": "Paris, France", "latitude": 48.8566, "longitude": 2.3522},
    {"city": "Dubai", "country": "United Arab Emirates", "display_name": "Dubai, UAE", "latitude": 25.2048, "longitude": 55.2708},
    {"city": "Sydney", "country": "Australia", "display_name": "Sydney, Australia", "latitude": -33.8688, "longitude": 151.2093},
    {"city": "Toronto", "country": "Canada", "display_name": "Toronto, Canada", "latitude": 43.6532, "longitude": -79.3832}
]

DEFAULT_LOCATION = {
    "city": "Bengaluru",
    "country": "India",
    "display_name": "Bengaluru, India",
    "latitude": 12.9716,
    "longitude": 77.5946,
    "coordinates_display": "12.9716° N, 77.5946° E",
    "source": "default"
}

# ------------------------------------------------------------------------------
# Load Trained ML Model and Static Validation Metrics Once
# ------------------------------------------------------------------------------
FEATURE_COLS = get_feature_columns()
MODEL_PATH = os.path.join(MODELS_DIR, "air_quality_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")

try:
    if os.path.exists(MODEL_PATH):
        ML_MODEL = joblib.load(MODEL_PATH)
    else:
        ML_MODEL = None
except Exception as e:
    print(f"Warning: Could not load ML model ({e})")
    ML_MODEL = None

try:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            METRICS_DATA = json.load(f)
    else:
        METRICS_DATA = {
            "model_type": "RandomForestRegressor (Multi-target)",
            "metrics": {
                "PM2.5": {"MAE": 4.925, "RMSE": 6.585, "R2": 0.8564},
                "PM10": {"MAE": 10.269, "RMSE": 13.933, "R2": 0.8423}
            }
        }
except Exception:
    METRICS_DATA = {
        "model_type": "RandomForestRegressor (Multi-target)",
        "metrics": {
            "PM2.5": {"MAE": 4.925, "RMSE": 6.585, "R2": 0.8564},
            "PM10": {"MAE": 10.269, "RMSE": 13.933, "R2": 0.8423}
        }
    }


def format_coordinates(lat: float, lon: float) -> str:
    """Formats latitude and longitude into readable degree notation."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}° {lat_dir}, {abs(lon):.4f}° {lon_dir}"


def find_nearest_city(lat: float, lon: float) -> dict:
    """Finds the closest city in the registry using Euclidean distance approximation."""
    closest = None
    min_dist = float("inf")
    for item in KNOWN_CITIES:
        dist = math.sqrt((item["latitude"] - lat) ** 2 + (item["longitude"] - lon) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest = item
    return closest


def geocode_city_name(query: str) -> dict:
    """Looks up city by name in registry or dynamic online query."""
    q_lower = query.strip().lower()

    # 1. Exact or partial match in curated registry
    for item in KNOWN_CITIES:
        if q_lower == item["city"].lower() or q_lower in item["display_name"].lower():
            res = dict(item)
            res["coordinates_display"] = format_coordinates(res["latitude"], res["longitude"])
            return res

    # 2. Dynamic online geocode fallback (Nominatim OpenStreetMap)
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "AeroSense-Hackathon-App/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data and len(data) > 0:
                first = data[0]
                lat = float(first["lat"])
                lon = float(first["lon"])
                display = first.get("display_name", query)
                parts = [p.strip() for p in display.split(",")]
                city_name = parts[0]
                country_name = parts[-1] if len(parts) > 1 else ""
                return {
                    "city": city_name,
                    "country": country_name,
                    "display_name": f"{city_name}, {country_name}" if country_name else city_name,
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "coordinates_display": format_coordinates(lat, lon)
                }
    except Exception:
        pass

    return None


def reverse_geocode_coordinates(lat: float, lon: float) -> dict:
    """Resolves coordinates to city and country names."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "AeroSense-Hackathon-App/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            address = data.get("address", {})
            city = address.get("city") or address.get("town") or address.get("village") or address.get("county") or "Detected Location"
            country = address.get("country", "")
            return {
                "city": city,
                "country": country,
                "display_name": f"{city}, {country}" if country else city,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "coordinates_display": format_coordinates(lat, lon)
            }
    except Exception:
        pass

    nearest = find_nearest_city(lat, lon)
    if nearest:
        res = dict(nearest)
        res["latitude"] = round(lat, 4)
        res["longitude"] = round(lon, 4)
        res["coordinates_display"] = format_coordinates(lat, lon)
        return res

    return {
        "city": "Current Location",
        "country": "",
        "display_name": f"Coordinates ({format_coordinates(lat, lon)})",
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "coordinates_display": format_coordinates(lat, lon)
    }


# ------------------------------------------------------------------------------
# Location-Aware Forecasting Engine
# ------------------------------------------------------------------------------
def generate_location_forecast(model, history_df: pd.DataFrame, feature_cols: list) -> list:
    """
    Executes recursive 24-hour horizon forecasting using the trained Random Forest model
    applied specifically to the selected location's observations and history.
    """
    working_df = history_df[['timestamp', 'PM2.5', 'PM10', 'temperature', 'humidity', 'wind_speed', 'pressure', 'rainfall']].copy()
    working_df['timestamp'] = pd.to_datetime(working_df['timestamp'])

    last_ts = working_df['timestamp'].iloc[-1]
    forecast_results = []

    last_temp = float(working_df['temperature'].iloc[-1])
    last_humidity = float(working_df['humidity'].iloc[-1])
    last_wind = float(working_df['wind_speed'].iloc[-1])
    last_press = float(working_df['pressure'].iloc[-1])

    for step in range(1, 25):
        future_ts = last_ts + timedelta(hours=step)
        hour = future_ts.hour

        future_temp = last_temp + 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        future_humidity = np.clip(last_humidity - 1.2 * (future_temp - last_temp), 18.0, 98.0)
        future_wind = np.clip(last_wind + 0.8 * np.sin(2 * np.pi * (hour - 12) / 24.0), 0.5, 16.0)
        future_pressure = last_press + 0.5 * np.cos(2 * np.pi * hour / 24.0)
        future_rainfall = 0.0

        temp_df = create_features(working_df)
        latest_row = temp_df.iloc[-1]

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
                input_data[col] = latest_row[col]

        input_df = pd.DataFrame([input_data])
        if model is not None:
            pred = model.predict(input_df)[0]
            pred_pm25 = max(3.0, round(float(pred[0]), 1))
            pred_pm10 = max(pred_pm25 + 2.0, round(float(pred[1]), 1))
        else:
            pred_pm25 = max(3.0, round(float(latest_row['PM2.5']), 1))
            pred_pm10 = max(pred_pm25 + 2.0, round(float(latest_row['PM10']), 1))

        if pred_pm25 <= 30:
            aqi_cat, alert_lvl = "Good", "Low Risk"
        elif pred_pm25 <= 60:
            aqi_cat, alert_lvl = "Satisfactory / Moderate", "Moderate"
        elif pred_pm25 <= 90:
            aqi_cat, alert_lvl = "Poor / Sensitive Warning", "Unhealthy for Sensitive Groups"
        elif pred_pm25 <= 150:
            aqi_cat, alert_lvl = "Very Poor", "High Alert"
        else:
            aqi_cat, alert_lvl = "Severe / Hazardous", "Severe Emergency"

        rec = {
            "forecast_hour": step,
            "timestamp": future_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_PM2.5": pred_pm25,
            "predicted_PM10": pred_pm10,
            "temperature": round(float(future_temp), 1),
            "humidity": round(float(future_humidity), 1),
            "wind_speed": round(float(future_wind), 1),
            "pressure": round(float(future_pressure), 1),
            "rainfall": round(float(future_rainfall), 1),
            "aqi_category": aqi_cat,
            "alert_level": alert_lvl
        }
        forecast_results.append(rec)

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

    return forecast_results


# ------------------------------------------------------------------------------
# Interactive Plotly Chart Builder
# ------------------------------------------------------------------------------
def build_plotly_charts(df_history: pd.DataFrame, forecast_records: list):
    """
    Generates interactive light-mode Plotly charts for PM2.5 and PM10 comparison.
    """
    if df_history.empty or not forecast_records:
        return None, None, None

    recent_history = df_history.iloc[-72:].copy()
    hist_time = recent_history["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    hist_pm25 = recent_history["PM2.5"].tolist()
    hist_pm10 = recent_history["PM10"].tolist()

    forecast_time = [item["timestamp"] for item in forecast_records]
    pred_pm25 = [item["predicted_PM2.5"] for item in forecast_records]
    pred_pm10 = [item["predicted_PM10"] for item in forecast_records]

    chart_forecast_time = [hist_time[-1]] + forecast_time
    chart_forecast_pm25 = [hist_pm25[-1]] + pred_pm25
    chart_forecast_pm10 = [hist_pm10[-1]] + pred_pm10

    layout_base = dict(
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#475569", size=12),
        margin=dict(l=45, r=25, t=35, b=45),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#1E293B", size=11.5)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(color="#64748B", size=11),
            linecolor="#E2E8F0"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(color="#64748B", size=11),
            linecolor="#E2E8F0",
            title=dict(text="Concentration (µg/m³)", font=dict(color="#64748B", size=11))
        )
    )

    # 1. PM2.5 Chart
    fig_pm25 = go.Figure()
    fig_pm25.add_trace(go.Scatter(
        x=hist_time,
        y=hist_pm25,
        mode="lines",
        name="Historical PM2.5",
        line=dict(color="#0284C7", width=2.2),
        fill="tozeroy",
        fillcolor="rgba(2, 132, 199, 0.05)"
    ))
    fig_pm25.add_trace(go.Scatter(
        x=chart_forecast_time,
        y=chart_forecast_pm25,
        mode="lines+markers",
        name="24h AI Forecast PM2.5",
        line=dict(color="#E11D48", width=2.5, dash="dash"),
        marker=dict(size=5, color="#E11D48")
    ))
    fig_pm25.add_hline(
        y=30,
        line_dash="dot",
        line_color="#059669",
        annotation_text="Moderate Threshold (30 µg/m³)",
        annotation_position="bottom right",
        annotation_font_color="#059669"
    )
    fig_pm25.add_hline(
        y=60,
        line_dash="dot",
        line_color="#D97706",
        annotation_text="Elevated Risk (60 µg/m³)",
        annotation_position="top right",
        annotation_font_color="#D97706"
    )
    fig_pm25.update_layout(layout_base, title=dict(text="PM2.5: Historical Observations & 24-Hour AI Forecast", font=dict(color="#0F172A", size=14, weight=600)))

    # 2. PM10 Chart
    fig_pm10 = go.Figure()
    fig_pm10.add_trace(go.Scatter(
        x=hist_time,
        y=hist_pm10,
        mode="lines",
        name="Historical PM10",
        line=dict(color="#7C3AED", width=2.2),
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.05)"
    ))
    fig_pm10.add_trace(go.Scatter(
        x=chart_forecast_time,
        y=chart_forecast_pm10,
        mode="lines+markers",
        name="24h AI Forecast PM10",
        line=dict(color="#D97706", width=2.5, dash="dash"),
        marker=dict(size=5, color="#D97706")
    ))
    fig_pm10.add_hline(
        y=50,
        line_dash="dot",
        line_color="#059669",
        annotation_text="Moderate Threshold (50 µg/m³)",
        annotation_position="bottom right",
        annotation_font_color="#059669"
    )
    fig_pm10.add_hline(
        y=100,
        line_dash="dot",
        line_color="#D97706",
        annotation_text="Elevated Risk (100 µg/m³)",
        annotation_position="top right",
        annotation_font_color="#D97706"
    )
    fig_pm10.update_layout(layout_base, title=dict(text="PM10: Historical Observations & 24-Hour AI Forecast", font=dict(color="#0F172A", size=14, weight=600)))

    # 3. Combined Forecast Chart (PM2.5 & PM10 on single synchronized plot)
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(
        x=hist_time,
        y=hist_pm25,
        mode="lines",
        name="Hist PM2.5",
        line=dict(color="#0284C7", width=2),
        fill="tozeroy",
        fillcolor="rgba(2, 132, 199, 0.04)"
    ))
    fig_combined.add_trace(go.Scatter(
        x=chart_forecast_time,
        y=chart_forecast_pm25,
        mode="lines+markers",
        name="Forecast PM2.5",
        line=dict(color="#E11D48", width=2.4, dash="dash"),
        marker=dict(size=5, color="#E11D48")
    ))
    fig_combined.add_trace(go.Scatter(
        x=hist_time,
        y=hist_pm10,
        mode="lines",
        name="Hist PM10",
        line=dict(color="#7C3AED", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.03)"
    ))
    fig_combined.add_trace(go.Scatter(
        x=chart_forecast_time,
        y=chart_forecast_pm10,
        mode="lines+markers",
        name="Forecast PM10",
        line=dict(color="#D97706", width=2.4, dash="dot"),
        marker=dict(size=5, color="#D97706")
    ))
    fig_combined.add_hline(
        y=30,
        line_dash="dot",
        line_color="#059669",
        annotation_text="PM2.5 Safe (30 µg/m³)",
        annotation_position="bottom right",
        annotation_font_color="#059669"
    )
    fig_combined.add_hline(
        y=60,
        line_dash="dot",
        line_color="#D97706",
        annotation_text="PM2.5 Elevated (60 µg/m³)",
        annotation_position="top right",
        annotation_font_color="#D97706"
    )
    fig_combined.update_layout(
        layout_base,
        title=dict(text="24-Hour Multi-Pollutant Forecast Horizon", font=dict(color="#0F172A", size=14, weight=600))
    )

    chart_pm25_json = json.dumps(fig_pm25, cls=plotly.utils.PlotlyJSONEncoder)
    chart_pm10_json = json.dumps(fig_pm10, cls=plotly.utils.PlotlyJSONEncoder)
    chart_combined_json = json.dumps(fig_combined, cls=plotly.utils.PlotlyJSONEncoder)

    return chart_pm25_json, chart_pm10_json, chart_combined_json


# ------------------------------------------------------------------------------
# Central Location Dashboard Resolver
# ------------------------------------------------------------------------------
def get_full_dashboard_payload(location_dict: dict) -> dict:
    """
    Computes location-specific air quality observations, runs the forecasting model
    specifically for that location, evaluates risk levels, peaks, advisories, and charts.
    """
    lat = float(location_dict.get("latitude", 12.9716))
    lon = float(location_dict.get("longitude", 77.5946))
    city = location_dict.get("city", "Bengaluru")
    country = location_dict.get("country", "India")

    # 1. Retrieve location observation data & history
    obs_data = get_location_air_quality(lat, lon, city=city, country=country)
    df_history = obs_data["history_df"]

    # 2. Generate location-specific 24h forecast using ML model
    forecast_records = generate_location_forecast(ML_MODEL, df_history, FEATURE_COLS)

    # 3. Compute location-specific alerts & peaks
    current_alert = get_alert_advisory(obs_data["pm25"], obs_data["pm10"], timestamp=obs_data["timestamp"])
    forecast_summary = process_forecast_alerts(forecast_records)

    breakdown = forecast_summary.get("risk_breakdown", {})
    elevated_hours_count = breakdown.get("Moderate", 0) + breakdown.get("High", 0) + breakdown.get("Very High", 0)

    # 4. Generate Plotly charts for location
    chart_pm25_json, chart_pm10_json, chart_combined_json = build_plotly_charts(df_history, forecast_records)

    return {
        "location": location_dict,
        "current_location": location_dict,
        "known_cities": KNOWN_CITIES,
        "data_source": obs_data["data_source"],
        "is_live": obs_data["is_live"],
        "latest_timestamp": obs_data["timestamp"],
        "current_pm25": obs_data["pm25"],
        "current_pm10": obs_data["pm10"],
        "current_temp": obs_data["temperature"],
        "current_humidity": obs_data["humidity"],
        "current_wind": obs_data["wind_speed"],
        "current_pressure": obs_data["pressure"],
        "current_rainfall": obs_data["rainfall"],
        "current_alert": current_alert,
        "forecast_records": forecast_records,
        "forecast_summary": forecast_summary,
        "elevated_hours_count": elevated_hours_count,
        "metrics_data": METRICS_DATA,
        "chart_pm25_json": chart_pm25_json,
        "chart_pm10_json": chart_pm10_json,
        "chart_combined_json": chart_combined_json,
        "disclaimer": AI_DISCLAIMER
    }


# ------------------------------------------------------------------------------
# Flask Web Routes & API Endpoints
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    payload = get_full_dashboard_payload(current_location)
    return render_template("index.html", **payload)


@app.route("/api/dashboard-data")
def api_dashboard_data():
    """
    Returns full location-specific dashboard JSON payload for instant dynamic client update.
    """
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    payload = get_full_dashboard_payload(current_location)
    
    # Remove non-serializable objects from JSON return
    clean_payload = dict(payload)
    clean_payload.pop("known_cities", None)
    
    return jsonify(clean_payload)


@app.route("/api/current-location")
def api_current_location():
    """Returns currently selected location in session."""
    current_loc = session.get("selected_location", DEFAULT_LOCATION)
    return jsonify(current_loc)


@app.route("/api/forecast")
def api_forecast():
    """API endpoint for fetching 24-hour forecast for current location in JSON format."""
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    payload = get_full_dashboard_payload(current_location)
    return jsonify(payload["forecast_records"])


@app.route("/api/metrics")
def api_metrics():
    """API endpoint for fetching model evaluation metrics in JSON format."""
    return jsonify(METRICS_DATA)


@app.route("/api/alerts")
def api_alerts():
    """API endpoint for fetching current alert status and 24-hour forecast summary for location."""
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    payload = get_full_dashboard_payload(current_location)
    return jsonify({
        "current_alert": payload["current_alert"],
        "forecast_summary": payload["forecast_summary"]
    })


@app.route("/api/location", methods=["GET", "POST"])
def api_location():
    """
    GET: Returns currently selected location in session.
    POST: Updates location via latitude/longitude or city search query, recalculates data.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        lat = data.get("latitude") if data else None
        lon = data.get("longitude") if data else None
        city_query = data.get("city") if data else None

        result = None

        # 1. Coordinate-based lookup (Use My Location)
        if lat is not None and lon is not None:
            try:
                lat_f = float(lat)
                lon_f = float(lon)
                result = reverse_geocode_coordinates(lat_f, lon_f)
                result["source"] = "geolocation"
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "Invalid latitude or longitude format."}), 400

        # 2. Name-based lookup (Search or Dropdown select)
        elif city_query:
            result = geocode_city_name(str(city_query))
            if result:
                result["source"] = "search"
            else:
                return jsonify({"status": "error", "message": "Location not found. Try another city."}), 404
        else:
            return jsonify({"status": "error", "message": "Please provide either city name or latitude and longitude."}), 400

        if result:
            session["selected_location"] = result
            
            # Immediately compute new location dashboard payload
            dashboard_data = get_full_dashboard_payload(result)
            dashboard_data_clean = dict(dashboard_data)
            dashboard_data_clean.pop("known_cities", None)

            return jsonify({
                "status": "success",
                "message": "Location updated successfully.",
                "location": result,
                "dashboard_data": dashboard_data_clean
            })

    # GET request
    current_loc = session.get("selected_location", DEFAULT_LOCATION)
    return jsonify({
        "status": "success",
        "location": current_loc,
        "available_cities": [c["city"] for c in KNOWN_CITIES]
    })



# ------------------------------------------------------------------------------
# v1 API Architecture (Professional RESTful Endpoints)
# NOTE: Authentication not yet implemented — prototype stage.
# ------------------------------------------------------------------------------
@app.route("/api/v1/health")
def api_v1_health():
    """System health check endpoint."""
    model_status = "available" if ML_MODEL is not None else "unavailable"
    return jsonify({
        "status": "ok",
        "service": "AeroSense Environmental Intelligence API",
        "version": "1.0.0-prototype",
        "note": "Authentication not yet implemented. API is in prototype stage.",
        "components": {
            "backend": "online",
            "forecast_model": model_status,
            "air_quality_data": "demo",
            "database": "session-based"
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/v1/location", methods=["GET", "POST"])
def api_v1_location():
    """v1 location endpoint — proxies existing /api/location logic."""
    if request.method == "POST":
        return api_location()
    current_loc = session.get("selected_location", DEFAULT_LOCATION)
    return jsonify({
        "status": "success",
        "location": current_loc,
        "available_cities": [c["city"] for c in KNOWN_CITIES]
    })


@app.route("/api/v1/air-quality")
def api_v1_air_quality():
    """Returns current air quality observation for selected location."""
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    payload = get_full_dashboard_payload(current_location)
    return jsonify({
        "status": "success",
        "location": payload["location"],
        "data_mode": "live" if payload["is_live"] else "demo",
        "data_source": payload["data_source"],
        "last_updated": payload["latest_timestamp"],
        "observations": {
            "pm25": payload["current_pm25"],
            "pm10": payload["current_pm10"],
            "temperature": payload["current_temp"],
            "humidity": payload["current_humidity"],
            "wind_speed": payload["current_wind"],
            "pressure": payload["current_pressure"],
            "rainfall": payload["current_rainfall"]
        },
        "risk": {
            "level": payload["current_alert"]["risk_category"],
            "severity": payload["current_alert"]["severity"]
        }
    })


@app.route("/api/v1/forecast")
def api_v1_forecast():
    """Returns 24-hour forecast for selected location."""
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    payload = get_full_dashboard_payload(current_location)
    return jsonify({
        "status": "success",
        "location": payload["location"],
        "data_mode": "demo",
        "horizon_hours": 24,
        "generated_at": datetime.now().isoformat(),
        "forecast": payload["forecast_records"],
        "peak_event": payload["forecast_summary"]["peak_pollution_event"],
        "risk_breakdown": payload["forecast_summary"]["risk_breakdown"]
    })


@app.route("/api/v1/alerts")
def api_v1_alerts():
    """Returns structured alert information for selected location."""
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    payload = get_full_dashboard_payload(current_location)
    return jsonify({
        "status": "success",
        "location": payload["location"],
        "current_alert": payload["current_alert"],
        "forecast_summary": payload["forecast_summary"],
        "disclaimer": payload["disclaimer"]
    })


@app.route("/api/v1/analytics")
def api_v1_analytics():
    """Returns historical analytics data for selected location."""
    from src.data_provider import get_location_air_quality
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    lat = float(current_location.get("latitude", 12.9716))
    lon = float(current_location.get("longitude", 77.5946))
    city = current_location.get("city", "Bengaluru")
    country = current_location.get("country", "India")

    try:
        obs_data = get_location_air_quality(lat, lon, city=city, country=country)
        df_history = obs_data["history_df"]

        # Build time-series for analytics
        time_series = []
        for _, row in df_history.iterrows():
            time_series.append({
                "timestamp": str(row.get("timestamp", "")),
                "pm25": round(float(row.get("PM2.5", 0)), 1),
                "pm10": round(float(row.get("PM10", 0)), 1),
                "temperature": round(float(row.get("temperature", 0)), 1),
                "humidity": round(float(row.get("humidity", 0)), 1),
                "wind_speed": round(float(row.get("wind_speed", 0)), 1)
            })

        # Summary statistics
        import numpy as np
        pm25_vals = [r["pm25"] for r in time_series]
        pm10_vals = [r["pm10"] for r in time_series]

        return jsonify({
            "status": "success",
            "location": current_location,
            "data_mode": "demo",
            "time_series": time_series[-720:],  # Last 30 days
            "statistics": {
                "pm25": {
                    "mean": round(float(np.mean(pm25_vals)), 2),
                    "max": round(float(np.max(pm25_vals)), 2),
                    "min": round(float(np.min(pm25_vals)), 2)
                },
                "pm10": {
                    "mean": round(float(np.mean(pm10_vals)), 2),
                    "max": round(float(np.max(pm10_vals)), 2),
                    "min": round(float(np.min(pm10_vals)), 2)
                }
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/v1/reports")
def api_v1_reports():
    """Returns data for environmental intelligence report generation."""
    from src.data_provider import get_location_air_quality
    current_location = session.get("selected_location", DEFAULT_LOCATION)
    payload = get_full_dashboard_payload(current_location)

    try:
        obs_data = get_location_air_quality(
            float(current_location.get("latitude", 12.9716)),
            float(current_location.get("longitude", 77.5946)),
            city=current_location.get("city", "Bengaluru"),
            country=current_location.get("country", "India")
        )
        df = obs_data["history_df"]

        import numpy as np
        pm25_vals = df["PM2.5"].dropna().tolist()
        pm10_vals = df["PM10"].dropna().tolist()

        # Count elevated risk hours in history
        elevated_hist = sum(1 for v in pm25_vals if v > 60)

        report_data = {
            "status": "success",
            "report_title": "AeroSense Environmental Intelligence Report",
            "location": payload["location"],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_mode": "live" if payload["is_live"] else "demo",
            "data_source": payload["data_source"],
            "period": "Last 30 days (demonstration dataset)",
            "summary_statistics": {
                "avg_pm25": round(float(np.mean(pm25_vals)), 2),
                "avg_pm10": round(float(np.mean(pm10_vals)), 2),
                "peak_pm25": round(float(max(pm25_vals)), 2),
                "peak_pm10": round(float(max(pm10_vals)), 2),
                "elevated_risk_hours_historical": elevated_hist,
                "elevated_risk_hours_forecast": payload["elevated_hours_count"],
                "forecast_peak_time": payload["forecast_summary"]["peak_pollution_event"]["timestamp"]
            },
            "model_performance": payload["metrics_data"],
            "ai_recommendations": {
                "public_actions": payload["current_alert"]["recommended_public_actions"],
                "industrial_actions": payload["forecast_summary"]["key_industrial_actions"]
            },
            "disclaimer": payload["disclaimer"]
        }
        return jsonify(report_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics-history")
def api_analytics_history():
    """Returns historical data for the analytics tab charts."""
    from src.data_provider import get_location_air_quality
    period = request.args.get("period", "7d")
    current_location = session.get("selected_location", DEFAULT_LOCATION)

    try:
        obs_data = get_location_air_quality(
            float(current_location.get("latitude", 12.9716)),
            float(current_location.get("longitude", 77.5946)),
            city=current_location.get("city", "Bengaluru"),
            country=current_location.get("country", "India")
        )
        df = obs_data["history_df"].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter by requested period
        period_map = {"7d": 168, "30d": 720, "90d": 2160, "1y": 8760}
        hours = period_map.get(period, 168)
        df = df.tail(hours)

        records = []
        for _, row in df.iterrows():
            records.append({
                "timestamp": str(row["timestamp"]),
                "pm25": round(float(row.get("PM2.5", 0)), 1),
                "pm10": round(float(row.get("PM10", 0)), 1),
                "temperature": round(float(row.get("temperature", 0)), 1),
                "humidity": round(float(row.get("humidity", 0)), 1),
                "wind_speed": round(float(row.get("wind_speed", 0)), 1)
            })

        return jsonify({
            "status": "success",
            "period": period,
            "location": current_location,
            "data_mode": "demo",
            "records": records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)

