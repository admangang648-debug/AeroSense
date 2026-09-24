"""
================================================================================
AeroSense - Dynamic Location Data Provider & OpenAQ Client
================================================================================
Problem Statement: PS-1A (Air Quality Forecasting & Public Health Alert)

This module handles:
1. Live OpenAQ API query based on coordinates (when OPENAQ_API_KEY is available).
2. Deterministic Geographic Fallback: Produces stable, location-specific synthetic
   air quality and meteorological time-series based on coordinates and geography.
3. Seamless handoff to the ML forecasting engine.
================================================================================
"""

import os
import math
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def get_openaq_api_key() -> str:
    """Reads OPENAQ_API_KEY from environment or .env file."""
    api_key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not api_key:
        # Check workspace .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAQ_API_KEY=") and not line.startswith("#"):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    return api_key


def fetch_live_openaq_data(lat: float, lon: float, radius_meters: int = 50000) -> dict:
    """
    Attempts to retrieve recent real-time air quality observations from OpenAQ API.
    Returns None if OpenAQ is unreachable, unauthorized, or has no nearby sensors.
    """
    api_key = get_openaq_api_key()
    if not api_key:
        return None

    try:
        url = f"https://api.openaq.org/v2/latest?coordinates={lat},{lon}&radius={radius_meters}&limit=1"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AeroSense-AirQuality-App/1.0",
                "X-API-Key": api_key
            }
        )
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode())
                results = payload.get("results", [])
                if results and len(results) > 0:
                    first = results[0]
                    measurements = first.get("measurements", [])
                    pm25_val, pm10_val = None, None
                    for m in measurements:
                        param = m.get("parameter")
                        val = m.get("value")
                        if param == "pm25" and val is not None and val >= 0:
                            pm25_val = float(val)
                        elif param == "pm10" and val is not None and val >= 0:
                            pm10_val = float(val)

                    if pm25_val is not None:
                        return {
                            "PM2.5": round(pm25_val, 1),
                            "PM10": round(pm10_val if pm10_val is not None else pm25_val * 1.6, 1),
                            "source_name": first.get("location", "OpenAQ Station"),
                            "is_live": True
                        }
    except Exception as e:
        # OpenAQ network error or timeout -> fall back cleanly
        pass

    return None


def get_location_profile(lat: float, lon: float, city: str = "") -> dict:
    """
    Calculates deterministic physical and environmental baselines for any geographic coordinate.
    Uses geographic heuristics (latitude, coastal proximity, inland continental factors).
    """
    city_lower = (city or "").lower()

    # Pre-calibrated regional profiles for major cities
    if "delhi" in city_lower or (28.0 <= lat <= 29.2 and 76.5 <= lon <= 77.8):
        return {
            "base_pm25": 74.0,
            "base_pm10": 145.0,
            "base_temp": 31.0,
            "temp_swing": 8.5,
            "base_humidity": 42.0,
            "base_wind": 3.2,
            "coastal": False,
            "profile_name": "Inland Northern Megacity (Elevated Base Pollution)"
        }
    elif "mumbai" in city_lower or (18.5 <= lat <= 19.5 and 72.5 <= lon <= 73.2):
        return {
            "base_pm25": 42.0,
            "base_pm10": 88.0,
            "base_temp": 30.0,
            "temp_swing": 4.5,
            "base_humidity": 78.0,
            "base_wind": 5.8,
            "coastal": True,
            "profile_name": "Coastal Maritime Megacity (High Humidity & Sea Breeze)"
        }
    elif "bengaluru" in city_lower or "bangalore" in city_lower or (12.5 <= lat <= 13.5 and 77.0 <= lon <= 78.0):
        return {
            "base_pm25": 24.0,
            "base_pm10": 52.0,
            "base_temp": 24.5,
            "temp_swing": 6.0,
            "base_humidity": 62.0,
            "base_wind": 4.2,
            "coastal": False,
            "profile_name": "Deccan Plateau Urban (Moderate Climate & Low Particulates)"
        }
    elif "hyderabad" in city_lower or (17.0 <= lat <= 17.8 and 78.0 <= lon <= 79.0):
        return {
            "base_pm25": 38.0,
            "base_pm10": 78.0,
            "base_temp": 29.0,
            "temp_swing": 7.0,
            "base_humidity": 55.0,
            "base_wind": 3.8,
            "coastal": False,
            "profile_name": "Semi-Arid Plateau Urban (Moderate Baseline)"
        }
    elif "chennai" in city_lower or (12.8 <= lat <= 13.4 and 80.0 <= lon <= 80.5):
        return {
            "base_pm25": 32.0,
            "base_pm10": 68.0,
            "base_temp": 32.0,
            "temp_swing": 4.8,
            "base_humidity": 76.0,
            "base_wind": 5.2,
            "coastal": True,
            "profile_name": "Coromandel Coastal (High Heat & Coastal Dispersion)"
        }
    elif "new york" in city_lower or (40.4 <= lat <= 41.0 and -74.3 <= lon <= -73.7):
        return {
            "base_pm25": 14.0,
            "base_pm10": 26.0,
            "base_temp": 18.0,
            "temp_swing": 7.0,
            "base_humidity": 58.0,
            "base_wind": 5.5,
            "coastal": True,
            "profile_name": "Temperate Coastal Urban"
        }
    elif "london" in city_lower or (51.2 <= lat <= 51.8 and -0.5 <= lon <= 0.3):
        return {
            "base_pm25": 12.0,
            "base_pm10": 22.0,
            "base_temp": 16.0,
            "temp_swing": 5.0,
            "base_humidity": 72.0,
            "base_wind": 6.0,
            "coastal": False,
            "profile_name": "Temperate Oceanic"
        }

    # Deterministic coordinate hash for arbitrary locations
    coord_seed = int((abs(lat) * 1000 + abs(lon) * 100) % 10000)
    lat_factor = min(1.0, abs(lat) / 60.0)
    base_temp = 32.0 - (lat_factor * 20.0)
    base_pm = 20.0 + (coord_seed % 35)

    return {
        "base_pm25": round(base_pm, 1),
        "base_pm10": round(base_pm * 1.85, 1),
        "base_temp": round(base_temp, 1),
        "temp_swing": 6.5,
        "base_humidity": 50.0 + (coord_seed % 30),
        "base_wind": 3.5 + (coord_seed % 4),
        "coastal": False,
        "profile_name": f"Coordinate-Derived Environmental Profile ({lat:.2f}°, {lon:.2f}°)"
    }


def generate_location_dataset(
    lat: float,
    lon: float,
    city: str = "Location",
    country: str = "",
    history_hours: int = 72
) -> pd.DataFrame:
    """
    Generates deterministic, physically grounded hourly time-series observation data
    for the selected location spanning the past `history_hours` up to current timestamp.
    """
    profile = get_location_profile(lat, lon, city)

    # Use deterministic seed derived from coordinates so values are stable across page loads
    coord_seed = int(abs(lat * 1000 + lon * 100)) % 1000000
    rng = np.random.RandomState(coord_seed)

    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=history_hours - 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(history_hours)]

    hours = np.array([ts.hour for ts in timestamps])
    is_weekend = np.array([1 if ts.weekday() >= 5 else 0 for ts in timestamps])

    # 1. Weather
    diurnal_temp = profile["temp_swing"] * np.sin(2 * np.pi * (hours - 9) / 24.0)
    temp = profile["base_temp"] + diurnal_temp + rng.normal(0, 0.6, history_hours)
    temperature = np.round(np.clip(temp, 4.0, 48.0), 1)

    hum_base = profile["base_humidity"] - (temperature - profile["base_temp"]) * 1.4
    humidity = np.round(np.clip(hum_base + rng.normal(0, 2.0, history_hours), 20.0, 95.0), 1)

    wind_base = profile["base_wind"] + 0.8 * np.maximum(0, np.sin(2 * np.pi * (hours - 12) / 24.0))
    wind_speed = np.round(np.clip(wind_base + rng.normal(0, 0.5, history_hours), 0.5, 15.0), 1)
    pressure = np.round(1013.25 + rng.normal(0, 1.2, history_hours), 1)
    rainfall = np.zeros(history_hours)

    # 2. Pollutants (Diurnal Traffic & Thermal Inversion Modeling)
    morning_rush = np.exp(-((hours - 8.5) ** 2) / 4.0)
    evening_rush = np.exp(-((hours - 19.5) ** 2) / 5.5)
    traffic_effect = (1.0 - 0.25 * is_weekend) * (1.5 * morning_rush + 1.8 * evening_rush + 0.5)

    night_inversion = np.where((hours < 7) | (hours > 21), 1.25, 1.0)
    wind_disp = np.clip(3.5 / (wind_speed + 0.5), 0.5, 2.0)

    pm25_series = (profile["base_pm25"] * traffic_effect * night_inversion * wind_disp) + rng.normal(0, 2.0, history_hours)
    pm25 = np.round(np.clip(pm25_series, 4.0, 450.0), 1)

    coarse_dust = (traffic_effect * 14.0) + (wind_speed * 2.0) + rng.normal(0, 3.0, history_hours)
    pm10 = np.round(np.clip((pm25 * 1.6) + coarse_dust, pm25 + 3.0, 650.0), 1)

    # Check if live OpenAQ reading is available to anchor the latest row
    live_data = fetch_live_openaq_data(lat, lon)
    if live_data and live_data.get("is_live"):
        pm25[-1] = live_data["PM2.5"]
        pm10[-1] = live_data["PM10"]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "PM2.5": pm25,
        "PM10": pm10,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "rainfall": rainfall
    })

    return df


def get_location_air_quality(lat: float, lon: float, city: str = "Location", country: str = "") -> dict:
    """
    Central function returning the complete observation state for any coordinate.
    """
    df_history = generate_location_dataset(lat, lon, city=city, country=country, history_hours=72)
    latest_row = df_history.iloc[-1]

    live_check = fetch_live_openaq_data(lat, lon)
    is_live = bool(live_check and live_check.get("is_live"))
    data_source = "OpenAQ (LIVE)" if is_live else "DEMO / SYNTHETIC"

    current_data = {
        "city": city,
        "country": country,
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "timestamp": df_history["timestamp"].iloc[-1].strftime("%b %d, %Y - %H:%M"),
        "pm25": float(latest_row["PM2.5"]),
        "pm10": float(latest_row["PM10"]),
        "temperature": float(latest_row["temperature"]),
        "humidity": float(latest_row["humidity"]),
        "wind_speed": float(latest_row["wind_speed"]),
        "pressure": float(latest_row["pressure"]),
        "rainfall": float(latest_row["rainfall"]),
        "is_live": is_live,
        "data_source": data_source,
        "history_df": df_history
    }

    # Debug Log Output to Terminal
    print("=" * 70)
    print(f"[AeroSense Debug] LOCATION: {city}, {country}")
    print(f"                 LAT: {lat:.4f} | LON: {lon:.4f}")
    print(f"                 DATA SOURCE: {data_source}")
    print(f"                 PM2.5: {current_data['pm25']} µg/m³ | PM10: {current_data['pm10']} µg/m³")
    print(f"                 TEMP: {current_data['temperature']}°C | HUMIDITY: {current_data['humidity']}%")
    print("=" * 70)

    return current_data
