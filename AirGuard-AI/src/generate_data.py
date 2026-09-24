"""
================================================================================
AeroSense - Synthetic Air Quality Dataset Generator
================================================================================
DISCLAIMER:
THIS SCRIPT GENERATES SYNTHETIC / DEMO DATA FOR HACKATHON DEMONSTRATION PURPOSES.
IT IS NOT SOURCED FROM REAL GOVERNMENT OR PHYSICAL SENSOR MEASUREMENTS.
================================================================================

This module simulates realistic 30-day hourly air quality and meteorological
observations incorporating realistic physical and environmental patterns:
1. Diurnal cycle (rush-hour traffic peaks, midday temperature highs, nighttime cooling).
2. Inverse humidity-temperature relationships.
3. Wind dispersion (higher wind speeds dilute particulate pollution).
4. Rain washout effect (precipitation significantly cleanses PM2.5 and PM10).
5. Particulate correlation (PM10 includes coarse dust + fine PM2.5).
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_synthetic_air_quality_data(
    num_days: int = 45,
    output_path: str = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic hourly synthetic air quality and weather data.

    Parameters:
    -----------
    num_days : int
        Number of days of hourly observations to simulate (default: 45 days >= 1080 hours).
    output_path : str
        Filepath to save the CSV. If None, saves to data/air_quality.csv.
    random_seed : int
        Seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        Generated air quality dataset.
    """
    np.random.seed(random_seed)

    # 1. Generate Hourly Timestamps (starting from 45 days ago up to now)
    total_hours = num_days * 24
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=total_hours - 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(total_hours)]

    # Time-based features
    hours = np.array([ts.hour for ts in timestamps])
    day_indices = np.array([i // 24 for i in range(total_hours)])
    is_weekend = np.array([1 if ts.weekday() >= 5 else 0 for ts in timestamps])

    # 2. Simulate Weather / Meteorology

    # Temperature (°C): Base seasonal temp + diurnal cycle (peaks ~14:00-15:00, coolest ~05:00) + random noise
    base_temp = 22.0
    diurnal_temp = 7.0 * np.sin(2 * np.pi * (hours - 9) / 24)
    temp_trend = 3.0 * np.sin(2 * np.pi * day_indices / 15)  # Multi-day weather fronts
    temperature = base_temp + diurnal_temp + temp_trend + np.random.normal(0, 1.2, total_hours)
    temperature = np.round(np.clip(temperature, 8.0, 42.0), 2)

    # Relative Humidity (%): Inversely related to temperature + random noise
    humidity_base = 65.0 - (temperature - 20.0) * 1.8
    humidity = humidity_base + np.random.normal(0, 4.0, total_hours)
    humidity = np.round(np.clip(humidity, 15.0, 98.0), 2)

    # Atmospheric Pressure (hPa): Standard sea-level pressure with synoptic scale waves
    base_pressure = 1013.25
    pressure_variation = 6.0 * np.cos(2 * np.pi * day_indices / 7) + np.random.normal(0, 0.8, total_hours)
    pressure = np.round(base_pressure + pressure_variation, 2)

    # Wind Speed (m/s): Higher during afternoon due to thermal turbulence, calmer at night
    wind_base = 2.5 + 1.2 * np.maximum(0, np.sin(2 * np.pi * (hours - 8) / 24))
    wind_speed = wind_base + np.random.gamma(shape=2.0, scale=0.8, size=total_hours)
    wind_speed = np.round(np.clip(wind_speed, 0.2, 18.0), 2)

    # Rainfall (mm): Episodic rain events (mostly 0, occasional storm showers)
    # Rain is more probable during lower pressure and high humidity
    rain_probability = np.where((pressure < 1010) & (humidity > 75), 0.25, 0.03)
    has_rain = np.random.rand(total_hours) < rain_probability
    rainfall = np.where(has_rain, np.random.exponential(scale=3.5, size=total_hours), 0.0)
    rainfall = np.round(rainfall, 2)

    # 3. Simulate Pollutants (PM2.5 and PM10 in µg/m³)

    # Traffic Rush Hour & Human Activity Cycles:
    # Morning rush (07:00 - 10:00) and Evening rush (18:00 - 22:00)
    morning_peak = np.exp(-((hours - 8.5) ** 2) / 3.5)
    evening_peak = np.exp(-((hours - 20.0) ** 2) / 5.0)
    traffic_activity = (1.0 - 0.3 * is_weekend) * (1.8 * morning_peak + 2.2 * evening_peak + 0.4)

    # Boundary layer / Nighttime Inversion effect: pollutants trap when wind is low and night cools
    inversion_effect = np.where((hours < 7) | (hours > 21), 1.35, 1.0)

    # Base PM2.5 calculation
    base_pm25 = 35.0
    pm25 = (base_pm25 * traffic_activity * inversion_effect)

    # Meteorological modulation:
    # - High wind speed disperses pollution (inverse wind factor)
    wind_dispersion = np.clip(3.0 / (wind_speed + 0.5), 0.4, 2.2)
    pm25 = pm25 * wind_dispersion

    # - Rain scavenging / washout effect: rain drastically clears particulates
    washout_factor = np.where(rainfall > 0.5, np.exp(-rainfall * 0.4), 1.0)
    pm25 = pm25 * washout_factor

    # Add realistic stochastic variance / spikes
    pm25_noise = np.random.normal(0, 4.5, total_hours)
    pm25 = np.round(np.clip(pm25 + pm25_noise, 4.0, 350.0), 2)

    # PM10 is correlated with PM2.5 (PM2.5 is a fraction of PM10) + coarse dust kicked up by wind/traffic
    coarse_dust = (traffic_activity * 12.0) + (wind_speed * 2.2) + np.random.normal(0, 6.0, total_hours)
    coarse_dust = np.clip(coarse_dust, 5.0, 120.0)
    pm10 = (pm25 * 1.55) + coarse_dust
    pm10 = np.round(np.clip(pm10 * washout_factor, pm25 + 2.0, 600.0), 2)

    # 4. Construct DataFrame
    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "PM2.5": pm25,
        "PM10": pm10,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "rainfall": rainfall
    })

    # 5. Resolve Output Path and Save
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        output_path = os.path.join(project_root, "data", "air_quality.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("=" * 70)
    print("✅ [DEMO/SYNTHETIC DATA] Air quality dataset successfully generated!")
    print(f"📊 Total Records : {len(df)} hourly rows ({num_days} days)")
    print(f"📁 Destination   : {output_path}")
    print("=" * 70)
    print("\nSample Preview (First 5 records):")
    print(df.head())
    print("\nDataset Summary Statistics:")
    print(df.describe())
    
    return df


if __name__ == "__main__":
    generate_synthetic_air_quality_data()
