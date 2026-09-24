# 🌍 AeroSense

**Problem Statement (PS-1A):** Air Quality Forecasting & Public Health Alert System

## 📋 Overview
AeroSense is an intelligent air quality prediction and public health warning platform designed to forecast pollutant levels (PM2.5, PM10, NO2, CO, SO2, O3), calculate the Air Quality Index (AQI), and trigger timely health advisories and alerts to vulnerable populations.

---

## 📁 Project Structure

```text
AirGuard-AI/
├── app.py              # Main web application entry point
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation and setup guide
├── data/               # Raw, processed, and sample datasets
├── models/             # Saved ML model artifacts (e.g., .pkl, .joblib)
├── src/                # Core Python modules (data processing, ML pipelines, alerts)
└── templates/          # HTML templates for the frontend web interface
```

---

## 🚀 Getting Started

### 1. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
python app.py
```
