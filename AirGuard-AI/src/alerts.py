"""
================================================================================
AeroSense - Public Health & Industrial Alert Advisory System
================================================================================
Problem Statement: PS-1A (Air Quality Forecasting & Public Health Alert)

DISCLAIMER:
ALL HEALTH ADVISORIES AND INDUSTRIAL ACTION PROTOCOLS PRODUCED BY THIS SYSTEM
ARE AUTOMATED AI-GENERATED RECOMMENDATIONS FOR INFORMATIONAL/DEMO PURPOSES ONLY.
THEY DO NOT CONSTITUTE OFFICIAL STATUTORY GOVERNMENT ORDERS OR MEDICAL DIAGNOSES.
================================================================================

This module evaluates predicted or measured concentrations of PM2.5 and PM10,
determines the risk category (Low, Moderate, High, Very High), and outputs:
1. Public Health Message (Vulnerability risk description)
2. Recommended Public Actions (Citizen guidelines, outdoor activities, masks)
3. Recommended Industrial Actions (Emission rescheduling, abatement, monitoring)
"""

import os
import json
from typing import Dict, Any, List, Union
import pandas as pd


# Mandatory AI advisory disclaimer notice
AI_DISCLAIMER = (
    "AI-Generated Advisory: These recommendations are automatically produced by AeroSense "
    "predictive models for public awareness and voluntary operational mitigation. "
    "They do not constitute official statutory government orders."
)


# Standard pollution risk rule-set based on PM2.5 and PM10 thresholds (µg/m³)
ALERT_RULES = {
    "Low": {
        "badge_color": "#10B981",  # Emerald Green
        "severity": "info",
        "description": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "pm25_range": "0 - 30 µg/m³",
        "pm10_range": "0 - 50 µg/m³",
        "health_message": "Air quality is good. It is ideal for outdoor activities for all population groups.",
        "public_actions": [
            "Enjoy normal outdoor activities and sports.",
            "Ventilate indoor spaces with fresh air.",
            "No special health precautions required for sensitive individuals."
        ],
        "industrial_actions": [
            "Maintain standard operating procedures and standard continuous emission monitoring (CEMS).",
            "Perform scheduled routine equipment maintenance.",
            "Keep dust suppression misting operational on haul roads."
        ]
    },
    "Moderate": {
        "badge_color": "#F59E0B",  # Amber / Yellow
        "severity": "warning",
        "description": "Air quality is acceptable; however, some pollutants may cause moderate health concern for sensitive individuals.",
        "pm25_range": "31 - 60 µg/m³",
        "pm10_range": "51 - 100 µg/m³",
        "health_message": "Mild respiratory discomfort may be experienced by unusually sensitive individuals, children, and the elderly.",
        "public_actions": [
            "Sensitive individuals (asthma, COPD, heart conditions) should consider reducing prolonged or heavy outdoor exertion.",
            "Keep emergency inhalers and medication accessible.",
            "Prefer public transportation or carpooling to reduce localized emissions."
        ],
        "industrial_actions": [
            "Increase automated emission monitoring frequency on primary discharge stacks.",
            "Ensure electrostatic precipitators (ESP) and baghouse filters operate at optimal efficiency.",
            "Avoid uncontained material transfer and increase water spraying at construction/mining sites."
        ]
    },
    "High": {
        "badge_color": "#F97316",  # Orange / Red-Orange
        "severity": "danger",
        "description": "Air quality reaches unhealthy levels. Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects.",
        "pm25_range": "61 - 90 µg/m³",
        "pm10_range": "101 - 250 µg/m³",
        "health_message": "Breathing discomfort is likely for people with lung/heart disease, children, and elderly. Healthy adults may notice irritation.",
        "public_actions": [
            "Sensitive groups should avoid strenuous outdoor physical activities.",
            "General public should limit prolonged outdoor exertion and wear N95/FFP2 masks near heavy traffic corridors.",
            "Keep windows closed during morning and evening rush-hour peaks; use HEPA air purifiers if available."
        ],
        "industrial_actions": [
            "Reschedule high-emission operational batch runs away from predicted peak pollution hours (e.g., postpone to afternoon windy periods).",
            "Engage secondary auxiliary emission-control equipment (scrubbers, activated carbon injectors).",
            "Halt open diesel generator testing and switch to grid power or cleaner backup sources.",
            "Temporarily suspend high-dust earthwork, demolition, and non-essential heavy transport transit."
        ]
    },
    "Very High": {
        "badge_color": "#EF4444",  # Crimson Red / Burgundy
        "severity": "critical",
        "description": "Emergency conditions. Health warnings of emergency conditions. The entire population is likely to be severely affected.",
        "pm25_range": "> 90 µg/m³",
        "pm10_range": "> 250 µg/m³",
        "health_message": "Severe public health risk. Significant increase in respiratory and cardiovascular symptoms across the general population.",
        "public_actions": [
            "Avoid all outdoor physical activity. Stay indoors with doors and windows tightly sealed.",
            "Vulnerable groups (children, pregnant women, elderly, asthmatics) must remain strictly indoors in filtered environments.",
            "Mandatory high-grade particulate filtering respirator (N95/N99) if stepping outside is unavoidable.",
            "Seek immediate medical attention if experiencing chest tightness, wheezing, or severe shortness of breath."
        ],
        "industrial_actions": [
            "Immediately reduce non-essential high-emission industrial operations and reduce production load on heavy fossil boilers.",
            "Mandate 100% maximum capacity on all flue-gas desulfurization and particulate scrubbing units.",
            "Enforce strict stoppage on all unpaved construction activities, concrete mixing, and open diesel machinery.",
            "Activate facility emergency environmental response plan and report mitigation telemetry to central dashboard."
        ]
    }
}


def classify_risk_level(pm25: float, pm10: float) -> str:
    """
    Determines the overall pollution risk level from PM2.5 and PM10 concentrations.
    Takes the worse of the two pollutant categories (conservative precautionary principle).
    """
    pm25 = float(pm25)
    pm10 = float(pm10)

    # Risk level from PM2.5
    if pm25 <= 30.0:
        risk_pm25 = 1
    elif pm25 <= 60.0:
        risk_pm25 = 2
    elif pm25 <= 90.0:
        risk_pm25 = 3
    else:
        risk_pm25 = 4

    # Risk level from PM10
    if pm10 <= 50.0:
        risk_pm10 = 1
    elif pm10 <= 100.0:
        risk_pm10 = 2
    elif pm10 <= 250.0:
        risk_pm10 = 3
    else:
        risk_pm10 = 4

    overall_score = max(risk_pm25, risk_pm10)
    score_map = {1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}
    return score_map[overall_score]


def get_alert_advisory(pm25: float, pm10: float, timestamp: str = None) -> Dict[str, Any]:
    """
    Generates a complete structured alert advisory for a given PM2.5 and PM10 measurement or forecast.
    """
    risk_category = classify_risk_level(pm25, pm10)
    rule = ALERT_RULES[risk_category]

    return {
        "timestamp": timestamp,
        "risk_category": risk_category,
        "badge_color": rule["badge_color"],
        "severity": rule["severity"],
        "observed_or_predicted": {
            "PM2.5": round(float(pm25), 2),
            "PM10": round(float(pm10), 2)
        },
        "description": rule["description"],
        "public_health_message": rule["health_message"],
        "recommended_public_actions": rule["public_actions"],
        "recommended_industrial_actions": rule["industrial_actions"],
        "is_ai_generated": True,
        "disclaimer": AI_DISCLAIMER
    }


def process_forecast_alerts(forecast_data_or_path: Union[str, List[Dict[str, Any]], pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Processes a 24-hour forecast series, determines hourly alert classifications,
    identifies peak pollution time windows, and creates an executive alert summary.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load data if path provided or default
    if forecast_data_or_path is None:
        forecast_path = os.path.join(base_dir, 'data', 'forecast_24h.json')
        with open(forecast_path, 'r') as f:
            records = json.load(f)
    elif isinstance(forecast_data_or_path, str):
        if forecast_data_or_path.endswith('.json'):
            with open(forecast_data_or_path, 'r') as f:
                records = json.load(f)
        else:
            df = pd.read_csv(forecast_data_or_path)
            records = df.to_dict(orient='records')
    elif isinstance(forecast_data_or_path, pd.DataFrame):
        records = forecast_data_or_path.to_dict(orient='records')
    else:
        records = forecast_data_or_path

    hourly_alerts = []
    max_pm25 = -1.0
    max_pm10 = -1.0
    peak_timestamp = None
    category_counts = {"Low": 0, "Moderate": 0, "High": 0, "Very High": 0}

    for item in records:
        ts = item.get("timestamp")
        pm25 = item.get("predicted_PM2.5", item.get("PM2.5", 0.0))
        pm10 = item.get("predicted_PM10", item.get("PM10", 0.0))

        advisory = get_alert_advisory(pm25, pm10, timestamp=ts)
        hourly_alerts.append(advisory)

        cat = advisory["risk_category"]
        category_counts[cat] += 1

        if pm25 > max_pm25:
            max_pm25 = pm25
            max_pm10 = pm10
            peak_timestamp = ts

    # Overall worst-case advisory across the 24-hour forecast horizon
    worst_risk = classify_risk_level(max_pm25, max_pm10)
    primary_advisory = ALERT_RULES[worst_risk]

    summary = {
        "horizon_hours": len(hourly_alerts),
        "peak_pollution_event": {
            "timestamp": peak_timestamp,
            "peak_pm25": round(max_pm25, 2),
            "peak_pm10": round(max_pm10, 2),
            "peak_PM2.5": round(max_pm25, 2),
            "peak_PM10": round(max_pm10, 2),
            "peak_risk_level": worst_risk
        },
        "risk_breakdown": category_counts,
        "executive_health_message": primary_advisory["health_message"],
        "key_public_actions": primary_advisory["public_actions"],
        "key_industrial_actions": primary_advisory["industrial_actions"],
        "hourly_timeline": hourly_alerts,
        "disclaimer": AI_DISCLAIMER
    }

    return summary


if __name__ == "__main__":
    print("=" * 75)
    print("📢 AeroSense - Alert System Test Run")
    print("=" * 75)

    # 1. Single sample evaluations across categories
    sample_tests = [
        ("Good Condition", 18.5, 38.0),
        ("Moderate Morning", 45.0, 85.0),
        ("High Evening Spike", 78.0, 160.0),
        ("Very High Winter Inversion", 115.0, 290.0),
    ]

    for label, p25, p10 in sample_tests:
        alert = get_alert_advisory(p25, p10)
        print(f"\n[{label}] PM2.5: {p25} µg/m³, PM10: {p10} µg/m³")
        print(f"👉 Risk Level    : {alert['risk_category']}")
        print(f"🏥 Health Message: {alert['public_health_message']}")
        print(f"👥 Public Action : {alert['recommended_public_actions'][0]}")
        print(f"🏭 Industry Rec  : {alert['recommended_industrial_actions'][0]}")

    # 2. Process 24-Hour Forecast if available
    try:
        forecast_summary = process_forecast_alerts()
        print("\n" + "=" * 75)
        print("📊 24-Hour Forecast Alert Summary:")
        print(f"Peak Risk Level  : {forecast_summary['peak_pollution_event']['peak_risk_level']}")
        print(f"Peak Timestamp   : {forecast_summary['peak_pollution_event']['timestamp']}")
        print(f"Peak PM2.5 / PM10: {forecast_summary['peak_pollution_event']['peak_PM2.5']} / {forecast_summary['peak_pollution_event']['peak_PM10']} µg/m³")
        print(f"Risk Breakdown   : {forecast_summary['risk_breakdown']}")
        print(f"AI Disclaimer    : {forecast_summary['disclaimer']}")
        print("=" * 75)
    except Exception as e:
        print(f"Notice: Could not load 24h forecast ({e})")
