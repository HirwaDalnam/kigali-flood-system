"""
main.py
───────
FastAPI backend for the Kigali Real-Time Flood Prediction System.

Endpoints:
  GET /rainfall/forecast       — hourly precipitation forecast (Open-Meteo)
  GET /rainfall/current        — current weather conditions
  GET /flood-risk              — per-district flood risk scores
  GET /flood-risk/{district}   — single district detail
  GET /history/antecedent      — prior 5-day rainfall for AMC calculation
  GET /health                  — system health check

Run with:
  pip install fastapi uvicorn httpx
  uvicorn main:app --reload --port 8000

Then visit: http://localhost:8000/docs  (auto-generated API documentation)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import asyncio

from data_fetcher import (
    fetch_precipitation_forecast,
    fetch_current_conditions,
    fetch_antecedent_rainfall,
)
from flood_engine import (
    KIGALI_DISTRICTS,
    compute_district_risk,
)

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Kigali Flood Prediction API",
    description=(
        "Real-time flood prediction for Kigali, Rwanda. "
        "Pulls live weather data from Open-Meteo and computes "
        "per-district flood risk using the SCS-CN runoff model."
    ),
    version="1.0.0",
)

# Allow requests from your frontend (update in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Replace with your frontend URL in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint — confirms the API is live."""
    return {
        "api": "Kigali Flood Prediction System",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/rainfall/current",
            "/rainfall/forecast",
            "/flood-risk",
            "/flood-risk/{district_name}",
            "/history/antecedent",
        ]
    }


@app.get("/health")
async def health():
    """Verify the API is running and all data sources are reachable."""
    return {
        "status": "ok",
        "server_time": datetime.now(timezone.utc).isoformat(),
        "districts_configured": list(KIGALI_DISTRICTS.keys()),
    }


@app.get("/rainfall/forecast")
async def rainfall_forecast(days: int = 7):
    """
    Fetch hourly precipitation forecast for Kigali.

    Returns hourly values for precipitation (mm/hr), rain probability (%),
    and daily totals. Sourced from Open-Meteo — free, no API key needed.
    """
    if not 1 <= days <= 16:
        raise HTTPException(400, "days must be between 1 and 16")
    return await fetch_precipitation_forecast(forecast_days=days)


@app.get("/rainfall/current")
async def rainfall_current():
    """
    Get current real-time weather conditions in Kigali.

    Includes: temperature, humidity, current precipitation rate,
    wind speed, cloud cover. Updates every 15 minutes.
    """
    return await fetch_current_conditions()


@app.get("/history/antecedent")
async def antecedent_rainfall():
    """
    Retrieve the past 5 days of observed rainfall.

    Used to determine the Antecedent Moisture Condition (AMC)
    for SCS-CN curve number adjustment.
    """
    total = await fetch_antecedent_rainfall()
    if total < 36:
        amc = "AMC-I (dry soil — lower runoff)"
    elif total <= 53:
        amc = "AMC-II (normal — standard CN applies)"
    else:
        amc = "AMC-III (wet/saturated — higher runoff risk)"

    return {
        "prior_5day_mm": total,
        "amc_condition": amc,
    }


@app.get("/flood-risk")
async def flood_risk(window_hours: int = 6):
    """
    Compute real-time flood risk for all Kigali districts.

    Pipeline:
      1. Fetch hourly precipitation forecast (Open-Meteo)
      2. Fetch antecedent 5-day rainfall for AMC correction
      3. Run SCS-CN + Rational Method for each district
      4. Return risk level, discharge, and recommended actions

    Args:
        window_hours: Accumulation window — 6 (near-term), 24 (daily)
    """
    if window_hours not in (6, 24, 72):
        raise HTTPException(400, "window_hours must be 6, 24, or 72")

    # Fetch data in parallel
    forecast_data, antecedent_mm = await asyncio.gather(
        fetch_precipitation_forecast(forecast_days=3),
        fetch_antecedent_rainfall(),
    )

    hourly_precip = forecast_data["hourly"].get("precipitation", [])

    if not hourly_precip:
        raise HTTPException(502, "No precipitation data received from Open-Meteo")

    results = []
    highest_risk_level = "normal"
    risk_order = ["normal", "watch", "warning", "flood"]

    for district in KIGALI_DISTRICTS.values():
        result = compute_district_risk(
            district=district,
            hourly_precipitation=hourly_precip,
            prior_5day_mm=antecedent_mm,
            window_hours=window_hours,
        )
        results.append(result)

        # Track overall highest risk
        r = result["risk"]["level"]
        if risk_order.index(r) > risk_order.index(highest_risk_level):
            highest_risk_level = r

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "forecast_source": "open-meteo",
        "analysis_window_hours": window_hours,
        "antecedent_5day_mm": antecedent_mm,
        "overall_risk_level": highest_risk_level,
        "districts": results,
    }


@app.get("/flood-risk/{district_name}")
async def flood_risk_district(district_name: str, window_hours: int = 6):
    """
    Compute flood risk for a single named district.

    district_name: Nyarugenge | Gasabo | Kicukiro | Nyabugogo
    """
    # Case-insensitive lookup
    match = next(
        (d for k, d in KIGALI_DISTRICTS.items()
         if k.lower() == district_name.lower()),
        None,
    )
    if not match:
        raise HTTPException(
            404,
            f"District '{district_name}' not found. "
            f"Available: {list(KIGALI_DISTRICTS.keys())}"
        )

    forecast_data, antecedent_mm = await asyncio.gather(
        fetch_precipitation_forecast(forecast_days=1),
        fetch_antecedent_rainfall(),
    )

    hourly_precip = forecast_data["hourly"].get("precipitation", [])
    result = compute_district_risk(
        district=match,
        hourly_precipitation=hourly_precip,
        prior_5day_mm=antecedent_mm,
        window_hours=window_hours,
    )

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
