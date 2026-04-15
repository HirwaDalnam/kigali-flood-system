"""
data_fetcher.py
───────────────
Cloud data ingestion layer for the Kigali flood prediction system.

Sources:
  1. Open-Meteo  — hourly precipitation forecast (free, no API key)
  2. Open-Meteo  — current weather conditions
  3. NASA POWER  — historical daily precipitation (requires free Earthdata login)
  4. NASA IMERG  — near-real-time satellite rainfall estimates (30-min, 0.1° grid)

All fetchers are async (httpx). Each returns a normalized dict so the
rest of the system doesn't need to know about the source format.
"""

import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

KIGALI_LAT = -1.9441
KIGALI_LON = 30.0619
KIGALI_TZ  = "Africa/Kigali"

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
NASA_POWER_BASE  = "https://power.larc.nasa.gov/api/temporal/daily/point"

TIMEOUT = httpx.Timeout(15.0)

# ─── 1. Open-Meteo: Hourly Precipitation Forecast ───────────────────────────

async def fetch_precipitation_forecast(
    forecast_days: int = 7,
) -> dict:
    """
    Fetch hourly precipitation forecast from Open-Meteo.
    No API key required. Free tier: unlimited calls.

    Returns:
        {
          "source": "open-meteo",
          "fetched_at": ISO timestamp,
          "timezone": "Africa/Kigali",
          "hourly": {
              "time":                     ["2025-01-01T00:00", ...],
              "precipitation":            [0.0, 1.2, ...],   # mm/hr
              "precipitation_probability":[0, 30, ...],       # %
              "rain":                     [0.0, 1.2, ...]    # mm (liquid only)
          },
          "daily": {
              "time":                        ["2025-01-01", ...],
              "precipitation_sum":           [5.2, ...],       # mm
              "precipitation_probability_max": [70, ...]       # %
          }
        }
    """
    params = {
        "latitude":    KIGALI_LAT,
        "longitude":   KIGALI_LON,
        "hourly":      "precipitation,precipitation_probability,rain,cape",
        "daily":       "precipitation_sum,rain_sum,precipitation_probability_max,precipitation_hours",
        "forecast_days": forecast_days,
        "timezone":    KIGALI_TZ,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(OPEN_METEO_BASE, params=params)
        resp.raise_for_status()
        raw = resp.json()

    return {
        "source":     "open-meteo",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "timezone":   KIGALI_TZ,
        "latitude":   raw["latitude"],
        "longitude":  raw["longitude"],
        "hourly":     raw.get("hourly", {}),
        "daily":      raw.get("daily", {}),
    }


# ─── 2. Open-Meteo: Current Conditions ──────────────────────────────────────

async def fetch_current_conditions() -> dict:
    """
    Fetch current real-time weather for Kigali.

    Returns:
        {
          "source": "open-meteo-current",
          "time": "2025-01-01T14:00",
          "temperature_c": 22.5,
          "humidity_pct": 78,
          "precipitation_mm": 0.2,   # last hour accumulation
          "rain_mm": 0.2,
          "wind_speed_kmh": 12.0,
          "is_raining": bool,
          "raw": { ... full open-meteo current block ... }
        }
    """
    params = {
        "latitude":  KIGALI_LAT,
        "longitude": KIGALI_LON,
        "current":   (
            "precipitation,rain,temperature_2m,"
            "relative_humidity_2m,wind_speed_10m,"
            "weather_code,cloud_cover"
        ),
        "timezone":  KIGALI_TZ,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(OPEN_METEO_BASE, params=params)
        resp.raise_for_status()
        raw = resp.json()

    cur = raw.get("current", {})
    precip = cur.get("precipitation", 0.0) or 0.0

    return {
        "source":          "open-meteo-current",
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
        "time":            cur.get("time", ""),
        "temperature_c":   cur.get("temperature_2m"),
        "humidity_pct":    cur.get("relative_humidity_2m"),
        "precipitation_mm": precip,
        "rain_mm":          cur.get("rain", 0.0),
        "wind_speed_kmh":   cur.get("wind_speed_10m"),
        "cloud_cover_pct":  cur.get("cloud_cover"),
        "weather_code":     cur.get("weather_code"),
        "is_raining":       precip > 0.1,
        "raw":              cur,
    }


# ─── 3. NASA POWER: Historical Daily Rainfall ───────────────────────────────

async def fetch_historical_rainfall(
    start_date: str,   # "YYYYMMDD"
    end_date: str,     # "YYYYMMDD"
) -> dict:
    """
    Fetch historical daily precipitation from NASA POWER API.
    Free, but requires registration at https://power.larc.nasa.gov/

    Uses PRECTOTCORR parameter: bias-corrected precipitation (mm/day).

    Args:
        start_date: e.g. "20250101"
        end_date:   e.g. "20250107"

    Returns:
        {
          "source": "nasa-power",
          "dates": ["20250101", ...],
          "daily_mm": [5.2, 0.0, 12.4, ...],
          "total_mm": 17.6,
          "mean_mm":  5.9
        }
    """
    params = {
        "parameters": "PRECTOTCORR",
        "community":  "RE",
        "longitude":  KIGALI_LON,
        "latitude":   KIGALI_LAT,
        "start":      start_date,
        "end":        end_date,
        "format":     "JSON",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        resp = await client.get(NASA_POWER_BASE, params=params)
        resp.raise_for_status()
        raw = resp.json()

    daily_data = (
        raw.get("properties", {})
           .get("parameter", {})
           .get("PRECTOTCORR", {})
    )

    dates = sorted(daily_data.keys())
    values = [daily_data[d] for d in dates]
    valid = [v for v in values if v >= 0]  # -999 = missing data

    return {
        "source":    "nasa-power",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "dates":     dates,
        "daily_mm":  values,
        "total_mm":  round(sum(valid), 2),
        "mean_mm":   round(sum(valid) / len(valid), 2) if valid else 0.0,
    }


# ─── 4. Compute Antecedent 5-Day Rainfall (for AMC correction) ──────────────

async def fetch_antecedent_rainfall() -> float:
    """
    Fetch prior 5 days of observed rainfall from Open-Meteo historical endpoint.
    Used to determine Antecedent Moisture Condition (AMC) for CN adjustment.

    Returns:
        Total mm of rain over the last 5 days.
    """
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=5)).isoformat()
    end   = (today - timedelta(days=1)).isoformat()

    params = {
        "latitude":   KIGALI_LAT,
        "longitude":  KIGALI_LON,
        "daily":      "precipitation_sum",
        "start_date": start,
        "end_date":   end,
        "timezone":   KIGALI_TZ,
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            raw = resp.json()

        values = raw.get("daily", {}).get("precipitation_sum", [])
        total = sum(v for v in values if v is not None)
        logger.info(f"Antecedent 5-day rainfall: {total:.1f} mm")
        return round(total, 2)

    except Exception as e:
        logger.warning(f"Could not fetch antecedent rainfall: {e}. Defaulting to AMC-II (20mm).")
        return 20.0  # neutral fallback (AMC-II)
