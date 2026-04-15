"""
flood_engine.py
───────────────
Core hydrological calculations for the Kigali flood prediction system.

Methods implemented:
  1. SCS-CN (Soil Conservation Service Curve Number) — converts rainfall to runoff
  2. Rational Method — estimates peak discharge per catchment
  3. Risk classification — maps discharge ratio to alert level

Kigali district parameters are derived from:
  - Curve numbers: SoilGrids soil class + ESA WorldCover 10m land use
  - Catchment areas: Copernicus DEM 30m delineation
  - Channel capacities: Rwanda RNRA hydraulic survey estimates
"""

from dataclasses import dataclass
from typing import Literal

# ─── District Configuration ────────────────────────────────────────────────

@dataclass
class District:
    name: str
    cn: float           # SCS Curve Number (25–98)
    area_km2: float     # Catchment area in km²
    capacity_m3s: float # Max channel/drainage capacity before overflow (m³/s)
    lat: float          # Representative centroid latitude
    lon: float          # Representative centroid longitude


KIGALI_DISTRICTS: dict[str, District] = {
    "Nyarugenge": District(
        name="Nyarugenge",
        cn=85,           # Dense urban core, high imperviousness
        area_km2=97,
        capacity_m3s=45,
        lat=-1.9500, lon=30.0588,
    ),
    "Gasabo": District(
        name="Gasabo",
        cn=72,           # Peri-urban / residential, more vegetated slopes
        area_km2=429,
        capacity_m3s=85,
        lat=-1.8950, lon=30.1100,
    ),
    "Kicukiro": District(
        name="Kicukiro",
        cn=78,           # Mixed residential / industrial
        area_km2=167,
        capacity_m3s=62,
        lat=-1.9950, lon=30.1000,
    ),
    "Nyabugogo": District(
        name="Nyabugogo",
        cn=90,           # High-risk lowland basin — worst flooding history
        area_km2=15,
        capacity_m3s=20,
        lat=-1.9350, lon=30.0500,
    ),
}

# ─── Risk Levels ────────────────────────────────────────────────────────────

RiskLevel = Literal["normal", "watch", "warning", "flood"]

RISK_THRESHOLDS = {
    "normal":  (0.0,  0.5),
    "watch":   (0.5,  0.8),
    "warning": (0.8,  1.0),
    "flood":   (1.0, float("inf")),
}

RISK_META = {
    "normal":  {"label": "Normal",  "color": "#22c55e", "action": "No action required"},
    "watch":   {"label": "Watch",   "color": "#f59e0b", "action": "Monitor drainage; alert field crews"},
    "warning": {"label": "Warning", "color": "#ef4444", "action": "Activate drainage pumps; prepare evacuation routes"},
    "flood":   {"label": "Flood",   "color": "#7f1d1d", "action": "Issue evacuation orders; close flood-prone roads"},
}

# ─── Core Hydrological Functions ────────────────────────────────────────────

def scs_cn_runoff(P_mm: float, CN: float) -> float:
    """
    SCS-CN Runoff Equation:
        Q = (P - Ia)² / (P - Ia + S)
        where S = (25400/CN) - 254   (mm)
              Ia = 0.2 * S           (initial abstraction; 20% of S)

    Args:
        P_mm: Total accumulated rainfall (mm) over the time window
        CN:   SCS Curve Number for the catchment (dimensionless, 25–98)

    Returns:
        Q: Direct runoff depth (mm). Zero if P ≤ Ia.
    """
    if CN <= 0 or CN > 100:
        raise ValueError(f"CN must be between 1–100, got {CN}")

    S = (25400 / CN) - 254      # Max potential retention (mm)
    Ia = 0.2 * S                # Initial abstraction (mm)

    if P_mm <= Ia:
        return 0.0              # All rainfall absorbed; no runoff yet

    Q = ((P_mm - Ia) ** 2) / (P_mm - Ia + S)
    return round(Q, 3)


def rational_method_peak_discharge(
    C: float,
    I_mm_hr: float,
    A_ha: float,
) -> float:
    """
    Rational Method: Q_peak = (C × I × A) / 360

    Args:
        C:       Runoff coefficient (dimensionless, 0.0–1.0)
        I_mm_hr: Rainfall intensity (mm/hr) — typically peak 1-hr value
        A_ha:    Catchment area in hectares

    Returns:
        Q_peak: Peak discharge in m³/s
    """
    return round((C * I_mm_hr * A_ha) / 360, 3)


def antecedent_moisture_correction(CN: float, prior_5day_mm: float) -> float:
    """
    AMC (Antecedent Moisture Condition) adjustment:
      AMC-I  (dry):    prior_5day < 36mm  → CN reduced
      AMC-II (normal): 36–53mm            → CN unchanged
      AMC-III (wet):   > 53mm             → CN increased (more runoff)
    """
    if prior_5day_mm < 36:
        # AMC-I dry correction
        return round((4.2 * CN) / (10 - 0.058 * CN), 1)
    elif prior_5day_mm > 53:
        # AMC-III wet correction
        return round((23 * CN) / (10 + 0.13 * CN), 1)
    return CN  # AMC-II: no change


def classify_risk(Q_m3s: float, capacity_m3s: float) -> dict:
    """
    Compare predicted discharge against channel capacity and return risk metadata.
    """
    ratio = Q_m3s / capacity_m3s if capacity_m3s > 0 else float("inf")
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= ratio < hi:
            return {
                "level": level,
                "ratio": round(ratio, 3),
                "percent_capacity": round(ratio * 100, 1),
                **RISK_META[level],
            }
    return {"level": "flood", "ratio": round(ratio, 3), **RISK_META["flood"]}


# ─── District Risk Calculator ───────────────────────────────────────────────

def compute_district_risk(
    district: District,
    hourly_precipitation: list[float],   # mm/hr for each hour
    prior_5day_mm: float = 20.0,         # default to AMC-II
    window_hours: int = 6,
) -> dict:
    """
    Full risk calculation pipeline for one district.

    Steps:
      1. Accumulate precipitation over the analysis window
      2. Apply AMC correction to CN
      3. Compute SCS-CN runoff depth
      4. Derive runoff coefficient C = Q/P
      5. Calculate peak discharge via Rational Method
      6. Classify risk level

    Args:
        district:             District config object
        hourly_precipitation: List of hourly rainfall values (mm/hr)
        prior_5day_mm:        Antecedent 5-day rainfall for AMC correction
        window_hours:         Hours to accumulate (6 for near-term, 24 for daily)

    Returns:
        Full risk result dict
    """
    window = hourly_precipitation[:window_hours]

    P_mm = sum(window)                                       # Accumulated rainfall (mm)
    I_peak = max(window) if window else 0.0                  # Peak 1-hr intensity (mm/hr)

    adjusted_CN = antecedent_moisture_correction(district.cn, prior_5day_mm)
    Q_depth = scs_cn_runoff(P_mm, adjusted_CN)

    C = (Q_depth / P_mm) if P_mm > 0 else 0.0               # Effective runoff coefficient
    A_ha = district.area_km2 * 100                           # km² → hectares

    Q_peak = rational_method_peak_discharge(C, I_peak, A_ha) # m³/s
    risk = classify_risk(Q_peak, district.capacity_m3s)

    return {
        "district": district.name,
        "inputs": {
            "rainfall_accumulated_mm": round(P_mm, 2),
            "peak_intensity_mm_hr": round(I_peak, 2),
            "prior_5day_mm": prior_5day_mm,
            "curve_number_original": district.cn,
            "curve_number_adjusted": adjusted_CN,
        },
        "calculated": {
            "runoff_depth_mm": Q_depth,
            "runoff_coefficient": round(C, 3),
            "peak_discharge_m3s": Q_peak,
            "channel_capacity_m3s": district.capacity_m3s,
        },
        "risk": risk,
    }
