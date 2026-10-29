"""Transparent market and Maharashtra new-car reference adjustments."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

CATALOG_LAST_UPDATED = "2026-10"
MARKET_HISTORY_PATH = Path(__file__).resolve().parent / "artifacts" / "market_snapshots.json"


# Indicative Maharashtra ex-showroom starting prices (INR). Keep this catalog
# versioned and review it whenever OEM prices or tax rules change.
MAHARASHTRA_EX_SHOWROOM: Dict[str, Dict[str, Any]] = {
    "Maruti Alto": {"brand": "Maruti", "price": 399000, "source": "OEM starting-price reference"},
    "Maruti Swift": {"brand": "Maruti", "price": 649000, "source": "OEM starting-price reference"},
    "Maruti Baleno": {"brand": "Maruti", "price": 699000, "source": "OEM starting-price reference"},
    "Maruti Brezza": {"brand": "Maruti", "price": 839000, "source": "OEM starting-price reference"},
    "Hyundai Grand i10 Nios": {"brand": "Hyundai", "price": 589000, "source": "OEM starting-price reference"},
    "Hyundai i20": {"brand": "Hyundai", "price": 719000, "source": "OEM starting-price reference"},
    "Hyundai Venue": {"brand": "Hyundai", "price": 799000, "source": "OEM starting-price reference"},
    "Tata Tiago": {"brand": "Tata", "price": 549000, "source": "OEM starting-price reference"},
    "Tata Altroz": {"brand": "Tata", "price": 679000, "source": "OEM starting-price reference"},
    "Tata Punch": {"brand": "Tata", "price": 619000, "source": "OEM starting-price reference"},
    "Tata Nexon": {"brand": "Tata", "price": 799000, "source": "OEM starting-price reference"},
    "Honda Amaze": {"brand": "Honda", "price": 799000, "source": "OEM starting-price reference"},
    "Honda City": {"brand": "Honda", "price": 1199000, "source": "OEM starting-price reference"},
    "Toyota Glanza": {"brand": "Toyota", "price": 689000, "source": "OEM starting-price reference"},
    "Toyota Urban Cruiser Hyryder": {"brand": "Toyota", "price": 1169000, "source": "OEM starting-price reference"},
    "Kia Sonet": {"brand": "Kia", "price": 799000, "source": "OEM starting-price reference"},
    "Mahindra XUV 3XO": {"brand": "Mahindra", "price": 799000, "source": "OEM starting-price reference"},
    "Maruti Fronx": {"brand": "Maruti", "price": 759000, "source": "OEM starting-price reference"},
    "Maruti Ertiga": {"brand": "Maruti", "price": 869000, "source": "OEM starting-price reference"},
    "Maruti Grand Vitara": {"brand": "Maruti", "price": 1070000, "source": "OEM starting-price reference"},
    "Hyundai Exter": {"brand": "Hyundai", "price": 629000, "source": "OEM starting-price reference"},
    "Hyundai Verna": {"brand": "Hyundai", "price": 1098000, "source": "OEM starting-price reference"},
    "Hyundai Creta": {"brand": "Hyundai", "price": 1111000, "source": "OEM starting-price reference"},
    "Tata Tigor": {"brand": "Tata", "price": 599000, "source": "OEM starting-price reference"},
    "Tata Curvv": {"brand": "Tata", "price": 999000, "source": "OEM starting-price reference"},
    "Tata Harrier": {"brand": "Tata", "price": 1499000, "source": "OEM starting-price reference"},
    "Honda Elevate": {"brand": "Honda", "price": 1191000, "source": "OEM starting-price reference"},
    "Toyota Rumion": {"brand": "Toyota", "price": 1030000, "source": "OEM starting-price reference"},
    "Toyota Innova Hycross": {"brand": "Toyota", "price": 1980000, "source": "OEM starting-price reference"},
    "Kia Seltos": {"brand": "Kia", "price": 1113000, "source": "OEM starting-price reference"},
    "Kia Carens": {"brand": "Kia", "price": 1045000, "source": "OEM starting-price reference"},
    "Mahindra Bolero": {"brand": "Mahindra", "price": 979000, "source": "OEM starting-price reference"},
    "Mahindra Scorpio N": {"brand": "Mahindra", "price": 1399000, "source": "OEM starting-price reference"},
    "Mahindra XUV700": {"brand": "Mahindra", "price": 1499000, "source": "OEM starting-price reference"},
    "Volkswagen Taigun": {"brand": "Volkswagen", "price": 1199000, "source": "OEM starting-price reference"},
    "Skoda Kushaq": {"brand": "Skoda", "price": 1099000, "source": "OEM starting-price reference"},
    "Renault Kwid": {"brand": "Renault", "price": 469000, "source": "OEM starting-price reference"},
    "Renault Kiger": {"brand": "Renault", "price": 629000, "source": "OEM starting-price reference"},
    "Nissan Magnite": {"brand": "Nissan", "price": 599000, "source": "OEM starting-price reference"},
}

MAINTENANCE_ADJUSTMENTS = {"Poor": -0.10, "Average": 0.0, "Excellent": 0.08}
DEMAND_ADJUSTMENTS = {"Low": -0.05, "Normal": 0.0, "High": 0.05}


def market_index(as_of: Optional[date] = None) -> Dict[str, Any]:
    """Return a deterministic, refreshable market index for the prediction date.

    The trained data is historical, so this index prevents old observations
    from being presented as today's value. It is deliberately exposed in the
    UI so the reference year and methodology remain visible.
    """

    observed_on = as_of or date.today()
    reference_year = 2024
    annual_change = 0.06
    years_since_reference = max(observed_on.year - reference_year, 0)
    seasonal_change = {1: 0.00, 2: 0.01, 3: 0.01, 4: 0.00, 5: -0.01, 6: -0.01,
                       7: 0.00, 8: 0.01, 9: 0.01, 10: 0.02, 11: 0.02, 12: 0.01}
    factor = (1 + annual_change) ** years_since_reference
    factor *= 1 + seasonal_change[observed_on.month]
    return {
        "as_of": observed_on,
        "month": observed_on.strftime("%Y-%m"),
        "reference_year": reference_year,
        "factor": factor,
        "label": f"Market index {observed_on.strftime('%b %Y')}",
        "method": "6% annual trend with a small seasonal adjustment; refresh with verified market data.",
        "source": "Built-in trend baseline",
    }


def load_market_snapshots() -> List[Dict[str, Any]]:
    """Load persisted monthly market snapshots, tolerating a first run."""

    if not MARKET_HISTORY_PATH.exists():
        return []
    try:
        with MARKET_HISTORY_PATH.open("r", encoding="utf-8") as snapshot_file:
            data = json.load(snapshot_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read market snapshots: {MARKET_HISTORY_PATH}") from exc
    if not isinstance(data, list):
        raise ValueError("Market snapshot file must contain a list.")
    return data


def refresh_market_snapshot(
    market_change_pct: float,
    source: str = "Verified monthly market update",
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Persist the current month's market factor and return the saved snapshot."""

    observed_on = as_of or date.today()
    month = observed_on.strftime("%Y-%m")
    snapshot = {
        "month": month,
        "as_of": observed_on.isoformat(),
        "factor": round(1 + float(market_change_pct) / 100, 4),
        "market_change_pct": round(float(market_change_pct), 2),
        "source": source.strip() or "Verified monthly market update",
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
    }
    snapshots = [item for item in load_market_snapshots() if item.get("month") != month]
    snapshots.append(snapshot)
    snapshots.sort(key=lambda item: item.get("month", ""))
    MARKET_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MARKET_HISTORY_PATH.with_suffix(".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as snapshot_file:
            json.dump(snapshots, snapshot_file, indent=2)
        temporary_path.replace(MARKET_HISTORY_PATH)
    except OSError as exc:
        raise RuntimeError(f"Unable to save market snapshot: {MARKET_HISTORY_PATH}") from exc
    return snapshot


def current_market_snapshot(as_of: Optional[date] = None) -> Dict[str, Any]:
    """Return the saved current-month snapshot or the transparent baseline."""

    observed_on = as_of or date.today()
    month = observed_on.strftime("%Y-%m")
    for snapshot in reversed(load_market_snapshots()):
        if snapshot.get("month") == month:
            snapshot_date = date.fromisoformat(snapshot["as_of"])
            return {
                **snapshot,
                "as_of": snapshot_date,
                "reference_year": 2024,
                "label": f"Market index {snapshot_date.strftime('%b %Y')}",
                "method": "Monthly refreshed market snapshot; update using verified Maharashtra market data.",
            }
    return market_index(observed_on)


def monthly_market_history(months: int = 12, as_of: Optional[date] = None) -> pd.DataFrame:
    """Build a monthly trend series, using saved snapshots where available."""

    if months not in (6, 12):
        raise ValueError("Market history must contain either 6 or 12 months.")
    observed_on = as_of or date.today()
    snapshots = {item.get("month"): item for item in load_market_snapshots()}
    records = []
    for offset in range(months - 1, -1, -1):
        month_number = observed_on.month - offset
        year = observed_on.year + (month_number - 1) // 12
        month = (month_number - 1) % 12 + 1
        month_date = date(year, month, 1)
        baseline = market_index(month_date)
        snapshot = snapshots.get(month_date.strftime("%Y-%m"))
        factor = float(snapshot["factor"]) if snapshot else baseline["factor"]
        records.append({
            "Month": month_date.strftime("%b %Y"),
            "Market index": round(factor, 4),
            "Source": snapshot["source"] if snapshot else baseline["source"],
        })
    return pd.DataFrame(records)


def adjust_prediction(
    prediction: float,
    maintenance: str,
    demand: str,
    index: Dict[str, Any],
) -> Dict[str, float]:
    """Apply transparent market, maintenance, and demand adjustments."""

    market_adjusted = prediction * float(index["factor"])
    maintenance_factor = 1 + MAINTENANCE_ADJUSTMENTS[maintenance]
    demand_factor = 1 + DEMAND_ADJUSTMENTS[demand]
    final_price = max(market_adjusted * maintenance_factor * demand_factor, 0.0)
    return {
        "market_adjusted": market_adjusted,
        "final_price": final_price,
        "market_change": market_adjusted - prediction,
        "condition_change": final_price - market_adjusted,
    }


def catalog_entry(model_name: str) -> Optional[Dict[str, Any]]:
    """Return a Maharashtra ex-showroom entry when the selected model is known."""

    return MAHARASHTRA_EX_SHOWROOM.get(model_name)
