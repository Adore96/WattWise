"""
Shared problem definition for WattWise: battery parameters, data loading and
the schedule simulator / bill. Both the DP (dp_solver.py) and the GA
(heuristic_solver.py) import this module, so the two methods optimise exactly
the same model and are scored by exactly the same cost function.

Sign convention (matches the inverter logs): a_t > 0 discharge, a_t < 0 charge.
a_t is the battery energy on the HOUSE side in one slot (kWh).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "profiles_30min.csv"

SLOTS_PER_DAY = 48          # 30-minute slots
SLOT_HOURS = 0.5            # dt

CAPACITIES = [0, 2, 4, 7]   # kWh, the capacity sweep

SOC_RESERVE_FRAC = 0.2      # usable SoC range is [0.2*C, C]

CHARGE_RATE_KW = 2.2        # R_c, observed in the logs
DISCHARGE_RATE_KW = 2.5     # R_d, observed in the logs
MAX_CHARGE_KWH = CHARGE_RATE_KW * SLOT_HOURS          # 1.10 kWh per slot (house side)
MAX_DISCHARGE_KWH = DISCHARGE_RATE_KW * SLOT_HOURS    # 1.25 kWh per slot (house side)

# Charge / discharge efficiency, calibrated from the 5-minute logs by
# scripts/calibrate_eta.py: round trip eta_c*eta_d = 0.83, split symmetrically
# (fitted 0.912, effective capacity 6.86 kWh). Was 0.95 as a placeholder.
ETA_C = 0.91
ETA_D = 0.91

TOL = 1e-9                  # numerical tolerance for constraint checks


# ---------------------------------------------------------------- data

def load_profiles(path=DATA_PATH):
    """Load the cleaned 30-minute profile table."""
    return pd.read_csv(path)


def list_days(df):
    """All dates in the table, in order."""
    return sorted(df["date"].unique())


def get_day(df, date):
    """One day as 48-element arrays: solar, demand, price (plus labels)."""
    d = df[df["date"] == date].sort_values("slot_start").reset_index(drop=True)
    if len(d) != SLOTS_PER_DAY:
        raise ValueError(f"{date}: expected {SLOTS_PER_DAY} slots, got {len(d)}")
    return {
        "date": date,
        "solar": d["production_kWh"].to_numpy(float),
        "demand": d["consumption_kWh"].to_numpy(float),
        "price": d["price_LKR_per_kWh"].to_numpy(float),
        "band": d["band"].to_numpy(),
        "slot": d["slot_start"].to_numpy(),
    }


def get_horizon(df, dates):
    """Several days chained into one horizon (48 * len(dates) slots).
    Used for the horizon-length scalability experiment."""
    days = [get_day(df, d) for d in dates]
    out = {"date": " + ".join(dates)}
    for key in ("solar", "demand", "price", "band", "slot"):
        out[key] = np.concatenate([d[key] for d in days])
    return out


# ---------------------------------------------------------------- model

def soc_bounds(capacity, reserve_frac=SOC_RESERVE_FRAC):
    """Usable SoC range (kWh)."""
    return reserve_frac * capacity, float(capacity)


def grid_import(demand, solar, a):
    """Energy bought from the grid in a slot. A definition, not a constraint:
    the grid is an unlimited backstop and nothing is ever exported."""
    return np.maximum(0.0, demand - solar - a)


def simulate(schedule, day, capacity, eta_c=ETA_C, eta_d=ETA_D,
             max_charge=MAX_CHARGE_KWH, max_discharge=MAX_DISCHARGE_KWH,
             reserve_frac=SOC_RESERVE_FRAC, soc0=None):
    """
    Replay a dispatch schedule through the battery model.

    Returns bill (Rs), total constraint violation (kWh, 0 if legal),
    feasible flag and the SoC trajectory (SoC at the END of each slot).

    Constraints checked:
      rate       -max_charge <= a_t <= max_discharge
      no export  a_t <= max(0, d_t - g_t)
      SoC        soc_min <= SoC_t <= C
    The battery starts at soc0 (default: the reserve floor).
    """
    schedule = np.asarray(schedule, dtype=float)
    solar, demand, price = day["solar"], day["demand"], day["price"]
    n = len(price)
    if len(schedule) != n:
        raise ValueError(f"schedule has {len(schedule)} slots, day has {n}")

    soc_min, soc_max = soc_bounds(capacity, reserve_frac)
    soc = soc_min if soc0 is None else soc0
    soc_traj = np.zeros(n)
    bill = 0.0
    violation = 0.0

    for t in range(n):
        a = schedule[t]

        # rate limits (asymmetric)
        if a > max_discharge:
            violation += a - max_discharge
        elif -a > max_charge:
            violation += -a - max_charge

        # no export: discharge only up to the net load
        net_load = max(0.0, demand[t] - solar[t])
        if a > net_load:
            violation += a - net_load

        # SoC transition with efficiency on both sides
        if a >= 0:
            soc = soc - a / eta_d
        else:
            soc = soc + eta_c * (-a)

        if soc < soc_min:
            violation += soc_min - soc
        if soc > soc_max:
            violation += soc - soc_max
        soc_traj[t] = soc

        bill += price[t] * max(0.0, demand[t] - solar[t] - a)

    return {
        "bill": bill,
        "violation": violation,
        "feasible": violation < TOL,
        "soc": soc_traj,
    }


def baseline_bill(day):
    """Bill with no battery at all (C = 0)."""
    return float(np.sum(day["price"] * grid_import(day["demand"], day["solar"], 0.0)))
