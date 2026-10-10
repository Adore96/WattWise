"""
Sanity-check / calibrate the battery efficiency from the raw 5-minute logs.

The logged Battery(kW) is measured on the house (AC) side, the same side as
a_t in the model, and SOC(%) is the inverter's state of charge. Over any
interval the model says

    dSOC% = (100 / Cap) * ( eta_c * E_charge  -  E_discharge / eta_d )

which is linear in E_charge and E_discharge:

    dSOC% = b1 * E_charge - b2 * E_discharge,   b1 = 100 eta_c / Cap,
                                                b2 = 100 / (Cap * eta_d)

Least squares gives b1, b2. Only two quantities are identifiable from them:
the round-trip efficiency eta_c * eta_d = b1 / b2 and Cap. Splitting the
round trip symmetrically (eta_c = eta_d = eta) gives
    eta = sqrt(b1 / b2),   Cap = 100 * eta / b1.

SOC% is logged as an integer, so single 5-minute differences are mostly
quantisation noise; we regress on 30-minute differences instead. Windows that
touch >= 99% SOC are dropped: the BMS tapers/stops charging near full, which
breaks the linear relation.

A constant standby-drain term (regression intercept) was also tried: it
pushes eta above 1, which is physically impossible. The battery is idle in
only 1-8% of samples, so losses and a standby drain cannot be separated from
these logs; the no-intercept fit lumps both into eta (conservative for the
battery's value).

Usage (from the repo root):
    python scripts/calibrate_eta.py data/raw/*.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

STEP_H = 5 / 60          # 5-minute samples
WINDOW = 6               # 6 x 5 min = 30-minute differences
PLAUSIBLE_ETA = (0.85, 0.99)
PLAUSIBLE_CAP = (5.5, 8.5)  # kWh around the 7 kWh nameplate
FULL_SOC_PCT = 99


def load_log(path):
    df = pd.read_excel(path)
    df["t"] = pd.to_datetime(df["Time"])
    df = df.drop_duplicates("t").set_index("t").sort_index()
    batt = df[[c for c in df.columns if c.startswith("Battery")][0]].astype(float)
    soc = df[[c for c in df.columns if c.startswith("SOC")][0]].astype(float)
    return pd.DataFrame({"p": batt, "soc": soc})


def interval_energy(p):
    """Charge / discharge energy (kWh, house side) per 5-minute interval,
    trapezoid rule applied separately to the positive and negative parts."""
    dis = np.clip(p, 0, None)
    ch = np.clip(-p, 0, None)
    e_dis = (dis[:-1] + dis[1:]) / 2 * STEP_H
    e_ch = (ch[:-1] + ch[1:]) / 2 * STEP_H
    return e_ch, e_dis


def windows(log):
    """Non-overlapping 30-minute windows: (E_charge, E_discharge, dSOC%)."""
    p, soc = log["p"].to_numpy(), log["soc"].to_numpy()
    e_ch, e_dis = interval_energy(p)
    rows = []
    for s in range(0, len(p) - WINDOW, WINDOW):
        # skip windows that span a logging gap
        span = (log.index[s + WINDOW] - log.index[s]).total_seconds() / 60
        if abs(span - WINDOW * 5) > 1e-6:
            continue
        if soc[s:s + WINDOW + 1].max() >= FULL_SOC_PCT:
            continue
        rows.append((e_ch[s:s + WINDOW].sum(), e_dis[s:s + WINDOW].sum(),
                     soc[s + WINDOW] - soc[s]))
    return np.array(rows)


def fit(rows):
    X = np.column_stack([rows[:, 0], -rows[:, 1]])
    y = rows[:, 2]
    (b1, b2), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ np.array([b1, b2])
    round_trip = b1 / b2
    eta = np.sqrt(round_trip)
    cap = 100 * eta / b1
    return {"b1": b1, "b2": b2, "round_trip": round_trip, "eta": eta, "cap_kWh": cap,
            "rmse_pct": float(np.sqrt(np.mean(resid ** 2))), "n_windows": len(y)}


def replay(log, eta, cap):
    """Integrate the logged power through the model transition. SOC is
    clamped at 100 % like the real BMS (charging power logged while the
    battery is full is not stored)."""
    e_ch, e_dis = interval_energy(log["p"].to_numpy())
    soc = [log["soc"].iloc[0]]
    for c, d in zip(e_ch, e_dis):
        soc.append(min(100.0, soc[-1] + 100 / cap * (eta * c - d / eta)))
    return np.array(soc)


def main(paths):
    logs = {Path(p).stem: load_log(p) for p in paths}
    rows = np.vstack([windows(log) for log in logs.values()])
    res = fit(rows)

    print(f"30-min windows used: {res['n_windows']}")
    print(f"round-trip efficiency eta_c*eta_d = {res['round_trip']:.3f}")
    print(f"symmetric split eta = {res['eta']:.3f}   effective capacity = {res['cap_kWh']:.2f} kWh")
    print(f"fit RMSE = {res['rmse_pct']:.2f} SOC-% per 30 min")

    plausible = (PLAUSIBLE_ETA[0] <= res["eta"] <= PLAUSIBLE_ETA[1]
                 and PLAUSIBLE_CAP[0] <= res["cap_kWh"] <= PLAUSIBLE_CAP[1])
    print("plausible:", plausible)

    # replay error with the fitted values vs the 0.95 / 7 kWh placeholders
    per_day = []
    for name, log in logs.items():
        for label, eta, cap in [("fitted", res["eta"], res["cap_kWh"]), ("placeholder", 0.95, 7.0)]:
            sim = replay(log, eta, cap)
            err = sim - log["soc"].to_numpy()
            per_day.append({"day": name, "params": label,
                            "end_error_pct": round(float(err[-1]), 2),
                            "max_abs_error_pct": round(float(np.abs(err).max()), 2)})
    per_day = pd.DataFrame(per_day)
    print(per_day.to_string(index=False))

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame([res | {"plausible": plausible}]).to_csv(RESULTS / "eta_calibration.csv", index=False)
    per_day.to_csv(RESULTS / "eta_replay_errors.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(exist_ok=True)
    name = sorted(logs)[1]
    log = logs[name]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(log.index, log["soc"], color="#333", lw=1.6, label="logged SOC")
    ax.plot(log.index, replay(log, res["eta"], res["cap_kWh"]), color="#1a7f5a", lw=1.2,
            label=f"replay, fitted (eta={res['eta']:.3f}, C={res['cap_kWh']:.2f} kWh)")
    ax.plot(log.index, replay(log, 0.95, 7.0), color="#c0562a", lw=1.2, ls="--",
            label="replay, placeholder (eta=0.95, C=7 kWh)")
    ax.set_ylabel("SOC (%)")
    ax.set_title(f"Battery transition replay vs logged SOC ({name})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "eta_replay.png", dpi=150)
    return res, plausible


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pass the raw day files, e.g. data/raw/*.xlsx")
        sys.exit(1)
    main(sys.argv[1:])
