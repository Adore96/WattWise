"""
DP vs GA comparison: solution quality, feasibility, runtime and scalability.

Experiments (all results written to results/*.csv):
  1. sweep      5 days x capacities {0, 2, 4, 7} kWh.
                DP: bill, replay feasibility, median runtime of repeats.
                GA: N seeds -> best/mean/worst bill, feasible rate, gap to DP,
                    median runtime, generations to get within 1 % of DP.
  2. delta      DP scalability in the SoC grid step (all days, C = 7).
  3. horizon    1 / 2 / 5 chained days (C = 7): DP vs GA as the horizon grows.
  4. eta        DP sensitivity to the efficiency (0.91 calibrated, 0.95, 1.0).

Timing: time.perf_counter around the solve only (data loading excluded),
median over repeats / seeds, all on the same machine (results/machine.txt).

Usage (from the repo root):
    python src/compare.py               # full run (~15 min)
    python src/compare.py --quick       # 3 seeds, fewer repeats (smoke test)
"""
import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

import model
from dp_solver import solve_dp
from heuristic_solver import greedy_schedule, run_ga

RESULTS = model.ROOT / "results"

DELTAS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
HORIZONS = [1, 2, 5]
ETAS = [0.91, 0.95, 1.0]
SCALING_CAPACITY = 7


def timed_dp(day, capacity, repeats, **kw):
    """Solve repeatedly; return the last result and the median runtime."""
    times = []
    for _ in range(repeats):
        res = solve_dp(day, capacity, **kw)
        times.append(res["runtime_s"])
    return res, statistics.median(times)


def gens_to_within(history, target, tol=0.01):
    """First generation (1-based) whose best fitness is within tol of target."""
    for g, f in enumerate(history, start=1):
        if f <= target * (1 + tol) + 1e-9:
            return g
    return np.nan


def ga_runs(day, capacity, seeds, dp_bill, label):
    rows = []
    for s in seeds:
        r = run_ga(day, capacity, seed=s)
        rows.append({
            **label, "seed": s, "bill": r["bill"], "feasible": bool(r["feasible"]),
            "violation": r["violation"], "runtime_s": r["runtime_s"],
            "gap_pct": 100 * (r["bill"] - dp_bill) / dp_bill if dp_bill > 0 else 0.0,
            "gens_to_1pct": gens_to_within(r["history"], dp_bill),
        })
    return rows


def summarise_ga(runs):
    df = pd.DataFrame(runs)
    feas = df[df["feasible"]]
    return {
        "ga_best": feas["bill"].min() if len(feas) else np.nan,
        "ga_mean": feas["bill"].mean() if len(feas) else np.nan,
        "ga_worst": feas["bill"].max() if len(feas) else np.nan,
        "ga_feasible_rate": df["feasible"].mean(),
        "ga_gap_best_pct": feas["gap_pct"].min() if len(feas) else np.nan,
        "ga_gap_mean_pct": feas["gap_pct"].mean() if len(feas) else np.nan,
        "ga_runtime_med_s": df["runtime_s"].median(),
        "ga_gens_to_1pct_med": df["gens_to_1pct"].median(),
        "ga_reached_1pct_rate": df["gens_to_1pct"].notna().mean(),
    }


def exp_sweep(df, seeds, repeats):
    summary, all_runs = [], []
    for date in model.list_days(df):
        day = model.get_day(df, date)
        base = model.baseline_bill(day)
        for cap in model.CAPACITIES:
            dp, dp_t = timed_dp(day, cap, repeats)
            replay = model.simulate(dp["schedule"], day, cap)
            greedy = model.simulate(greedy_schedule(day, cap), day, cap)
            label = {"date": date, "capacity": cap}
            runs = ga_runs(day, cap, seeds, dp["bill"], label)
            all_runs += runs
            row = {**label, "baseline": base, "greedy": greedy["bill"],
                   "dp_bill": dp["bill"], "dp_replay_bill": replay["bill"],
                   "dp_feasible": bool(replay["feasible"]), "dp_runtime_med_s": dp_t,
                   "dp_states": dp["n_states"], "dp_moves": dp["n_moves"],
                   "dp_saving_pct": 100 * (base - dp["bill"]) / base,
                   **summarise_ga(runs)}
            summary.append(row)
            print(f"  sweep {date} C={cap}: base {base:7.2f}  DP {dp['bill']:7.2f} "
                  f"({dp_t:.2f}s)  GA best {row['ga_best']:7.2f} mean {row['ga_mean']:7.2f} "
                  f"feas {row['ga_feasible_rate']:.0%} ({row['ga_runtime_med_s']:.1f}s)")
    return pd.DataFrame(summary), pd.DataFrame(all_runs)


def exp_delta(df, repeats):
    rows = []
    for date in model.list_days(df):
        day = model.get_day(df, date)
        for d in DELTAS:
            res, t = timed_dp(day, SCALING_CAPACITY, repeats, delta=d)
            rows.append({"date": date, "capacity": SCALING_CAPACITY, "delta": d,
                         "n_states": res["n_states"], "n_moves": res["n_moves"],
                         "dp_bill": res["bill"], "runtime_med_s": t})
        print(f"  delta {date}: done")
    return pd.DataFrame(rows)


def exp_horizon(df, seeds, repeats):
    days = model.list_days(df)
    rows, runs_all = [], []
    for h in HORIZONS:
        dates = days[:h]
        hor = model.get_horizon(df, dates)
        dp, dp_t = timed_dp(hor, SCALING_CAPACITY, repeats)
        replay = model.simulate(dp["schedule"], hor, SCALING_CAPACITY)
        # same days solved independently (each starting at the floor) for reference
        indep = sum(solve_dp(model.get_day(df, d), SCALING_CAPACITY)["bill"] for d in dates)
        label = {"horizon_days": h, "slots": len(hor["price"])}
        runs = ga_runs(hor, SCALING_CAPACITY, seeds, dp["bill"], label)
        runs_all += runs
        rows.append({**label, "dates": " + ".join(dates), "capacity": SCALING_CAPACITY,
                     "baseline": model.baseline_bill(hor), "dp_bill": dp["bill"],
                     "dp_independent_days_bill": indep, "dp_feasible": bool(replay["feasible"]),
                     "dp_runtime_med_s": dp_t, **summarise_ga(runs)})
        print(f"  horizon {h} day(s): DP {dp['bill']:.2f} ({dp_t:.2f}s)  "
              f"GA mean {rows[-1]['ga_mean']:.2f} ({rows[-1]['ga_runtime_med_s']:.1f}s)")
    return pd.DataFrame(rows), pd.DataFrame(runs_all)


def exp_eta(df):
    rows = []
    for date in model.list_days(df):
        day = model.get_day(df, date)
        base = model.baseline_bill(day)
        for cap in [2, 4, 7]:
            for eta in ETAS:
                res = solve_dp(day, cap, eta_c=eta, eta_d=eta)
                rows.append({"date": date, "capacity": cap, "eta": eta, "dp_bill": res["bill"],
                             "saving_pct": 100 * (base - res["bill"]) / base})
    print("  eta: done")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="DP vs GA comparison")
    p.add_argument("--quick", action="store_true", help="3 GA seeds, 1 repeat")
    p.add_argument("--only", nargs="*", choices=["sweep", "delta", "horizon", "eta"])
    args = p.parse_args()

    seeds = [1, 2, 3] if args.quick else list(range(1, 11))
    repeats = 1 if args.quick else 5
    todo = args.only or ["sweep", "delta", "horizon", "eta"]
    RESULTS.mkdir(exist_ok=True)
    df = model.load_profiles()

    (RESULTS / "machine.txt").write_text(
        f"{platform.platform()}\n{platform.processor()}\nPython {platform.python_version()}\n"
        f"numpy {np.__version__}, pandas {pd.__version__}\n"
        f"GA seeds: {len(seeds)}, DP timing repeats: {repeats}\n"
        f"eta_c = eta_d = {model.ETA_C}\n", encoding="utf-8")

    t0 = time.perf_counter()
    if "sweep" in todo:
        summary, runs = exp_sweep(df, seeds, repeats)
        summary.to_csv(RESULTS / "sweep_summary.csv", index=False)
        runs.to_csv(RESULTS / "sweep_ga_runs.csv", index=False)
    if "delta" in todo:
        exp_delta(df, min(repeats, 3)).to_csv(RESULTS / "scaling_delta.csv", index=False)
    if "horizon" in todo:
        hz, hz_runs = exp_horizon(df, seeds, min(repeats, 3))
        hz.to_csv(RESULTS / "scaling_horizon.csv", index=False)
        hz_runs.to_csv(RESULTS / "scaling_horizon_ga_runs.csv", index=False)
    if "eta" in todo:
        exp_eta(df).to_csv(RESULTS / "eta_sensitivity.csv", index=False)
    print(f"done in {time.perf_counter() - t0:.0f} s -> {RESULTS}")


if __name__ == "__main__":
    main()
