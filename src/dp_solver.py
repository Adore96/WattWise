"""
Exact method: Dynamic Programming (backward induction) for battery dispatch.

Stage   t = slot (0 .. T-1)
State   i = SoC grid level, SoC_i = soc_min + i * delta   (i = 0 .. N)
Action  k = number of grid steps moved in the slot
        k > 0 discharge k steps, k < 0 charge |k| steps, k = 0 hold
        next state j = i - k, which is always ON the grid, so no rounding
        or interpolation of the value function is needed.

House-side energy of a move (from the settled transition equations):
    discharge k steps:  a = k * delta * eta_d        (SoC falls by a / eta_d)
    charge   |k| steps: a = k * delta / eta_c  (<0)  (SoC rises by |a| * eta_c)

Bellman recursion (V_T = 0, leftover energy has no value):
    V_t(i) = min_k [ price_t * max(0, d_t - g_t - a(k)) + V_{t+1}(i - k) ]
over moves that respect the rate limits, the SoC range and no-export
(a(k) <= max(0, d_t - g_t)).

The result is exactly optimal for the discretised model; delta controls how
closely that matches the continuous one.

Usage:
    python src/dp_solver.py --capacity 7 --date 2026-09-11 [--delta 0.001]
"""
import argparse
import math
import time

import numpy as np

import model

EPS = 1e-12

# Main-results grid step. The discretisation error is roughly linear in delta
# (one partial discharge step of load per slot cannot be matched exactly):
# on Sep 11 / 4 kWh the bill is 58.05 at 0.01, 54.63 at 0.005, 52.83 at 0.002
# and 52.39 at 0.001 (~1 % above the extrapolated continuous optimum).
DEFAULT_DELTA = 0.001


def soc_grid(capacity, delta, reserve_frac=model.SOC_RESERVE_FRAC):
    """SoC levels soc_min, soc_min + delta, ..., C."""
    soc_min, soc_max = model.soc_bounds(capacity, reserve_frac)
    span = (soc_max - soc_min) / delta
    n = int(round(span))
    if abs(span - n) > 1e-6:
        raise ValueError(f"usable range {soc_max - soc_min} kWh is not a multiple of delta={delta}")
    return soc_min + delta * np.arange(n + 1)


def step_limits(delta, eta_c=model.ETA_C, eta_d=model.ETA_D,
                max_charge=model.MAX_CHARGE_KWH, max_discharge=model.MAX_DISCHARGE_KWH):
    """Rate limits (house side) as caps on the number of grid steps per slot.
    Always round DOWN: rounding up would allow a move above the rate limit."""
    k_charge = math.floor(max_charge * eta_c / delta + 1e-9)        # |a| = k*delta/eta_c <= max_charge
    k_discharge = math.floor(max_discharge / (eta_d * delta) + 1e-9)  # a = k*delta*eta_d <= max_discharge
    return k_charge, k_discharge


def solve_dp(day, capacity, delta=None, eta_c=model.ETA_C, eta_d=model.ETA_D,
             max_charge=model.MAX_CHARGE_KWH, max_discharge=model.MAX_DISCHARGE_KWH,
             reserve_frac=model.SOC_RESERVE_FRAC, soc0=None):
    """
    Solve one horizon (any number of slots) exactly on the SoC grid.

    Returns a dict with the optimal schedule (house-side a_t), its bill, the
    SoC trajectory, the value table V[t, i], the policy table K[t, i] (optimal
    move in steps) and the solve time in seconds.
    """
    t0 = time.perf_counter()
    if delta is None:
        delta = DEFAULT_DELTA
    solar, demand, price = day["solar"], day["demand"], day["price"]
    T = len(price)
    net_load = np.maximum(0.0, demand - solar)

    if capacity == 0:
        levels = np.array([0.0])
        k_charge = k_discharge = 0
    else:
        levels = soc_grid(capacity, delta, reserve_frac)
        k_charge, k_discharge = step_limits(delta, eta_c, eta_d, max_charge, max_discharge)
    n_states = len(levels)
    k_charge = min(k_charge, n_states - 1)
    k_discharge = min(k_discharge, n_states - 1)

    # candidate moves, ordered by |k| so ties resolve to the smallest move (hold first)
    moves = sorted(range(-k_charge, k_discharge + 1), key=lambda k: (abs(k), k))
    a_of = {k: (k * delta * eta_d if k >= 0 else k * delta / eta_c) for k in moves}

    V = np.zeros((T + 1, n_states))               # V[T] = 0: terminal value
    K = np.zeros((T, n_states), dtype=np.int32)   # policy: best k at (t, i)
    idx = np.arange(n_states)

    for t in range(T - 1, -1, -1):                # backward induction
        best = np.full(n_states, np.inf)
        best_k = np.zeros(n_states, dtype=np.int32)
        for k in moves:
            a = a_of[k]
            if a > net_load[t] + EPS:             # no export
                continue
            j = idx - k                            # next state for every i
            ok = (j >= 0) & (j < n_states)        # stays inside [soc_min, C]
            cost_now = price[t] * max(0.0, demand[t] - solar[t] - a)
            q = np.full(n_states, np.inf)
            q[ok] = cost_now + V[t + 1, j[ok]]
            better = q < best - EPS
            best[better] = q[better]
            best_k[better] = k
        V[t] = best
        K[t] = best_k

    # forward pass: follow the policy from the start state
    soc_start = levels[0] if soc0 is None else soc0
    i = int(np.argmin(np.abs(levels - soc_start)))
    if abs(levels[i] - soc_start) > 1e-6:
        raise ValueError(f"start SoC {soc_start} is not on the grid")
    start_index = i
    schedule = np.zeros(T)
    soc_traj = np.zeros(T)
    for t in range(T):
        k = int(K[t, i])
        schedule[t] = a_of[k]
        i -= k
        soc_traj[t] = levels[i]

    runtime = time.perf_counter() - t0
    bill = float(np.sum(price * model.grid_import(demand, solar, schedule)))
    return {
        "schedule": schedule,
        "bill": bill,
        "optimal_value": float(V[0, start_index]),
        "soc": soc_traj,
        "V": V,
        "K": K,
        "levels": levels,
        "runtime_s": runtime,
        "n_states": n_states,
        "n_moves": len(moves),
    }


def main():
    p = argparse.ArgumentParser(description="Exact DP battery dispatch for one day")
    p.add_argument("--capacity", type=float, default=7)
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: every day)")
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="SoC grid step (kWh)")
    p.add_argument("--data", default=str(model.DATA_PATH))
    args = p.parse_args()

    df = model.load_profiles(args.data)
    dates = [args.date] if args.date else model.list_days(df)
    for date in dates:
        day = model.get_day(df, date)
        res = solve_dp(day, args.capacity, delta=args.delta)
        check = model.simulate(res["schedule"], day, args.capacity)
        base = model.baseline_bill(day)
        print(f"{date}  C={args.capacity:g} kWh  baseline Rs {base:7.2f}  "
              f"DP Rs {res['bill']:7.2f}  saving {100 * (base - res['bill']) / base:5.1f}%  "
              f"replay feasible={check['feasible']}  "
              f"{res['n_states']} states x {res['n_moves']} moves  {res['runtime_s'] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
