"""
Heuristic method: Genetic Algorithm for battery dispatch.

Author of the algorithm: Bimsara (developed in heuristic_energy_optimization.ipynb).
Moved here unchanged in logic so the notebook, the comparison script and the
DP all share one model (model.py). Changes when moving:
  - horizon length comes from the data (len(day["price"])) instead of a fixed
    48, so the same GA runs on chained multi-day horizons;
  - the greedy seed discharges only up to the NET load (demand - solar),
    because no-export is now a constraint in the shared simulate();
  - no-export is a per-slot upper bound on a_t, so it is handled exactly
    like the rate limits already were: by clipping each gene to its bounds.
    The penalty then only has to deal with the inter-slot SoC constraints.
    (Penalty-only handling left ~97% of mutants infeasible and the GA
    stalled near the greedy seed.)

Encoding: one individual = a vector of a_t (kWh, house side) for every slot,
positive = discharge, negative = charge.
Fitness: bill + penalty * total constraint violation (graded penalty).

Usage:
    python src/heuristic_solver.py --capacity 7 --date 2026-09-11 [--seed 1]
"""
import argparse
import time

import numpy as np

import model

PENALTY = 100000.0  # constraint-violation penalty weight


def fitness(schedule, day, capacity, penalty=PENALTY):
    """fitness = bill + penalty * violation (feasible schedules keep their true bill)."""
    res = model.simulate(schedule, day, capacity)
    res["fitness"] = res["bill"] + penalty * res["violation"]
    return res


def greedy_schedule(day, capacity, eta_c=model.ETA_C, eta_d=model.ETA_D,
                    max_charge=model.MAX_CHARGE_KWH, max_discharge=model.MAX_DISCHARGE_KWH,
                    reserve_frac=model.SOC_RESERVE_FRAC):
    """Feasible rule-based seed: charge from surplus solar, discharge at peak."""
    n = len(day["price"])
    s = np.zeros(n)
    soc_min, soc_max = model.soc_bounds(capacity, reserve_frac)
    soc = soc_min

    for t in range(n):
        surplus = day["solar"][t] - day["demand"][t]

        if surplus > 0:  # charge from surplus
            room = soc_max - soc
            charge = min(surplus, max_charge, room / eta_c if eta_c > 0 else room)
            charge = max(charge, 0.0)
            s[t] = -charge
            soc += eta_c * charge

        elif day["price"][t] >= 100:  # discharge at peak, never beyond the net load
            avail = soc - soc_min
            discharge = min(-surplus, max_discharge, avail * eta_d)
            discharge = max(discharge, 0.0)
            s[t] = discharge
            soc -= discharge / eta_d

    return s


def run_ga(day, capacity, pop_size=100, generations=200, cx_rate=0.8,
           mut_rate=0.1, mut_sigma=0.2, tournament_k=3, elites=2, seed=0):
    """Genetic Algorithm. Returns the best schedule, bill, feasibility, history and runtime."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(day["price"])
    # per-slot bounds: rate limits, and no-export caps discharge at the net load
    net_load = np.maximum(0.0, day["demand"] - day["solar"])
    lo = np.full(n, -model.MAX_CHARGE_KWH)
    hi = np.minimum(model.MAX_DISCHARGE_KWH, net_load)

    # initial population: mostly small-random, plus a greedy seed and a zero seed
    pop = rng.normal(0, 0.3, size=(pop_size, n)).clip(lo, hi)
    pop[0] = greedy_schedule(day, capacity)  # warm-start seed
    pop[1] = 0.0                             # do-nothing seed (always feasible)
    fits = np.array([fitness(ind, day, capacity)["fitness"] for ind in pop])
    history = []

    def tournament():
        idx = rng.integers(0, pop_size, tournament_k)
        return pop[idx[np.argmin(fits[idx])]]  # fittest of k random individuals

    for g in range(generations):
        order = np.argsort(fits)
        next_pop = [pop[order[i]].copy() for i in range(elites)]  # keep best few

        while len(next_pop) < pop_size:
            p1, p2 = tournament(), tournament()

            if rng.random() < cx_rate:
                a = rng.random(n)
                child = a * p1 + (1 - a) * p2  # blend (arithmetic) crossover
            else:
                child = p1.copy()

            m = rng.random(n) < mut_rate       # Gaussian mutation
            child[m] += rng.normal(0, mut_sigma, size=m.sum())
            child = child.clip(lo, hi)         # clamp to rate limits and no-export
            next_pop.append(child)

        pop = np.array(next_pop)
        fits = np.array([fitness(ind, day, capacity)["fitness"] for ind in pop])
        history.append(fits.min())

    best = pop[np.argmin(fits)]
    res = fitness(best, day, capacity)

    return {
        "schedule": best,
        "bill": res["bill"],
        "feasible": res["feasible"],
        "violation": res["violation"],
        "fitness": res["fitness"],
        "soc": res["soc"],
        "history": history,
        "runtime_s": time.perf_counter() - t0,
    }


def main():
    p = argparse.ArgumentParser(description="GA battery dispatch for one day")
    p.add_argument("--capacity", type=float, default=7)
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: every day)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data", default=str(model.DATA_PATH))
    args = p.parse_args()

    df = model.load_profiles(args.data)
    dates = [args.date] if args.date else model.list_days(df)
    for date in dates:
        day = model.get_day(df, date)
        res = run_ga(day, args.capacity, seed=args.seed)
        base = model.baseline_bill(day)
        print(f"{date}  C={args.capacity:g} kWh  baseline Rs {base:7.2f}  "
              f"GA Rs {res['bill']:7.2f}  feasible={res['feasible']}  {res['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
