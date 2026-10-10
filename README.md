# Dispatch Duo — Battery Dispatch Optimization vs CEB Time-of-Use Tariffs

Optimization Methods (MSc AI) group assignment. We optimize when a home battery should charge or discharge over the day to minimize the electricity bill under Sri Lanka's CEB Time-of-Use tariff, comparing an exact method (Dynamic Programming) against a heuristic (Genetic Algorithm / Simulated
Annealing) across a sweep of battery sizes.

## Team

| Member  | Role                                                                |
|---------|---------------------------------------------------------------------|
| Kasun   | Data pipeline, DP (exact method) solver, results & discussion       |
| Bimsara | Heuristic (GA/SA) solver, mathematical formulation, report sections |

Split by module, not by task — see [`docs/task-board.md`](docs/task-board.md) (or the shared board link in `submission.txt`) for the full breakdown and week-by-week plan.

## Problem summary

Decide how much to charge or discharge a home battery in every 30-minute slot of the day, to minimize the cost of electricity bought from the grid under a tariff that changes by time of day:

| Band     | Hours       | Rate       |
|----------|-------------|------------|
| Off-Peak | 22:30–05:30 | Rs 33/kWh  |
| Day      | 05:30–18:30 | Rs 47/kWh  |
| Peak     | 18:30–22:30 | Rs 106/kWh |

*(CEB optional domestic Time-of-Use tariff, May 2026 revision — reverify against the current official CEB/PUCSL sheet before the tariffs go in the final report.)*

Because the real household's 7 kWh battery already makes it close to grid-independent (~6 kWh/month drawn from the grid), we don't optimize a single fixed capacity — we sweep **battery capacity across 0 / 2 / 4 / 7 kWh**, using the same real load and solar data at every size, so the comparison shows
*where* optimized dispatch actually matters rather than a single, thin result.

## Data

Five real days of household inverter/monitoring logs, 5-minute resolution, resampled to 30-minute slots (aligned to the tariff boundaries above, which fall on the half-hour):

- **Source**: Kasun's own solar + battery system (Production, Consumption, Grid, Battery power, Battery SOC%). Not synthetic, not third-party.
- **Days used**: Sep 10, 11, 13, 16, 17, 2026.
- **Sign convention**: positive `Battery(kW)` = discharging, negative = charging (verified against the logs, not assumed).
- **Known data notes**: Sep 12's original log had a ~3-hour gap (39 missing 5-minute rows) during the production ramp and was dropped in favor of a re-logged Sep 13; it is kept in `data/raw/excluded/` as evidence and is not picked up by the pipeline. Sep 11 is missing one 5-minute reading at 05:20 (interpolated, negligible). Full write-up of every data issue and fix is in the report's Data Description
  section.
  The five raw `.xlsx` logs are committed in `data/raw/`; run `scripts/build_profiles.py` to regenerate `data/profiles_30min.csv` from them (the output is reproduced exactly).

## Repository structure

```
.
├── data/
│   ├── raw/                          # source .xlsx logs, one per day (5 days, committed)
│   │   └── excluded/                 # 2026-09-12.xlsx: faulty log (3 h gap), kept as evidence, not used
│   └── profiles_30min.csv            # cleaned, resampled output: 5 days × 48 slots
├── scripts/
│   ├── build_profiles.py             # raw 5-min logs -> tariff-aligned 30-min profile table
│   └── calibrate_eta.py              # fits battery efficiency + capacity to the logged SOC
├── src/
│   ├── model.py                      # shared problem: data, parameters, simulate() (bill + constraint check)
│   ├── dp_solver.py                  # exact method: backward-induction DP over a discretised SoC grid
│   ├── heuristic_solver.py           # heuristic: Genetic Algorithm (Bimsara)
│   └── compare.py                    # DP vs GA experiments -> results/*.csv
├── tests/
│   └── test_dp_hand_example.py       # DP hand example (optimal 33 vs 106 doing nothing)
├── notebooks/
│   └── dp_solver.ipynb               # DP walkthrough, calibration, comparison plots
├── heuristic_energy_optimization.ipynb  # GA walkthrough and experiments
├── results/                          # experiment outputs (CSV) and machine spec
├── figures/                          # plots for the report
├── report/                           #                                                    [planned]
│   ├── report.pdf
│   └── code_appendix.txt
├── docs/
│   └── task-board.md                 #                                                    [planned]
├── requirements.txt
├── pytest.ini
├── members.txt                       #                                                    [planned]
├── submission.txt                    #                                                    [planned]
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists pandas, numpy, openpyxl (reads the `.xlsx` logs), matplotlib, pytest and the
Jupyter pieces needed to run the notebooks. The DP is hand-rolled backward induction and the GA is
hand-written, so no solver library is required.

## Usage

All commands run from the repo root.

```bash
# 1. Rebuild the profile table from raw logs (only needed if you add/replace a day)
python scripts/build_profiles.py data/raw/*.xlsx

# 2. Calibrate the battery efficiency from the raw logs
python scripts/calibrate_eta.py data/raw/*.xlsx

# 3. Tests (DP hand example)
python -m pytest

# 4. DP for one capacity (every day, or one --date); --delta sets the SoC grid step
python src/dp_solver.py --capacity 7 [--date 2026-09-11] [--delta 0.001]

# 5. GA for one capacity
python src/heuristic_solver.py --capacity 7 [--date 2026-09-11] [--seed 1]

# 6. Full comparison: capacity sweep, grid-step and horizon scalability, eta sensitivity (~15 min)
python src/compare.py            # --quick for a 3-seed smoke test
```

## Methodology

- **Exact method — Dynamic Programming.** Stage: 30-minute slot. State: SoC level on a grid
  `0.2·C, 0.2·C + δ, …, C`. Action: move `k` grid steps (next state always on the grid, so no rounding
  or value-function interpolation); the house-side energy is backed out of the move
  (`a = k·δ·η_d` discharging, `a = k·δ/η_c` charging). Solved by backward induction (`V_T = 0`), and the
  schedule is recovered by a forward pass through the policy table. Exact for the discretised model;
  δ = 0.001 kWh keeps the bill within ~1 % of the continuous optimum (see `notebooks/dp_solver.ipynb`).
  Validated against a hand-solved example (`tests/`), and every DP schedule is replayed through the
  shared `simulate()` to confirm it is feasible and costs what the DP claims.
- **Heuristic — Genetic Algorithm.** One individual = the vector of `a_t` for every slot. Tournament
  selection, blend crossover, Gaussian mutation, elitism; warm-started with a greedy rule. Per-slot
  bounds (rate limits, no export) are enforced by clipping; the SoC limits, which couple slots, by a
  graded penalty (`fitness = bill + 10^5 · violation`).
- **Both methods import `src/model.py`**, so they optimise the same model and are scored by the same
  `simulate()`.
- **Evaluation.** Solution quality (gap to the DP), feasibility (rate over 10 GA seeds; DP replay check),
  runtime (median of repeats, same machine) and scalability (SoC grid step, horizon of 1 / 2 / 5 chained
  days), across 5 real days × 4 battery sizes.

### Model assumptions

| Item               | Rule                                                                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Usable capacity    | `0.2·C ≤ SoC_t ≤ C` (20% reserve; confirm against datasheet). C ∈ {0, 2, 4, 7} kWh                                                   |
| Rate               | `−R_c·Δt ≤ a_t ≤ R_d·Δt`, Δt = 0.5 h (logs show ~2.2 kW charge / ~2.5 kW discharge)                                                  |
| SoC transition     | discharge: `SoC − a_t/η_d`; charge: `SoC + \|a_t\|·η_c`, η_c = η_d = 0.91 (round trip 0.83, fitted to the logs by `scripts/calibrate_eta.py`) |
| No export          | `a_t ≤ max(0, d_t − g_t)`; surplus solar beyond load + battery is curtailed at zero value                                            |
| Grid import / cost | `grid_t = max(0, d_t − g_t − a_t)`; cost = `Σ price_t · grid_t` (a definition, not a constraint — the grid is an unlimited backstop) |
| Grid charging      | allowed in the model; verify against the inverter's actual configuration                                                             |
| Start / end of day | every day starts at the reserve floor `0.2·C`; energy left at the end has no value (`V_T = 0`)                                       |

## Results (summary)

Means over the 5 real days. DP at δ = 0.001 kWh, η = 0.91, GA over 10 seeds (pop 100 × 200 generations).
Full tables in `results/`, plots in `figures/`, walkthrough in `notebooks/dp_solver.ipynb`.

| Battery | No battery | Greedy rule | GA (mean / best seed) | DP (exact) | DP saving | GA feasible | Runtime DP / GA |
|---------|-----------:|------------:|----------------------:|-----------:|----------:|------------:|----------------:|
| 2 kWh   | Rs 196.7   | Rs 103.4    | Rs 89.3 / 84.9        | Rs 76.8    | 61 %      | 100 %       | 0.9 s / 1.4 s   |
| 4 kWh   | Rs 196.7   | Rs 93.3     | Rs 66.5 / 63.4        | Rs 56.9    | 71 %      | 100 %       | 1.5 s / 1.6 s   |
| 7 kWh   | Rs 196.7   | Rs 93.3     | Rs 62.2 / 59.4        | Rs 54.8    | 72 %      | 100 %       | 2.2 s / 1.6 s   |

- Every DP schedule is feasible on replay, and no GA run beat the DP. The GA's mean gap is 15–24 %.
- **Grid step:** DP work grows as 1/δ². The bill converges roughly linearly in δ (Rs 59.6 at δ = 0.01,
  Rs 54.8 at δ = 0.001).
- **Horizon (1 → 2 → 5 chained days, 7 kWh):** the DP stays exact, with runtime growing linearly (1.9 → 9.8 s).
  The GA's gap with the same budget grows 9 % → 118 % → 486 %.
- **Start-at-floor assumption:** 5 chained days cost Rs 79, versus Rs 274 when each day is solved
  independently from the 20 % floor. Carrying charge overnight is the biggest remaining saving.

## Status

- [x] Problem formulation & scope
- [x] Real data collected, cleaned, and resampled to 30-minute tariff-aligned slots
- [x] DP solver implemented & validated (hand example, replay check)
- [x] Heuristic solver implemented & validated
- [x] Battery efficiency calibrated from the logs
- [x] Comparative evaluation (cost / feasibility / runtime / scalability)
- [ ] Report, code appendix, and video

## References

- CEB optional domestic Time-of-Use tariff, May 2026 revision.
 