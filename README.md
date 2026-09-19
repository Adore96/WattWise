# Dispatch Duo — Battery Dispatch Optimization vs CEB Time-of-Use Tariffs

Optimization Methods (MSc AI) group assignment. We optimize when a home battery should charge or discharge over the day to minimize the electricity bill under Sri Lanka's CEB Time-of-Use tariff, comparing an exact method (Dynamic Programming) against a heuristic (Genetic Algorithm / Simulated Annealing) across a sweep of battery sizes.

## Team

| Member | Role |
|---|---|
| Kasun | Data pipeline, DP (exact method) solver, results & discussion |
| Bimsara | Heuristic (GA/SA) solver, mathematical formulation, report sections |

Split by module, not by task — see [`docs/task-board.md`](docs/task-board.md) (or the shared board link in `submission.txt`) for the full breakdown and week-by-week plan.

## Problem summary

Decide how much to charge or discharge a home battery in every 30-minute slot of the day, to minimize the cost of electricity bought from the grid under a tariff that changes by time of day:

| Band | Hours | Rate |
|---|---|---|
| Off-Peak | 22:30–05:30 | Rs 33/kWh |
| Day | 05:30–18:30 | Rs 47/kWh |
| Peak | 18:30–22:30 | Rs 106/kWh |

*(CEB optional domestic Time-of-Use tariff, May 2026 revision — reverify against the current official CEB/PUCSL sheet before the tariffs go in the final report.)*

Because the real household's 7 kWh battery already makes it close to grid-independent (~6 kWh/month drawn from the grid), we don't optimize a single fixed capacity — we sweep **battery capacity across 0 / 2 / 4 / 7 kWh**, using the same real load and solar data at every size, so the comparison shows *where* optimized dispatch actually matters rather than a single, thin result.

## Data

Five real days of household inverter/monitoring logs, 5-minute resolution, resampled to 30-minute slots (aligned to the tariff boundaries above, which fall on the half-hour):

- **Source**: Kasun's own solar + battery system (Production, Consumption, Grid, Battery power, Battery SOC%). Not synthetic, not third-party.
- **Days used**: Sep 10, 11, 13, 16, 17, 2026.
- **Sign convention**: positive `Battery(kW)` = discharging, negative = charging (verified against the logs, not assumed).
- **Known data notes**: Sep 12's original log had a ~3-hour gap during the production ramp and was dropped in favor of a re-logged Sep 13; Sep 11 is missing one 5-minute reading at 05:20 (interpolated, negligible). Full write-up of every data issue and fix is in the report's Data Description section.

Raw logs and the resampling step aren't checked in raw here for size/privacy — see `scripts/build_profiles.py` to regenerate `data/profiles_30min.csv` from your own `.xlsx` exports.

## Repository structure

```
.
├── data/
│   ├── raw/                    # source .xlsx logs, one per day (not all committed — see note above)
│   └── profiles_30min.csv      # cleaned, resampled output: 5 days × 48 slots
├── scripts/
│   └── build_profiles.py       # raw 5-min logs -> tariff-aligned 30-min profile table
├── src/
│   ├── dp_solver.py            # exact method: backward-induction DP over discretized SoC          [planned]
│   ├── heuristic_solver.py     # GA/SA over the 24h dispatch vector                                  [planned]
│   └── compare.py              # runs both solvers across the capacity sweep, produces results       [planned]
├── notebooks/                  # exploratory work, plots for the report                              [planned]
├── report/
│   ├── report.pdf
│   └── code_appendix.txt
├── docs/
│   └── task-board.md
├── requirements.txt
├── members.txt
├── submission.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (adjust once the solvers are built — DP here is hand-rolled backward induction, so no solver library is required unless the heuristic implementation pulls one in):

```
pandas
numpy
openpyxl
matplotlib
```

## Usage

```bash
# 1. Rebuild the profile table from raw logs (only needed if you add/replace a day)
python scripts/build_profiles.py data/raw/*.xlsx

# 2. Run the DP solver for one battery capacity                         [planned]
python src/dp_solver.py --capacity 7 --data data/profiles_30min.csv

# 3. Run the heuristic solver for one battery capacity                  [planned]
python src/heuristic_solver.py --capacity 7 --data data/profiles_30min.csv

# 4. Run the full comparison across the capacity sweep and all 5 days   [planned]
python src/compare.py --capacities 0 2 4 7 --data data/profiles_30min.csv
```

## Methodology

- **Exact method — Dynamic Programming.** State: (30-minute slot, discretized battery SoC level). Decision: charge/discharge amount per slot, bounded by rate limits and battery capacity, with round-trip efficiency applied on transition. Solved by backward induction, giving the provably optimal schedule for a given day and battery capacity.
- **Heuristic — GA or SA.** A day's schedule encoded as a 48-length vector of charge/discharge decisions; fitness is total cost with a penalty for constraint violations. Validated against DP's known-optimal answer on small cases before being used where DP doesn't scale (e.g. a longer or stochastic horizon).
- **Evaluation.** Solution quality (cost), runtime, and scalability, compared across the four battery-capacity scenarios and all five real days (20 runs total).

## Status

- [x] Problem formulation & scope
- [x] Real data collected, cleaned, and resampled to 30-minute tariff-aligned slots
- [x] DP solver implemented & validated
- [ ] Heuristic solver implemented & validated
- [ ] Comparative evaluation (cost / runtime / scalability across the capacity sweep)
- [ ] Report, code appendix, and video

## Submission checklist

Single ZIP (`Optimization-assignment.zip`) containing `members.txt`, `submission.txt` (dataset + GitHub + YouTube links), this repo with a genuine commit history, the 15-minute YouTube video, the PDF report, and a plain-text code appendix. Missing components score zero; Turnitin capped at 20%; one upload only; late resubmission caps at 45%.

## References

- CEB optional domestic Time-of-Use tariff, May 2026 revision.
