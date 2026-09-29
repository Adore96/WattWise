"""
Turns raw 5-minute inverter/battery logs (one .xlsx per day, exported from
the monitoring app) into a clean 30-minute profile table ready for the
DP / heuristic dispatch models.

Why 30-minute slots: CEB's optional domestic Time-of-Use tariff switches
at 05:30, 18:30 and 22:30 -- all half-hour marks. A 30-minute slot is the
coarsest resolution that never splits a single time-step across two
different tariff rates (hourly slots would: 05:00-06:00, 18:00-19:00 and
22:00-23:00 would each mix two prices).

Usage:
    python build_profiles.py *.xlsx
    (or edit FILES below to point at your own day files)
"""
import re
import sys
import pandas as pd

DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')

FILES = {
    # 'YYYY-MM-DD': 'path/to/that/day.xlsx'
}

TARIFF_BANDS = [
    (5.5, 18.5, 'Day', 47.0),
    (18.5, 22.5, 'Peak', 106.0),
    # anything else (22:30-05:30, wrapping past midnight) is Off-Peak
]
OFF_PEAK_PRICE = 33.0


def tariff_band(hour_float: float):
    for start, end, name, price in TARIFF_BANDS:
        if start <= hour_float < end:
            return name, price
    return 'Off-Peak', OFF_PEAK_PRICE


def build_day_profile(day: str, path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df['t'] = pd.to_datetime(df['Time'])
    df = df.drop_duplicates('t').set_index('t').sort_index().drop(columns=['Time'])

    # reindex onto the complete 5-minute grid for the day and linearly
    # interpolate any gaps (logger dropouts, wifi hiccups, etc.) -- flagged
    # per-day so you know which days had a repair applied.
    full_idx = pd.date_range(f'{day} 00:00:00', f'{day} 23:55:00', freq='5min')
    n_missing = len(full_idx.difference(df.index))
    df = df.reindex(full_idx).interpolate(method='time').ffill().bfill()

    prod = df['Production（kW）'].resample('30min').mean()
    cons = df['Consumption（kW）'].resample('30min').mean()
    soc = df['SOC（%）'].resample('30min').last()

    out = pd.DataFrame({
        'date': day,
        'slot_start': prod.index.strftime('%H:%M'),
        'production_kWh': (prod * 0.5).round(3).values,
        'consumption_kWh': (cons * 0.5).round(3).values,
        'soc_pct_end': soc.round(1).values,
    })
    hour_float = prod.index.hour + prod.index.minute / 60
    out[['band', 'price_LKR_per_kWh']] = [tariff_band(h) for h in hour_float]
    out['gap_interpolated'] = n_missing > 0
    out['n_missing_5min_rows'] = n_missing
    return out


def main(files: dict[str, str]):
    frames = [build_day_profile(day, path) for day, path in files.items()]
    profiles = pd.concat(frames, ignore_index=True)
    profiles.to_csv('profiles_30min.csv', index=False)
    print(f'Wrote profiles_30min.csv: {len(profiles)} rows '
          f'({len(files)} days x 48 slots)')
    print(profiles.groupby('date')[['production_kWh', 'consumption_kWh']].sum().round(2))


def day_from_filename(path: str) -> str:
    """Pull a YYYY-MM-DD date out of a file, regardless of any prefix the
    export/upload process tacked on (e.g. '855e01ba-2026-09-10.xlsx').
    Falls back to the workbook's sheet name if the filename itself has no
    date in it -- some exports from the monitoring app are named things
    like 'PlantsDetails-History.xlsx' and only carry the date on the sheet
    tab (we hit exactly this with the Sep 13 replacement file)."""
    match = DATE_RE.search(path.split('/')[-1])
    if match:
        return match.group(1)

    sheet_names = pd.ExcelFile(path).sheet_names
    for name in sheet_names:
        match = DATE_RE.search(str(name))
        if match:
            return match.group(1)

    raise ValueError(
        f"Can't find a YYYY-MM-DD date in the filename or sheet names of {path!r} "
        f"(sheets: {sheet_names}). Either rename the file, or fill in FILES by hand."
    )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        files = {day_from_filename(p): p for p in sys.argv[1:]}
    else:
        files = FILES
    if not files:
        print('No files given. Pass day files as args, or fill in FILES in the script.')
        sys.exit(1)
    main(files)
