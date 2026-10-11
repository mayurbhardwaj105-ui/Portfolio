"""Generate realistic sample restaurant history so the app works out of the box.

Replace data/restaurant_history.csv with your own data (same columns) to use real numbers.

Columns:
    date         - YYYY-MM-DD
    customers    - number of people who visited that day
    temperature  - average temperature in degrees C
    rain_mm      - rainfall in mm
    is_holiday   - 1 if public holiday / festival, else 0
    promotion    - 1 if you ran an offer/discount, else 0
"""
from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAYS = {
    (1, 1), (1, 26), (3, 25), (8, 15), (10, 2), (10, 20), (11, 9), (12, 25),
}


def generate(days: int = 730, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq="D")

    rows = []
    for d in dates:
        month, dow = d.month, d.dayofweek
        temp = 25 + 10 * np.sin((month - 4) / 12 * 2 * np.pi) + rng.normal(0, 2)
        rain = max(0.0, rng.normal(2, 6)) if month in (6, 7, 8, 9) else max(0.0, rng.normal(-3, 3))
        holiday = int((d.month, d.day) in HOLIDAYS)
        promo = int(rng.random() < 0.15)

        base = 110
        base *= {0: 0.85, 1: 0.85, 2: 0.9, 3: 0.95, 4: 1.15, 5: 1.5, 6: 1.45}[dow]
        base *= 1.35 if holiday else 1.0
        base *= 1.18 if promo else 1.0
        base *= 1 - min(rain, 30) * 0.012
        base *= 1 - max(0, temp - 35) * 0.02
        base *= 1 + d.dayofyear / days * 0.0  # placeholder for trend
        customers = max(10, int(rng.normal(base, base * 0.08)))

        rows.append((d.date(), customers, round(temp, 1), round(rain, 1), holiday, promo))

    return pd.DataFrame(rows, columns=["date", "customers", "temperature", "rain_mm", "is_holiday", "promotion"])


if __name__ == "__main__":
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out / "restaurant_history.csv", index=False)
    print(f"Saved {len(df)} rows to {out / 'restaurant_history.csv'}")
