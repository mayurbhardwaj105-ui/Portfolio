"""Feature engineering: turns raw daily history into model-ready features."""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "customers", "temperature", "rain_mm", "is_holiday", "promotion"]

FEATURES = [
    "day_of_week", "is_weekend", "month", "day_of_month",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "temperature", "rain_mm", "is_holiday", "promotion",
    "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28",
]


def load_history(path_or_df) -> pd.DataFrame:
    df = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, weather/promo and past-demand (lag) features.

    Lags are shifted so a row only uses information known the day before.
    """
    df = df.copy()
    d = df["date"]
    df["day_of_week"] = d.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = d.dt.month
    df["day_of_month"] = d.dt.day
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    s = df["customers"]
    df["lag_1"] = s.shift(1)
    df["lag_7"] = s.shift(7)
    df["lag_14"] = s.shift(14)
    df["roll_mean_7"] = s.shift(1).rolling(7).mean()
    df["roll_mean_28"] = s.shift(1).rolling(28).mean()
    return df


def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df).dropna(subset=FEATURES + ["customers"]).reset_index(drop=True)
