"""Predict customers for the day after the last day in the history."""
import joblib
import pandas as pd

from .features import FEATURES, build_features, load_history
from .train import DATA_PATH, MODEL_PATH


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_day(history, date=None, temperature=28.0, rain_mm=0.0, is_holiday=0, promotion=0,
                model_bundle=None) -> int:
    """Forecast footfall for `date` (default: the day after the last recorded day)."""
    hist = load_history(history)
    bundle = model_bundle or load_model()
    date = pd.Timestamp(date) if date is not None else hist["date"].max() + pd.Timedelta(days=1)

    new_row = pd.DataFrame([{
        "date": date, "customers": float("nan"), "temperature": temperature,
        "rain_mm": rain_mm, "is_holiday": int(is_holiday), "promotion": int(promotion),
    }])
    # Lag features come from the history that precedes `date`.
    past = hist[hist["date"] < date]
    combined = pd.concat([past, new_row], ignore_index=True)
    row = build_features(combined).iloc[[-1]][bundle["features"]]
    if row.isna().any(axis=1).iloc[0]:
        raise ValueError("Not enough history before that date (need 28+ days).")
    return max(0, int(round(bundle["model"].predict(row)[0])))


if __name__ == "__main__":
    print("Tomorrow's predicted customers:", predict_day(DATA_PATH))
