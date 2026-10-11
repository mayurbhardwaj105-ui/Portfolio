"""Train the footfall model.  Usage:  python -m src.train [path/to/history.csv]"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from .features import FEATURES, load_history, training_frame

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "restaurant_history.csv"
MODEL_PATH = ROOT / "models" / "footfall_model.joblib"


def train(data=DATA_PATH, test_days: int = 60, model_path: Path = MODEL_PATH) -> dict:
    df = load_history(data)
    frame = training_frame(df)
    if len(frame) < test_days + 60:
        raise ValueError("Need at least ~4 months of daily data to train.")

    # Time-based split: never test on the past, always on the most recent days.
    train_df, test_df = frame.iloc[:-test_days], frame.iloc[-test_days:]

    def make():
        return GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
        )

    model = make().fit(train_df[FEATURES], train_df["customers"])
    pred = model.predict(test_df[FEATURES])
    mae = mean_absolute_error(test_df["customers"], pred)
    mape = mean_absolute_percentage_error(test_df["customers"], pred) * 100
    baseline = mean_absolute_error(test_df["customers"], test_df["lag_7"])  # "same as last week"

    # Refit on everything for the production model.
    final = make().fit(frame[FEATURES], frame["customers"])
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES}, model_path)

    importances = dict(sorted(zip(FEATURES, final.feature_importances_), key=lambda x: -x[1]))
    return {
        "mae": float(mae), "mape": float(mape), "baseline_mae": float(baseline),
        "rows": int(len(frame)), "importances": importances,
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    m = train(src)
    print(f"Trained on {m['rows']} days")
    print(f"Test MAE : {m['mae']:.1f} customers  ({m['mape']:.1f}% error)")
    print(f"Baseline (same day last week) MAE: {m['baseline_mae']:.1f}")
    print("Top drivers:", ", ".join(f"{k} ({v:.2f})" for k, v in list(m['importances'].items())[:5]))
