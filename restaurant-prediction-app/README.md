# Restaurant Prediction App

Predicts how many people will visit your restaurant **tomorrow** using your past
daily data and a machine-learning model (Gradient Boosting).

## What the AI looks at
- Day of week, month, weekend (seasonality)
- Holidays / festivals and promotions
- Weather (temperature, rain)
- Recent demand: yesterday, same day last week, 7- and 28-day averages

## Project layout
```
restaurant-prediction-app/
├── app.py                    # Streamlit dashboard
├── generate_sample_data.py   # creates demo data (2 years)
├── requirements.txt
├── data/restaurant_history.csv
├── models/footfall_model.joblib   (created after training)
└── src/
    ├── features.py           # feature engineering
    ├── train.py              # training + accuracy check
    └── predict.py            # next-day prediction
```

## Quick start
```bash
pip install -r requirements.txt
python generate_sample_data.py      # skip if you have your own data
python -m src.train                 # trains and prints accuracy
python -m src.predict               # prints tomorrow's forecast
streamlit run app.py                # opens the dashboard
```

## Using your own data
Make a CSV at `data/restaurant_history.csv` (or upload it in the app) with one row per day:

| date | customers | temperature | rain_mm | is_holiday | promotion |
|------|-----------|-------------|---------|------------|-----------|
| 2026-09-01 | 142 | 31.5 | 0 | 0 | 1 |

Use at least 4–6 months of data; 1–2 years is better. Retrain whenever you add new days.

## How accuracy is measured
The last 60 days are held out and predicted using only earlier data. The app
shows the average error and compares it with a simple "same as last week" guess.

## Ideas for next steps
- Add real weather-forecast API for tomorrow's temperature/rain
- Add local events (cricket matches, festivals) as a feature
- Predict per meal (lunch/dinner) or per dish for inventory planning
