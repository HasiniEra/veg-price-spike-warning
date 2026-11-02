import pandas as pd
from veg_features import build_table

prices = pd.read_csv("data/sl_vegetable_prices_daily.csv", parse_dates=["Date"])
weather = pd.read_csv("data/weather_daily.csv", parse_dates=["date"])
X = build_table(prices, weather)

X.to_pickle("features.pkl")
out = X.copy(); out["week"] = out.week.dt.strftime("%Y-%m-%d"); out.to_csv("veg_weekly_model_table.csv", index=False)
print(f"{len(X)} vegetable-weeks, {X.Item.nunique()} vegetables, {X.week.min().date()} to {X.week.max().date()}")
print("rows with a known outcome:", int(X.spike2.notna().sum()), "| spike rate:", round(float(X.spike2.mean()), 3))
