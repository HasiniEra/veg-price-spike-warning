import numpy as np
import pandas as pd

PRICE = ["chg1", "chg2", "chg4", "vol8", "ratio_ma4", "market_chg1", "sin", "cos"]
EXTRA = ["ratio_med26", "ratio_med52", "drawdown8", "rebound8", "mkt_chg4", "mkt_breadth", "mkt_ratio_med26"]


def weekly_weather(W: pd.DataFrame) -> pd.DataFrame:
    W = W.copy(); W["week"] = W.date.dt.to_period("W").dt.start_time
    g = W.groupby(["location", "week"]).agg(rain=("rain_mm", "sum"), rain_days=("rain_mm", lambda s: (s >= 10).sum()),
                                            rain_max=("rain_mm", "max"), temp=("temp_c", "mean"), rh=("rh_pct", "mean")).reset_index()
    full = W.groupby("week").date.nunique(); g = g[g.week.isin(full[full == 7].index)]
    cols = ["rain", "rain_days", "rain_max", "temp", "rh"]
    up = g[g.location.isin(["Nuwara_Eliya", "Bandarawela"])].groupby("week")[cols].mean().add_prefix("up_")
    dz = g[g.location.isin(["Dambulla_Matale", "Kurunegala", "Anuradhapura", "Jaffna", "Hambantota"])].groupby("week")[cols].mean().add_prefix("dz_")
    wk = up.join(dz).sort_index()
    woy = wk.index.isocalendar().week.astype(int).values; wk["woy"] = woy
    for a in ["up_rain", "dz_rain"]:
        clim = wk[wk.index < "2023-01-01"].groupby("woy")[a].mean(); wk[a + "_anom"] = wk[a] - wk.woy.map(clim)
        for k in (1, 2, 3, 4): wk[f"{a}_l{k}"] = wk[a].shift(k)
        for k in (2, 4, 8): wk[f"{a}_s{k}"] = wk[a].rolling(k).sum()
    wk["up_temp_l1"] = wk.up_temp.shift(1); wk["dz_temp_l1"] = wk.dz_temp.shift(1)
    return wk.drop(columns="woy")


def build_table(prices: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    P = prices.dropna(subset=["Price"]).copy(); P["week"] = P.Date.dt.to_period("W").dt.start_time
    w0 = P[(P.Market == "Dambulla") & (P.Type == "Wholesale")].groupby(["Item", "week"]).Price.mean().reset_index()
    rows = []
    for it, g in w0.groupby("Item"):
        p = g.set_index("week").Price.asfreq("7D")
        f = pd.DataFrame({"price": p, "ma4": p.rolling(4, min_periods=3).mean(), "chg1": p / p.shift(1) - 1, "chg2": p.shift(1) / p.shift(2) - 1,
                          "chg4": p / p.shift(4) - 1, "vol8": (p / p.shift(1) - 1).rolling(8, min_periods=5).std(),
                          "ratio_med26": p / p.rolling(26, min_periods=13).median() - 1, "ratio_med52": p / p.rolling(52, min_periods=26).median() - 1,
                          "drawdown8": p / p.rolling(8, min_periods=4).max() - 1, "rebound8": p / p.rolling(8, min_periods=4).min() - 1,
                          "next1": p.shift(-1), "next2": p.shift(-2)}); f["Item"] = it; rows.append(f)
    X = pd.concat(rows).reset_index().rename(columns={"index": "week"})
    X["ratio_ma4"] = X.price / X.ma4 - 1
    X["market_chg1"] = X.groupby("week").chg1.transform("mean"); X["mkt_chg4"] = X.groupby("week").chg4.transform("mean")
    X["mkt_breadth"] = X.groupby("week").chg1.transform(lambda s: (s < -0.1).mean()); X["mkt_ratio_med26"] = X.groupby("week").ratio_med26.transform("mean")
    woy = X.week.dt.isocalendar().week.astype(int); X["sin"] = np.sin(2 * np.pi * woy / 52); X["cos"] = np.cos(2 * np.pi * woy / 52)
    X["chg_n2"] = X.next2 / X.price - 1; X["spike2"] = (X.chg_n2 > 0.25).astype(float); X.loc[X.next2.isna(), "spike2"] = np.nan
    X = X.merge(weekly_weather(weather).reset_index(), on="week", how="left")
    return X.replace([np.inf, -np.inf], np.nan).sort_values(["Item", "week"]).reset_index(drop=True)


def weather_cols(X): return [c for c in X.columns if c.startswith(("up_", "dz_"))]
