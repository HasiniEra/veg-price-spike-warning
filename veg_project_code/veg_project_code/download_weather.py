import sys
import time

import pandas as pd
import requests

START, END = "20160701", "20260331"

LOCATIONS = [
    ("Nuwara_Eliya", 6.97, 80.77),
    ("Bandarawela", 6.83, 80.99),
    ("Dambulla_Matale", 7.86, 80.65),
    ("Kurunegala", 7.49, 80.37),
    ("Anuradhapura", 8.31, 80.40),
    ("Jaffna", 9.66, 80.01),
    ("Hambantota", 6.12, 81.12),
]
PARAMS = "PRECTOTCORR,T2M,RH2M"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def parse_power_json(data: dict, name: str) -> pd.DataFrame:
    p = data["properties"]["parameter"]
    df = pd.DataFrame({"rain_mm": p["PRECTOTCORR"], "temp_c": p["T2M"], "rh_pct": p["RH2M"]})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999, float("nan")).rename_axis("date").reset_index()
    df.insert(1, "location", name)
    return df


def fetch(name, lat, lon, tries=4):
    q = dict(start=START, end=END, latitude=lat, longitude=lon, community="ag", parameters=PARAMS, format="json")
    for k in range(1, tries + 1):
        try:
            r = requests.get(URL, params=q, timeout=120)
            r.raise_for_status()
            return parse_power_json(r.json(), name)
        except Exception as e:
            print(f"   attempt {k}/{tries} failed: {e}")
            time.sleep(5 * k)
    return None


def main():
    frames = []
    for name, lat, lon in LOCATIONS:
        print(f"Downloading {name} ({lat}, {lon}) ...")
        df = fetch(name, lat, lon)
        if df is None:
            print(f"   could not download {name}, skipping.")
            continue
        print(f"   ok: {len(df)} days, rain mean {df.rain_mm.mean():.1f} mm/day")
        frames.append(df)
    if not frames:
        print("Nothing downloaded. Check your internet connection.")
        sys.exit(1)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv("weather_daily.csv", index=False)
    print(f"\nSaved weather_daily.csv with {len(out)} rows for {out.location.nunique()} locations.")


if __name__ == "__main__":
    main()
