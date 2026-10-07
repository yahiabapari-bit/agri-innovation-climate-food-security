"""Step 3. Growing-season climate shocks from the World Bank Climate Change Knowledge Portal (ERA5 0.25 deg).
Input : data/panel_v2.csv
Output: data/panel_v3.csv, data/cckp_monthly/*.csv
"""
import os, csv, gc, glob, functools, requests, numpy as np, pandas as pd
H = {"User-Agent": "Mozilla/5.0"}
os.makedirs("../data/cckp_monthly", exist_ok=True)
for v in ["tas", "tasmax", "txx", "pr", "rx1day", "rx5day", "r20mm", "r50mm", "hd30", "hd35", "cdd"]:
    fn = f"../data/cckp_monthly/{v}.csv"
    if os.path.exists(fn): continue
    u = f"https://cckpapi.worldbank.org/cckp/v1/era5-x0.25_timeseries_{v}_timeseries_monthly_1950-2023_mean_historical_era5_x0.25_mean/all_countries?_format=json"
    d = requests.get(u, headers=H, timeout=300).json()["data"]
    with open(fn, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso3", "year", "month", v])
        for iso, s in d.items():
            for k, val in s.items():
                if int(k[:4]) >= 1975: w.writerow([iso, int(k[:4]), int(k[5:7]), val])
    del d; gc.collect(); print("downloaded", v)
P = pd.read_csv("../data/panel_v2.csv", low_memory=False)
dev = P[P.developing_1990 == True]
M = functools.reduce(lambda a, b: a.merge(b, on=["iso3", "year", "month"], how="outer"), [pd.read_csv(f) for f in sorted(glob.glob("../data/cckp_monthly/*.csv"))])
M = M[M.iso3.isin(dev.iso3.unique())]
cal = dev.drop_duplicates("iso3")[["iso3", "gs_plant_month", "gs_months", "gs_crosses_year"]]
M = M.merge(cal, on="iso3", how="left")
M = M[[pd.isna(g) or str(m) in str(g).split(",") for m, g in zip(M.month, M.gs_months)]].copy()
M["season_year"] = np.where((M.gs_crosses_year == True) & (M.month >= M.gs_plant_month), M.year + 1, M.year)
M["clim_window"] = np.where(M.gs_months.isna(), "calendar_year", "growing_season")
A = M.groupby(["iso3", "season_year"]).agg(hd30_gs=("hd30", "sum"), hd35_gs=("hd35", "sum"), txx_gs=("txx", "max"), tasmax_gs=("tasmax", "mean"), tas_gs=("tas", "mean"),
    pr_gs=("pr", "sum"), rx1day_gs=("rx1day", "max"), rx5day_gs=("rx5day", "max"), r20mm_gs=("r20mm", "sum"), r50mm_gs=("r50mm", "sum"), cdd_gs=("cdd", "max"),
    clim_n_months=("month", "count"), clim_window=("clim_window", "first")).reset_index().rename(columns={"season_year": "year"})
A = A[A.year.between(1976, 2023)]
base = A[A.year.between(1981, 2010)].groupby("iso3")
for v in ["hd30_gs", "hd35_gs", "txx_gs", "rx5day_gs", "rx1day_gs", "r20mm_gs", "pr_gs", "cdd_gs", "tas_gs"]:
    mu, sd = A.iso3.map(base[v].mean()), A.iso3.map(base[v].std())
    A[v + "_z"] = np.where(sd > 0, (A[v] - mu) / sd, np.where(sd == 0, 0.0, np.nan))
A["tas_gs_clim_8110"] = A.iso3.map(base["tas_gs"].mean())
P3 = dev.merge(A, on=["iso3", "year"], how="left")
P3.to_csv("../data/panel_v3.csv", index=False); print("Saved ../data/panel_v3.csv", P3.shape)
