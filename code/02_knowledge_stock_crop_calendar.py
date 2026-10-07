"""Step 2. Knowledge stocks (Eq. 1) and dominant-cereal growing seasons (Sacks et al., 2010).
Input : data/panel_base.csv
Output: data/panel_v2.csv, data/crop_calendar_country.csv
"""
import numpy as np, pandas as pd, requests, logging
import country_converter as coco
logging.getLogger("country_converter").setLevel(logging.ERROR)
H = {"User-Agent": "Mozilla/5.0"}
P = pd.read_csv("../data/panel_base.csv", low_memory=False)

# ---- knowledge stock: trapezoid lags 0-19, weights normalised to 1 ----
R = P.pivot_table(index="year", columns="iso3", values="rd_pub_ag_mn_ppp2017").reindex(range(1960, 2026))
Ri = R.interpolate(limit_area="inside")
w = np.array([0] + [l / 5 for l in range(1, 5)] + [1.0] * 5 + [1 - (l - 9) / 10 for l in range(10, 20)]); w = w / w.sum(); L = len(w)
Kt = pd.DataFrame(index=Ri.index, columns=Ri.columns, dtype=float)
for t in Ri.index:
    win = [t - l for l in range(L)]
    if win[-1] < 1960: continue
    vals = Ri.loc[win].values; ok = ~np.isnan(vals).any(axis=0)
    Kt.loc[t] = np.where(ok, (w[:, None] * vals).sum(0), np.nan)
# perpetual inventory: K_t = (1-0.10) K_t-1 + R_t-2 ; K0 = R/(0.02+0.10)
d, gr = 0.10, 0.02
Kp = pd.DataFrame(index=Ri.index, columns=Ri.columns, dtype=float)
for c in Ri.columns:
    s, k = Ri[c], np.nan
    for t in Ri.index:
        r2 = s.get(t - 2, np.nan)
        if np.isnan(k):
            if not np.isnan(r2): k = r2 / (gr + d)
        else:
            k = (1 - d) * k + (r2 if not np.isnan(r2) else np.nan)
        Kp.loc[t, c] = k
kk = Kt.stack().rename("k_stock_trap").to_frame().join(Kp.stack().rename("k_stock_pim"), how="outer").reset_index()
P = P.merge(kk, on=["year", "iso3"], how="left")
b = P[P.year.between(1981, 1990)].groupby("iso3").agg(k_trap_mean_8190=("k_stock_trap", "mean"), k_pim_mean_8190=("k_stock_pim", "mean"), k_trap_n=("k_stock_trap", "count"))
b.loc[b.k_trap_n < 10, "k_trap_mean_8190"] = np.nan
P = P.merge(b.reset_index(), on="iso3", how="left")
P["k_intensity_trap_8190"] = P.k_trap_mean_8190 * 1000 / P.gpv_mean_8190
P["k_intensity_pim_8190"] = P.k_pim_mean_8190 * 1000 / P.gpv_mean_8190
P["k_stock_trap_lag10"] = P.groupby("iso3").k_stock_trap.shift(10)
dev = P[P.developing_1990 == True].drop_duplicates("iso3")
for v in ["k_intensity_trap_8190", "k_intensity_pim_8190"]:
    x = np.log(dev.set_index("iso3")[v].replace(0, np.nan)); z = (x - x.mean()) / x.std()
    P[v.replace("k_intensity", "K_z")] = P.iso3.map(z)

# ---- crop calendar ----
base = "https://sage-public-files.s3.amazonaws.com/crop-calendar-dataset/"
open("raw/sacks_all_data.csv", "wb").write(requests.get(base + "All_data_with_climate.csv", headers=H, timeout=180).content)
s = pd.read_csv("raw/sacks_all_data.csv", encoding="latin-1", low_memory=False)
cer = s[s.Crop.isin(["Rice", "Maize", "Wheat"])]
manual = {2: "CAN", 3: "RUS", 5: "USA", 75: "IND", 167: "MYS", 173: "COD", 174: "BRA", 177: "UGA", 187: "TZA", 202: "AUS"}
names = cer[cer.Level == "N"].groupby("Nation.code").Location.first()
iso = dict(zip(names.index, coco.convert(list(names.values), to="ISO3", not_found=None))); iso.update(manual)
cc = cer.copy(); cc["iso3"] = cc["Nation.code"].map(iso); cc = cc[cc.iso3.notna()]
cc["rank"] = cc.Level.map({"N": 0, "S": 1, "C": 2})
cc = cc.sort_values(["iso3", "Crop", "rank", "harvested.area"], ascending=[True, True, True, False])
rep = cc.groupby(["iso3", "Crop"]).first().reset_index()
doy2m = lambda d: int(min(12, max(1, (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(round(d)) - 1)).month)))
rep["plant_month"] = rep["Plant.median"].map(doy2m); rep["harvest_month"] = rep["Harvest.median"].map(doy2m)
def months(p, h):
    m, x = [], p
    while True:
        m.append(x)
        if x == h: break
        x = x % 12 + 1
    return m
rep["season_months"] = [",".join(map(str, months(p, h))) for p, h in zip(rep.plant_month, rep.harvest_month)]
rep["crosses_year"] = rep.harvest_month < rep.plant_month
cal = rep[["iso3", "Crop", "Qualifier", "Level", "Location", "plant_month", "harvest_month", "season_months", "crosses_year"]].rename(columns={"Crop": "crop"})
cal.to_csv("../data/crop_calendar_country.csv", index=False)
ar = P[P.year.between(1981, 1990)].groupby("iso3")[["rice_area_ha", "maize_area_ha", "wheat_area_ha"]].mean(); ar.columns = ["Rice", "Maize", "Wheat"]
rows = []
for c, r_ in ar.iterrows():
    avail = cal[cal.iso3 == c]
    pick = next((cr for cr in r_.dropna().sort_values(ascending=False).index if cr in set(avail.crop)), None)
    if pick is None and len(avail): pick = avail.crop.iloc[0]
    if pick is None: continue
    a = avail[avail.crop == pick].iloc[0]
    rows.append((c, pick, int(a.plant_month), int(a.harvest_month), a.season_months, bool(a.crosses_year), a.Level))
gs = pd.DataFrame(rows, columns=["iso3", "gs_crop", "gs_plant_month", "gs_harvest_month", "gs_months", "gs_crosses_year", "gs_calendar_level"])
P = P.merge(gs, on="iso3", how="left")
P.to_csv("../data/panel_v2.csv", index=False); print("Saved ../data/panel_v2.csv", P.shape)
