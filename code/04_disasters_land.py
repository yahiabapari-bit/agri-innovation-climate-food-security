"""Step 4. EM-DAT flood/drought impacts (via Our World in Data), population and land abundance.
Input : data/panel_v3.csv
Output: data/panel_v4.csv  (analysis panel)
"""
import io, requests, numpy as np, pandas as pd
H = {"User-Agent": "Mozilla/5.0"}
P = pd.read_csv("../data/panel_v3.csv", low_memory=False)
g = lambda slug: pd.read_csv(io.StringIO(requests.get(f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true", headers=H, timeout=120).text))
T, Dd = g("total-affected-by-natural-disasters"), g("number-of-deaths-from-natural-disasters")
T, Dd = T[T.code.notna() & (T.code.str.len() == 3)], Dd[Dd.code.notna() & (Dd.code.str.len() == 3)]
E = T[["code", "year", "total_affected_flood_yearly", "total_affected_drought_yearly", "total_affected_extreme_temperature_yearly"]].merge(
    Dd[["code", "year", "total_dead_flood_yearly", "total_dead_drought_yearly"]], on=["code", "year"], how="outer").rename(columns={"code": "iso3",
    "total_affected_flood_yearly": "emdat_flood_affected", "total_affected_drought_yearly": "emdat_drought_affected", "total_affected_extreme_temperature_yearly": "emdat_heat_cold_affected",
    "total_dead_flood_yearly": "emdat_flood_deaths", "total_dead_drought_yearly": "emdat_drought_deaths"})
grid = pd.MultiIndex.from_product([P.iso3.unique(), range(1980, 2025)], names=["iso3", "year"]).to_frame(index=False)
E = grid.merge(E, on=["iso3", "year"], how="left"); ec = [c for c in E.columns if c.startswith("emdat_")]; E[ec] = E[ec].fillna(0)
pop = requests.get("https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000&date=1960:2025", headers=H).json()[1]
pop = pd.DataFrame([(d["countryiso3code"], int(d["date"]), d["value"]) for d in pop], columns=["iso3", "year", "population"])
P = P.merge(E, on=["iso3", "year"], how="left").merge(pop, on=["iso3", "year"], how="left")
P["emdat_flood_affected_per1000"] = P.emdat_flood_affected / P.population * 1000
P["emdat_flood_event_year"] = ((P.emdat_flood_affected > 0) | (P.emdat_flood_deaths > 0)).astype(float).where(P.year.between(1980, 2024))
P["cropland_per_worker"] = P.usda_cropland_q / P.usda_labor_q
P["cropland_per_worker_8190"] = P.iso3.map(P[P.year.between(1981, 1990)].groupby("iso3").cropland_per_worker.mean())
x = np.log(P.drop_duplicates("iso3").set_index("iso3").cropland_per_worker_8190)
P["land_abundance_z_8190"] = P.iso3.map((x - x.mean()) / x.std())
P.to_csv("../data/panel_v4.csv", index=False); print("Saved ../data/panel_v4.csv", P.shape)
