"""Step 1. Download GRAPE, USDA ERS, FAOSTAT and World Bank data and build the base country-year panel.
Output: data/panel_base.csv
"""

import os, io, csv, re, zipfile, requests
import pandas as pd, numpy as np, pycountry

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
os.makedirs("raw", exist_ok=True)  # raw downloads are kept in code/raw/

def download(url, path):
    with requests.get(url, headers=H, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print("downloaded", path, os.path.getsize(path))

FAO = "https://bulks-faostat.fao.org/production/"
SOURCES = {
    "raw/FBS.zip":  FAO + "FoodBalanceSheets_E_All_Data_(Normalized).zip",
    "raw/FBSH.zip": FAO + "FoodBalanceSheetsHistoric_E_All_Data_(Normalized).zip",
    "raw/FS.zip":   FAO + "Food_Security_Data_E_All_Data_(Normalized).zip",
    "raw/QCL.zip":  FAO + "Production_Crops_Livestock_E_All_Data_(Normalized).zip",
    "raw/QV.zip":   FAO + "Value_of_Production_E_All_Data_(Normalized).zip",
    "raw/RL.zip":   FAO + "Inputs_LandUse_E_All_Data_(Normalized).zip",
    "raw/usda_iap.csv": "https://ers.usda.gov/media/5403/machine-readable-and-long-format-file-of-tfp-indices-and-components-for-countries-regions-countries-grouped-by-income-level-and-the-world-1961-2023.csv?v=84368",
    "raw/grape.xlsx": "https://zenodo.org/api/records/15507361/files/grape_v1.0.0.xlsx/content",
    "raw/OGHIST.xlsx": "https://datacatalogfiles.worldbank.org/ddh-published/0037712/DR0090754/OGHIST.xlsx",
}
for path, url in SOURCES.items():
    if not os.path.exists(path):
        download(url, path)

# Keep only the FAOSTAT rows we need (streams the large zips, low memory)
def fao_filter(zpath, out, item_re, elem_re):
    zf = zipfile.ZipFile(zpath)
    name = [f for f in zf.namelist() if f.endswith(".csv") and "All_Data" in f][0]
    rd = csv.reader(io.TextIOWrapper(zf.open(name), encoding="latin-1"))
    h = next(rd); ii, ei = h.index("Item"), h.index("Element")
    ir, er = re.compile(item_re), re.compile(elem_re)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(h)
        for row in rd:
            if ir.search(row[ii]) and er.search(row[ei]):
                w.writerow(row)
    print("filtered", out)

ELEM_FBS = r"^(Food supply \(kcal/capita/day\)|Production|Import [Qq]uantity|Export [Qq]uantity)$"
fao_filter("raw/FBS.zip",  "raw/fao_fbs_new.csv", r"^(Grand Total|Cereals - Excluding Beer)$", ELEM_FBS)
fao_filter("raw/FBSH.zip", "raw/fao_fbs_old.csv", r"^(Grand Total|Cereals - Excluding Beer)$", ELEM_FBS)
fao_filter("raw/FS.zip",   "raw/fao_fs.csv", r"(?i)undernourishment|cereal import dependency|dietary energy supply", r".")
fao_filter("raw/QCL.zip",  "raw/fao_crops.csv", r"^(Cereals, primary|Rice|Maize \(corn\)|Wheat)$", r"^(Production|Area harvested)$")
fao_filter("raw/RL.zip",   "raw/fao_land.csv", r"^(Cropland|Land area equipped for irrigation|Agricultural land|Arable land)$", r"^Area$")
fao_filter("raw/QV.zip",   "raw/fao_qv.csv", r"^Agriculture$", r"(?i)gross production value")

# World Bank: WDI, WGI and country metadata through the API
def wb_indicator(code, source=None):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000&date=1960:2025"
    if source: url += f"&source={source}"
    js = requests.get(url, headers=H, timeout=120).json()
    data = js[1]
    for p in range(2, js[0]["pages"] + 1):
        data += requests.get(url + f"&page={p}", headers=H, timeout=120).json()[1]
    return pd.DataFrame([(d["countryiso3code"], int(d["date"]), d["value"]) for d in data], columns=["iso3", "year", "value"]).assign(indicator=code)

wb = pd.concat([wb_indicator("NY.GDP.PCAP.KD"), wb_indicator("NV.AGR.TOTL.ZS"),
                wb_indicator("NV.AGR.TOTL.KD"), wb_indicator("GOV_WGI_GE.EST", 3)])
cty = requests.get("https://api.worldbank.org/v2/country?format=json&per_page=400", headers=H).json()[1]
wbm = pd.DataFrame([(c["id"], c["name"], c["region"]["value"], c["incomeLevel"]["value"]) for c in cty],
                   columns=["iso3", "name", "region", "income_level_current"])
wbm = wbm[wbm.region != "Aggregates"]
print("World Bank done")

def m49iso(s):
    s = str(s).lstrip("'")
    try:
        c = pycountry.countries.get(numeric=s.zfill(3)); return c.alpha_3 if c else None
    except Exception:
        return None

def fao(fn):
    d = pd.read_csv(fn, low_memory=False)
    d["iso3"] = d["Area Code (M49)"].map(m49iso)
    d = d[d.iso3.notna()].copy()
    d["year"] = d["Year"].astype(str).str.split("-").apply(lambda p: (int(p[0]) + int(p[-1])) // 2)  # 3-yr averages -> centre year
    d["below"] = d["Value"].astype(str).str.startswith("<").astype(int)
    d["Value"] = pd.to_numeric(d["Value"].astype(str).str.replace("<", ""), errors="coerce")
    return d

out = {}
# GRAPE
g = pd.read_excel("raw/grape.xlsx", sheet_name="grape")
gp = g.pivot_table(index=["iso3c", "year"], columns="variable", values="value").rename(columns={"RD": "rd_pub_ag_mn_ppp2017", "HR": "researchers_fte"})
gf = g[g.variable == "RD"].set_index(["iso3c", "year"])[["pre_processing", "linking"]].rename(columns={"pre_processing": "rd_preprocessing_code", "linking": "rd_linking_code"})
gp = gp.join(gf); gp.index.names = ["iso3", "year"]; out["grape"] = gp
# USDA
u = pd.read_csv("raw/usda_iap.csv", low_memory=False, encoding="utf-8-sig")
meta = u[["ISO3", "Region", "Sub-Region", "Inc I"]].drop_duplicates("ISO3").rename(columns={"ISO3": "iso3", "Region": "usda_region", "Sub-Region": "usda_subregion", "Inc I": "usda_income"})
up = u.pivot_table(index=["ISO3", "Year"], columns="Variable", values="Value")
up.columns = ["usda_" + c.lower() for c in up.columns]; up.index.names = ["iso3", "year"]; out["usda"] = up
# FAOSTAT food security suite
names = {"Prevalence of undernourishment (percent) (3-year average)": "pou_3yr_pct",
         "Dietary energy supply used in the estimation of the prevalence of undernourishment (kcal/cap/day)": "des_kcal_fs_annual",
         "Cereal import dependency ratio (percent) (3-year average)": "cidr_3yr_pct",
         "Average dietary energy supply adequacy (percent) (3-year average)": "desa_3yr_pct"}
fs = fao("raw/fao_fs.csv"); fs = fs[fs.Item.isin(names)].copy(); fs["v"] = fs.Item.map(names)
F = fs.pivot_table(index=["iso3", "year"], columns="v", values="Value")
F["pou_3yr_below_2_5_flag"] = fs[fs.v == "pou_3yr_pct"].set_index(["iso3", "year"])["below"]
out["fs"] = F
# Food balances
def fb(fn, suf):
    d = fao(fn); d["key"] = d.Item.str[:6] + "|" + d.Element.str.lower()
    m = {"Grand |food supply (kcal/capita/day)": "des_kcal_fbs", "Cereal|production": "cer_prod_kt",
         "Cereal|import quantity": "cer_imp_kt", "Cereal|export quantity": "cer_exp_kt"}
    d = d[d.key.isin(m)].copy(); d["v"] = d.key.map(m) + suf
    return d.pivot_table(index=["iso3", "year"], columns="v", values="Value")
out["fbsn"] = fb("raw/fao_fbs_new.csv", "_new"); out["fbso"] = fb("raw/fao_fbs_old.csv", "_old")
# Crops, land, production value
c = fao("raw/fao_crops.csv")
c["v"] = c.Item.map({"Cereals, primary": "cereals", "Rice": "rice", "Maize (corn)": "maize", "Wheat": "wheat"}) + "_" + c.Element.map({"Production": "prod_t", "Area harvested": "area_ha"})
out["crops"] = c.pivot_table(index=["iso3", "year"], columns="v", values="Value")
l = fao("raw/fao_land.csv")
l["v"] = l.Item.map({"Cropland": "cropland_kha", "Land area equipped for irrigation": "irrigated_kha", "Agricultural land": "agland_kha", "Arable land": "arable_kha"})
out["land"] = l.pivot_table(index=["iso3", "year"], columns="v", values="Value")
q = fao("raw/fao_qv.csv"); q = q[q.Element == "Gross Production Value (constant 2014-2016 thousand I$)"]
out["qv"] = q.pivot_table(index=["iso3", "year"], values="Value").rename(columns={"Value": "gpv_agri_const1416_thousand_intd"})
# World Bank
wb["indicator"] = wb.indicator.map({"NY.GDP.PCAP.KD": "gdppc_const_usd", "NV.AGR.TOTL.ZS": "ag_va_pct_gdp", "NV.AGR.TOTL.KD": "ag_va_const_usd", "GOV_WGI_GE.EST": "wgi_gov_effectiveness"})
wb = wb[wb.iso3.astype(str).str.len() == 3]
out["wb"] = wb.pivot_table(index=["iso3", "year"], columns="indicator", values="value")

P = None
for k, v in out.items():
    v = v.reset_index(); v["year"] = v.year.astype(int); v = v.set_index(["iso3", "year"])
    P = v if P is None else P.join(v, how="outer")
P = P.reset_index(); P = P[(P.year >= 1960) & (P.year <= 2025)]
oh = pd.read_excel("raw/OGHIST.xlsx", sheet_name="Country Analytical History", header=None)
yrs = oh.iloc[5]; c87 = yrs[yrs == 1987].index[0]; c90 = yrs[yrs == 1990].index[0]
oh = oh.iloc[11:, [0, c87, c90]]; oh.columns = ["iso3", "income_class_1987", "income_class_1990"]; oh = oh[oh.iso3.notna()]
P = (P[P.iso3.isin(wbm.iso3)].merge(wbm, on="iso3", how="left").merge(meta, on="iso3", how="left")
     .merge(oh, on="iso3", how="left").sort_values(["iso3", "year"]).reset_index(drop=True))
# Derived variables
r = P[(P.year >= 2010) & (P.year <= 2013)].assign(rt=lambda d: d.des_kcal_fbs_new / d.des_kcal_fbs_old).groupby("iso3").rt.mean()
P["splice_ratio"] = P.iso3.map(r)
P["des_kcal_fbs_spliced"] = np.where(P.year >= 2010, P.des_kcal_fbs_new, P.des_kcal_fbs_old * P.splice_ratio)
pk = lambda a, b: np.where(P.year >= 2010, P[a], P[b])
prod, imp, exp_ = pk("cer_prod_kt_new", "cer_prod_kt_old"), pk("cer_imp_kt_new", "cer_imp_kt_old"), pk("cer_exp_kt_new", "cer_exp_kt_old")
P["cidr_annual_pct"] = (imp - exp_) / (prod + imp - exp_) * 100
P["irrigated_share_cropland"] = P.irrigated_kha / P.cropland_kha
for v in ["tfp", "outall", "input", "land", "labor", "capital", "materials"]:
    P["dln_" + v] = P.groupby("iso3")["usda_" + v + "_index"].transform(lambda s: np.log(s).diff())
b = P[(P.year >= 1981) & (P.year <= 1990)].groupby("iso3").agg(
    rd_mean_8190=("rd_pub_ag_mn_ppp2017", "mean"), gpv_mean_8190=("gpv_agri_const1416_thousand_intd", "mean"),
    gdppc_mean_8190=("gdppc_const_usd", "mean"), agshare_mean_8190=("ag_va_pct_gdp", "mean"),
    irrshare_mean_8190=("irrigated_share_cropland", "mean"))
b["rd_intensity_8190"] = b.rd_mean_8190 * 1000 / b.gpv_mean_8190
P = P.merge(b.reset_index(), on="iso3", how="left")
P["developing_1990"] = P.income_class_1990.isin(["L", "LM", "UM"])
front = ["iso3", "name", "year", "region", "income_class_1987", "income_class_1990", "developing_1990",
         "income_level_current", "usda_region", "usda_subregion", "usda_income"]
P = P[front + [c for c in P.columns if c not in front]]
D = P[P.developing_1990]
print("Developing countries:", D.iso3.nunique(), "| rows:", len(D))
print("Check (should be ~0):", (D.dln_outall - D.dln_tfp - D.dln_input).abs().max())

os.makedirs("../data", exist_ok=True)
P.to_csv("../data/panel_base.csv", index=False)
print("Saved ../data/panel_base.csv")
