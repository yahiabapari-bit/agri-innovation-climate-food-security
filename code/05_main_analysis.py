"""Step 5. Main results (Tables 2-7 and 9, Figures 3-7 inputs, Section 4.2 significance region and economic magnitude).
Input : data/panel_v4.csv
Output: results/main_results.json, results/descriptives.csv
"""
import os, json, numpy as np, pandas as pd
from econ import fit, get, lincom, wild_boot, cd_test, load_panel, design, COMP
os.makedirs("../results", exist_ok=True)
P = load_panel(); S = P[P.year.between(1991, 2022)].copy(); R = {}
D, XS = design(S)
CS = D.dropna(subset=["y_outall"] + XS).copy()          # common estimation sample (93 countries)
R["sample"] = (len(CS), int(CS.iso3.nunique()))
pack = lambda o, vs: {v: get(o, v) for v in vs if v in o["xs"]} | {"_n": (o["n"], int(o["G"]))}
# Table 2 descriptives
lo, hi = S.y_cidr.quantile([.01, .99]); S["y_cidr_w"] = S.y_cidr.clip(lo, hi); D["y_cidr_w"] = S["y_cidr_w"]
desc = ["y_outall", "y_tfp", "y_input", "y_land", "y_des", "y_cidr_w", "hd30_gs", "rx5day_gs", "cdd_gs", "hd30_gs_z", "rx5day_gs_z", "K", "L",
        "gdppc_mean_8190", "irrshare_mean_8190", "agshare_mean_8190", "tas_gs_clim_8110"]
S[S.iso3.isin(CS.iso3.unique())][desc].describe().T.to_csv("../results/descriptives.csv")
# Table 3 correlations
R["corr"] = CS.drop_duplicates("iso3")[["K", "lgdp_z", "irr_z", "ag_z", "tclim_z", "L"]].corr().round(2).to_dict()
# Table 4
base_x = ["Hs", "Ws", "cdd_gs_z", "pr_gs_z"]; core = base_x + ["Hs_K", "Ws_K"]
for nm, xs in [("c1", base_x), ("c2", core), ("c3", XS)]:
    o = fit(CS, "y_outall", xs, dk=(nm == "c3"), conley=(nm == "c3")); R["t4_" + nm] = pack(o, xs); R["t4_" + nm]["_r2w"] = float(o["r2w"])
    if nm == "c3":
        R["t4_alt_se"] = {v: (float(o["se_dk"][o["xs"].index(v)]), float(o["se_conley1000"][o["xs"].index(v)]), float(o["se_conley2000"][o["xs"].index(v)])) for v in ["Hs", "Ws", "Hs_K", "Ws_K"]}
        R["t4_wild"] = {"Hs_K": wild_boot(o, "Hs_K"), "Ws_K": wild_boot(o, "Ws_K")}
        R["cd_test"] = cd_test(o)
        R["fig4_me"] = {f"{s}_K{kv}": lincom(o, {s: 1, f"{s}_K": kv}) for s in ("Hs", "Ws") for kv in (-1, 0, 1)}
        R["fig4_cov"] = {s: (float(o["V"][o["xs"].index(s), o["xs"].index(s)]), float(o["V"][o["xs"].index(s + "_K"), o["xs"].index(s + "_K")]), float(o["V"][o["xs"].index(s), o["xs"].index(s + "_K")])) for s in ("Hs", "Ws")}
# Table 5 channels
CS["y_diff"] = CS.y_tfp - CS.y_input
for y in ["y_tfp", "y_input", "y_diff", "y_land", "y_labor", "y_capital", "y_materials"]:
    R["t5_" + y] = pack(fit(CS, y, XS), ["Hs", "Ws", "Hs_K", "Ws_K"])
# Table 6 food security
for s_ in ["Hs", "Ws"]:
    D[s_ + "_L"] = D[s_] * D.L; D[s_ + "_K_L"] = D[s_] * D.K * D.L
XS3 = XS + ["Hs_L", "Ws_L", "Hs_K_L", "Ws_K_L"]
for y in ["y_des", "y_cidr_w"]:
    R[f"t6_{y}_a"] = pack(fit(D, y, XS), ["Hs", "Ws", "Hs_K", "Ws_K"])
    R[f"t6_{y}_b"] = pack(fit(D, y, XS3), ["Hs", "Ws", "Hs_K", "Ws_K", "Hs_L", "Ws_L", "Hs_K_L", "Ws_K_L"])
R["t6_y_cer"] = pack(fit(D, "y_cer", XS), ["Hs", "Ws", "Hs_K", "Ws_K"])
# Figures 5-6 local projections
P2 = P.copy()
P2["ln_out"] = np.log(P2.usda_outall_index); P2["ln_tfp"] = np.log(P2.usda_tfp_index); P2["ln_des"] = np.log(P2.des_kcal_fbs_spliced); P2["ln_cer"] = np.log(P2.cereals_prod_t.replace(0, np.nan))
P2, _ = design(P2)
for v in ["Hs", "Ws"]:
    for l in (1, 2): P2[f"{v}_l{l}"] = P2.groupby("iso3")[v].shift(l)
LP = {}
for yv in ["ln_out", "ln_tfp", "ln_cer", "ln_des"]:
    P2["d_" + yv] = P2.groupby("iso3")[yv].diff()
    for l in (1, 2): P2[f"d_{yv}_l{l}"] = P2.groupby("iso3")["d_" + yv].shift(l)
    for h in range(6):
        P2["lp"] = (P2.groupby("iso3")[yv].shift(-h) - P2.groupby("iso3")[yv].shift(1)) * 100
        xs = XS + [f"d_{yv}_l1", f"d_{yv}_l2", "Hs_l1", "Hs_l2", "Ws_l1", "Ws_l2"]
        o = fit(P2[P2.year.between(1991, 2022)], "lp", xs)
        for kv in (-1, 0, 1):
            for s in ("Hs", "Ws"): LP[f"{yv}|{s}|K{kv}|h{h}"] = lincom(o, {s: 1, s + "_K": kv})
R["lp"] = LP
# Table 7 robustness
def rob(nm, df, y="y_outall", **kw):
    Dd, X = design(df, **kw)
    R["t7_" + nm] = pack(fit(Dd, y, X), ["Hs", "Hs_K", "Ws", "Ws_K"])
rob("hd35", S, H="hd35_gs_z"); rob("txx", S, H="txx_gs_z"); rob("rx1day", S, W="rx1day_gs_z"); rob("r20mm", S, W="r20mm_gs_z"); rob("flood", S, W="flood_z")
rob("K_pim", S, Kvars=("K_pim",)); rob("K_lag10", S, Kvars=("K_lag10",)); rob("K_land", S, Kvars=("K_land",)); rob("K_worker", S, Kvars=("K_worker",))
rob("drop_big3", S[~S.iso3.isin(["BRA", "CHN", "IND"])])
Sw = S.copy(); lo, hi = Sw.y_outall.quantile([.01, .99]); Sw["y_outall"] = Sw.y_outall.clip(lo, hi); rob("winsor", Sw)
Dt = CS.copy(); tr = []
for c in Dt.iso3.unique(): Dt["tr_" + c] = np.where(Dt.iso3 == c, Dt.year - 2006, 0.0); tr.append("tr_" + c)
R["t7_trends"] = pack(fit(Dt, "y_outall", XS + tr), ["Hs", "Hs_K", "Ws", "Ws_K"])
rob("detrended", S, H="H_dt", W="W_dt")
for lead_src, tag in [("hd30_gs_z", "placebo"), ("H_dt", "placebo_detrended")]:
    Pp = P.copy(); gg = Pp.groupby("iso3")
    for l in (1, 2): Pp[f"Hl{l}"] = gg[lead_src].shift(-l); Pp[f"Hl{l}_K"] = Pp[f"Hl{l}"] * Pp.K
    Dp, X = design(Pp[Pp.year.between(1991, 2022)], H=lead_src, W="rx5day_gs_z" if tag == "placebo" else "W_dt")
    R["t7_" + tag] = pack(fit(Dp, "y_outall", X + ["Hl1", "Hl2", "Hl1_K", "Hl2_K"]), ["Hs", "Hs_K", "Ws", "Ws_K", "Hl1", "Hl2", "Hl1_K", "Hl2_K"])
Dsz, Xsz = design(S, mods=tuple(COMP) + ("size_z", "area_z")); o = fit(Dsz, "y_outall", Xsz)
R["t7_size_area"] = pack(o, ["Hs", "Hs_K", "Ws", "Ws_K"]); R["t7_size_area_wild_HK"] = wild_boot(o, "Hs_K")
# K terciles
kt = pd.qcut(CS.drop_duplicates("iso3").set_index("iso3").K, 3, labels=[1, 2, 3]).astype(float)
Dk = CS.copy(); Dk["Kt"] = Dk.iso3.map(kt)
for s in ("Hs", "Ws"):
    for t in (2, 3): Dk[f"{s}_T{t}"] = Dk[s] * (Dk.Kt == t)
xs = base_x + ["Hs_T2", "Hs_T3", "Ws_T2", "Ws_T3"] + [f"{s}_{z}" for s in ("Hs", "Ws") for z in COMP]
R["terciles"] = pack(fit(Dk, "y_outall", xs), ["Hs", "Hs_T2", "Hs_T3", "Ws", "Ws_T2", "Ws_T3"])
# Figure 7 heterogeneity
SA = ["BGD", "IND", "PAK", "NPL", "LKA", "BTN", "MDV", "AFG"]
Dh = CS.copy(); Dh["g_SA"] = Dh.iso3.isin(SA).astype(float); Dh["g_SSA"] = Dh.region.str.startswith("Sub-Saharan").astype(float)
Dh["g_LIC"] = (Dh.income_class_1990 == "L").astype(float); Dh["g_rice"] = (Dh.gs_crop == "Rice").astype(float); Dh["g_wheat"] = (Dh.gs_crop == "Wheat").astype(float)
HET = {}
for grp in [["g_SA", "g_SSA"], ["g_LIC"], ["g_rice", "g_wheat"]]:
    xs = list(XS)
    for gg_ in grp:
        for s in ("Hs", "Ws"): Dh[f"{s}_{gg_}"] = Dh[s] * Dh[gg_]; xs.append(f"{s}_{gg_}")
    o = fit(Dh, "y_outall", xs)
    for s in ("Hs", "Ws"):
        HET[f"base_{grp[0]}|{s}"] = lincom(o, {s: 1})
        for gg_ in grp: HET[f"{gg_}|{s}"] = lincom(o, {s: 1, f"{s}_{gg_}": 1})
R["heterogeneity"] = HET
# Section 4.2: significance region of the marginal effects (Figure 4) and economic magnitude
o = fit(CS, "y_outall", XS)
K_c = CS.drop_duplicates("iso3").set_index("iso3").K
grid = np.linspace(-3.5, 2.3, 5801); SIG = {}
for s in ("Hs", "Ws"):
    i, j = o["xs"].index(s), o["xs"].index(s + "_K")
    me = o["b"][i] + o["b"][j] * grid
    se = np.sqrt(o["V"][i, i] + grid ** 2 * o["V"][j, j] + 2 * grid * o["V"][i, j])
    sig = grid[(me + 1.96 * se) < 0]
    thr = float(sig.min()) if len(sig) else None
    SIG[s] = {"threshold_K_5pct": thr, "countries_above": int((K_c > thr).sum()) if thr is not None else 0, "n_countries": int(len(K_c))}
R["fig4_significance_region"] = SIG
R["fig4_K_by_country"] = K_c.round(3).to_dict()
hbar = float(CS[CS.year.between(2015, 2022)].groupby("year").Hs.mean().mean())
R["economic_magnitude"] = {"mean_heat_anomaly_2015_2022": hbar,
                           "implied_growth_loss_at_mean_K": lincom(o, {"Hs": hbar}),
                           "implied_growth_loss_at_K_plus1": lincom(o, {"Hs": hbar, "Hs_K": hbar}),
                           "mean_output_growth": float(CS.y_outall.mean())}
print("Figure 4 significance region:", SIG)
print("Economic magnitude:", R["economic_magnitude"])
json.dump(R, open("../results/main_results.json", "w"), indent=1, default=float)
print("Main coefficients (Table 4, col. 3):", R["t4_c3"]["Hs"], R["t4_c3"]["Hs_K"], R["t4_c3"]["Ws"], R["t4_c3"]["Ws_K"])
