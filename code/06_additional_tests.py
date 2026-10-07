"""Step 6. Additional tests reported in Section 4.7 and Table 8: research scale versus agricultural size and land area.
Input : data/panel_v4.csv
Output: results/additional_tests.json
"""
import os, json, numpy as np
from econ import fit, get, wild_boot, load_panel, design, COMP
os.makedirs("../results", exist_ok=True)
P = load_panel(); S = P[P.year.between(1991, 2022)]; R = {}
specs = {"col1_Kabs": COMP, "col2_Kabs_size": COMP + ["size_z"], "col3_Kabs_area": COMP + ["area_z"], "col4_Kabs_size_area": COMP + ["size_z", "area_z"]}
for nm, mods in specs.items():
    D, xs = design(S, Kvars=("K_abs",), mods=tuple(mods))
    D = D.dropna(subset=xs + ["y_outall", "size_z", "area_z", "K", "K_abs"])
    o = fit(D, "y_outall", xs, dk=True)
    R[nm] = {v: get(o, v) + (float(o["se_dk"][o["xs"].index(v)]),) for v in o["xs"] if v.startswith(("Hs_K", "Ws_K", "Hs_size", "Ws_size", "Hs_area", "Ws_area"))}
    R[nm]["_n"] = (o["n"], int(o["G"]))
    if nm == "col4_Kabs_size_area": R[nm]["_wild_Hs_K"] = wild_boot(o, "Hs_K"); R[nm]["_wild_Ws_K"] = wild_boot(o, "Ws_K")
for y in ["y_tfp", "y_input", "y_des"]:
    D, xs = design(S, Kvars=("K_abs",), mods=tuple(COMP + ["size_z", "area_z"])); o = fit(D, y, xs)
    R["channels_" + y] = {v: get(o, v) for v in ["Hs", "Ws", "Hs_K", "Ws_K", "Ws_size_z", "Ws_area_z"]}
for v in ["hd30_gs_z", "H_dt", "rx5day_gs_z", "W_dt"]:
    x = P[P.year.between(1981, 2022)]; x = x.assign(l=x.groupby("iso3")[v].shift(1)).dropna(subset=[v, "l"])
    R["AR1_" + v] = float(np.corrcoef(x[v], x.l)[0, 1])
R["corr_K_size"] = S.drop_duplicates("iso3")[["K", "K_abs", "size_z", "area_z"]].dropna().corr().round(2).to_dict()
json.dump(R, open("../results/additional_tests.json", "w"), indent=1, default=float)
print("W x K_abs, cols 1-4:", [R[k]["Ws_K"][:2] for k in specs])
