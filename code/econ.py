"""Estimation tools: high-dimensional fixed effects (alternating projections), clustered,
Driscoll-Kraay and Conley standard errors, linear combinations, wild cluster bootstrap and Pesaran CD test."""
import numpy as np, pandas as pd

def demean(df, cols, fes, tol=1e-9, maxit=500):
    X = df[cols].to_numpy(float).copy(); codes = [pd.factorize(df[f])[0] for f in fes]
    for _ in range(maxit):
        old = X.copy()
        for c in codes:
            n = np.bincount(c)
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(c, weights=X[:, j]) / n)[c]
        if np.max(np.abs(X - old)) < tol: break
    return X

def fit(df, y, xs, fes=("iso3", "regyear"), cluster="iso3", dk=False, conley=False):
    d = df.dropna(subset=[y] + xs).copy()
    A = demean(d, [y] + xs, list(fes)); yv, X = A[:, 0], A[:, 1:]
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ yv; e = yv - X @ b
    n, k = X.shape; s = X * e[:, None]
    g = pd.factorize(d[cluster])[0]; G = g.max() + 1
    S = np.zeros((G, k)); np.add.at(S, g, s)
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    o = dict(b=b, V=V, se=np.sqrt(np.diag(V)), xs=xs, n=n, G=G, d=d, e=e, X=X, y=yv, g=g, r2w=1 - (e @ e) / (yv @ yv))
    if dk:
        t = pd.factorize(d.year, sort=True)[0]; T = t.max() + 1; St = np.zeros((T, k)); np.add.at(St, t, s)
        m = int(np.floor(4 * (T / 100) ** (2 / 9))); Om = St.T @ St
        for l in range(1, m + 1):
            C = St[l:].T @ St[:-l]; Om += (1 - l / (m + 1)) * (C + C.T)
        o["se_dk"] = np.sqrt(np.diag(XtXi @ Om @ XtXi * (T / (T - 1))))
    if conley:
        ok = d[["lat", "lon"]].notna().all(1).to_numpy(); lat = np.radians(d.lat.to_numpy()); lon = np.radians(d.lon.to_numpy()); yr = d.year.to_numpy()
        for cut in (1000, 2000):
            Om = np.zeros((k, k))
            for yy in np.unique(yr):
                idx = np.where((yr == yy) & ok)[0]; la, lo = lat[idx], lon[idx]
                dd = 6371 * np.arccos(np.clip(np.sin(la)[:, None] * np.sin(la)[None, :] + np.cos(la)[:, None] * np.cos(la)[None, :] * np.cos(lo[:, None] - lo[None, :]), -1, 1))
                W = np.where(dd < cut, 1 - dd / cut, 0.0); si = s[idx]; Om += si.T @ W @ si
            for c in np.unique(g):          # serial correlation within country, Bartlett lag 3
                idx = np.where(g == c)[0]; yrs = yr[idx]; sc = s[idx]
                for l in range(1, 4):
                    for a_ in range(len(idx)):
                        for b_ in range(len(idx)):
                            if yrs[b_] - yrs[a_] == l:
                                C = np.outer(sc[a_], sc[b_]); Om += (1 - l / 4) * (C + C.T)
            o[f"se_conley{cut}"] = np.sqrt(np.diag(XtXi @ Om @ XtXi))
    return o

def get(o, v):
    i = o["xs"].index(v); return float(o["b"][i]), float(o["se"][i])

def lincom(o, w):
    w = np.array([w.get(x, 0) for x in o["xs"]]); return float(w @ o["b"]), float(np.sqrt(w @ o["V"] @ w))

def wild_boot(o, var, reps=999, seed=1):
    """Wild cluster bootstrap-t (Rademacher weights, null imposed)."""
    rng = np.random.default_rng(seed); X, y, g = o["X"], o["y"], o["g"]; G = g.max() + 1
    j = o["xs"].index(var); Xr = np.delete(X, j, axis=1)
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y; er = y - Xr @ br; yhat = Xr @ br
    XtXi = np.linalg.pinv(X.T @ X); n, k = X.shape; c = (G / (G - 1)) * ((n - 1) / (n - k))
    def tstat(yy):
        b = XtXi @ X.T @ yy; e = yy - X @ b; S = np.zeros((G, k)); np.add.at(S, g, X * e[:, None])
        V = XtXi @ (S.T @ S) @ XtXi * c; return b[j] / np.sqrt(V[j, j])
    t0 = tstat(y); ts = np.array([tstat(yhat + er * rng.choice([-1, 1], G)[g]) for _ in range(reps)])
    return float(t0), float(np.mean(np.abs(ts) >= abs(t0)))

def cd_test(o, min_obs=10):
    """Pesaran (2015) CD statistic for an unbalanced panel."""
    dd = o["d"].assign(e=o["e"]).pivot(index="year", columns="iso3", values="e"); N = dd.shape[1]; tot = 0.0; cnt = 0
    for i in range(N):
        for j in range(i + 1, N):
            m = dd.iloc[:, [i, j]].dropna()
            if len(m) >= min_obs:
                tot += np.sqrt(len(m)) * np.corrcoef(m.iloc[:, 0], m.iloc[:, 1])[0, 1]; cnt += 1
    return float(np.sqrt(2 / cnt) * tot), cnt

zc = lambda s: (s - s.mean()) / s.std()

def load_panel(path="../data/panel_v4.csv"):
    """Analysis variables used in all models."""
    import requests
    P = pd.read_csv(path, low_memory=False)
    cty = requests.get("https://api.worldbank.org/v2/country?format=json&per_page=400", headers={"User-Agent": "Mozilla/5.0"}).json()[1]
    geo = pd.DataFrame([(c["id"], float(c["latitude"]) if c["latitude"] else np.nan, float(c["longitude"]) if c["longitude"] else np.nan) for c in cty], columns=["iso3", "lat", "lon"])
    P = P.merge(geo, on="iso3", how="left").sort_values(["iso3", "year"])
    P["regyear"] = P.region.astype(str) + "_" + P.year.astype(str)
    g = P.groupby("iso3")
    base = P[P.year.between(1981, 1990)].groupby("iso3").agg(agl=("agland_kha", "mean"))
    fb = P[P.year.between(1981, 1995)].groupby("iso3").agg(gdp95=("gdppc_const_usd", "mean"), ag95=("ag_va_pct_gdp", "mean"), irr95=("irrigated_share_cropland", "mean"))
    c = g.agg(gdp=("gdppc_mean_8190", "max"), irr=("irrshare_mean_8190", "max"), ag=("agshare_mean_8190", "max"), tc=("tas_gs_clim_8110", "max"),
              kst=("k_trap_mean_8190", "max"), gpv=("gpv_mean_8190", "max")).join(fb).join(base)
    # baseline fallback: 1981-1995 averages where 1981-1990 is missing
    c["gdp"] = c.gdp.fillna(c.gdp95); c["ag"] = c.ag.fillna(c.ag95); c["irr"] = c.irr.fillna(c.irr95)
    Z = pd.DataFrame({"lgdp_z": zc(np.log(c.gdp)), "irr_z": zc(c.irr), "ag_z": zc(c.ag), "tclim_z": zc(c.tc),
                      "K_abs": zc(np.log(c.kst)), "size_z": zc(np.log(c.gpv)), "area_z": zc(np.log(c.agl))})
    P = P.merge(Z, left_on="iso3", right_index=True, how="left"); g = P.groupby("iso3")
    for v in ["outall", "tfp", "input", "land", "labor", "capital", "materials"]: P["y_" + v] = P["dln_" + v] * 100
    P["y_des"] = g.des_kcal_fbs_spliced.transform(lambda s: np.log(s).diff()) * 100
    P["y_cidr"] = g.cidr_annual_pct.diff()
    P["y_cer"] = g.cereals_prod_t.transform(lambda s: np.log(s.replace(0, np.nan)).diff()) * 100
    P["K"] = P.K_z_trap_8190; P["L"] = P.land_abundance_z_8190; P["K_pim"] = P.K_z_pim_8190
    kl = np.log(P.k_stock_trap_lag10 * 1000 / P.gpv_mean_8190); P["K_lag10"] = zc(kl)
    cc = P.drop_duplicates("iso3").set_index("iso3")
    lab = P[P.year.between(1981, 1990)].groupby("iso3").usda_labor_q.mean()
    P["K_land"] = P.iso3.map(zc(np.log(cc.k_trap_mean_8190 / base.agl.reindex(cc.index))))
    P["K_worker"] = P.iso3.map(zc(np.log(cc.k_trap_mean_8190 / lab.reindex(cc.index))))
    fl = np.log1p(P.emdat_flood_affected_per1000); P["flood_z"] = zc(fl)
    for raw, new in [("hd30_gs", "H_dt"), ("rx5day_gs", "W_dt")]:
        def det(s):
            yr = P.loc[s.index, "year"]; ok = s.notna()
            if ok.sum() < 10: return s * np.nan
            bb = np.polyfit(yr[ok], s[ok], 1); r = s - (bb[0] * yr + bb[1]); return r / r[ok].std()
        P[new] = g[raw].transform(det)
    return P

COMP = ["lgdp_z", "irr_z", "ag_z", "tclim_z"]

def design(P, H="hd30_gs_z", W="rx5day_gs_z", Kvars=("K",), mods=tuple(COMP)):
    P = P.copy(); P["Hs"] = P[H]; P["Ws"] = P[W]; xs = ["Hs", "Ws", "cdd_gs_z", "pr_gs_z"]
    for m in Kvars:                      # innovation-capacity interactions are always named Hs_K / Ws_K
        for s in ["Hs", "Ws"]:
            P[f"{s}_K"] = P[s] * P[m]; xs.append(f"{s}_K")
    for m in mods:
        for s in ["Hs", "Ws"]:
            P[f"{s}_{m}"] = P[s] * P[m]; xs.append(f"{s}_{m}")
    return P, xs
