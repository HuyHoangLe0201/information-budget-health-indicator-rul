"""Fleet-size curve for the age-conditional information lower bound on a battery data set (same design as fleet_size_curve.py).  Run from bench/: python3 fleet_size_battery.py HUST|MATR|Tongji"""
import sys, json, numpy as np
import battery_checks as bc
from sklearn.model_selection import GroupKFold
from realdata_checks import fit_ridge_hi
NAME = bc.NAME; L_all, raw, rel, ids = bc.load(); T0, STRIDE = bc.T0, bc.STRIDE; rng = np.random.default_rng(0); out = {}
def cert(idx):
    L = L_all[idx]; n = len(idx); folds = list(GroupKFold(min(5, n)).split(np.arange(n), groups=np.arange(n))); se = {'age': [], 'ridge': []}; hcs = []
    for tri, tei in folds:
        tmax = int(np.quantile(L[tri], 0.9)); hc = bc.cond_entropy_fn(L[tri]); hmap = {t: hc(t) for t in range(T0, tmax+1)}
        Zc = np.concatenate([rel[idx[i]][:, bc.CHARGE] for i in tri]); mu, sd = Zc.mean(0), Zc.std(0)+1e-9
        Xn = {i: (rel[idx[i]][:, bc.CHARGE]-mu)/sd for i in range(n)}; hr = fit_ridge_hi([Xn[i] for i in tri]); Hr = {i: hr(Xn[i]) for i in range(n)}
        sel = lambda i: np.arange(T0, min(L[i]-1, tmax)+1, STRIDE)-1
        tr = [(i, sel(i)) for i in tri]; te = [(i, sel(i)) for i in tei]
        if sum(len(r) for _, r in tr) < 40 or sum(len(r) for _, r in te) < 5: return None
        hcs += [hmap[t] for i, r in te for t in (r+1)]
        for k in se:
            F = lambda i: (np.arange(1, L[i]+1)[:, None].astype(float) if k == 'age' else np.c_[bc.feats_hi(Hr[i]), np.arange(1, L[i]+1)[:, None].astype(float)])
            m = bc.hgb().fit(np.vstack([F(i)[r] for i, r in tr]), np.concatenate([L[i]-(r+1) for i, r in tr]))
            for i, r in te: se[k] += list((m.predict(F(i)[r])-(L[i]-(r+1)))**2)
    hb = np.mean(hcs); return {k: float(hb-0.5*np.log(2*np.pi*np.e*np.mean(v))) for k, v in se.items()}
for k in [x for x in (5, 10, 20, 40, 80, 120) if x <= 0.8*len(L_all)]:
    rs = []
    for rep in range(10):
        r = cert(rng.choice(len(L_all), k, replace=False))
        if r: rs.append(r)
    if rs:
        a = np.array([r['age'] for r in rs]); h = np.array([r['ridge'] for r in rs])
        out[k] = dict(age_mean=float(a.mean()), age_max=float(a.max()), ridge_mean=float(h.mean()), gain_mean=float((h-a).mean()), pass_control=float(np.mean(a <= 0.05)), informative=float(np.mean(h-a > 0.1)), valid=float(np.mean((h-a > 0.1) & (a <= 0.05))), n=len(rs))
        print(NAME, k, {kk: round(v, 2) for kk, v in out[k].items()}, flush=True)
json.dump(out, open(f'../results/fleet_size_{NAME}.json', 'w'), indent=1)
