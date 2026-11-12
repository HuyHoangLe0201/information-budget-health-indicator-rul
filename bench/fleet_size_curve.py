"""How many run-to-failure units does the age-conditional certificate need?  Subsample k units from a large data set, run the five-fold certificate (age only, ridge HI), repeat 5 times, and report the mean and the share of
subsamples in which the age-only control is <= 0.05 nats and the ridge HI certificate is above the control.   Run from bench/: python3 fleet_size_curve.py FD002"""
import sys, json, numpy as np
import realdata_checks as rd
from sklearn.model_selection import GroupKFold
fd = sys.argv[1]; eng_all = rd.load(fd); multi = fd in ('FD002', 'FD004'); T0 = rd.T0; STRIDE = 3; rng = np.random.default_rng(0); out = {}
def cert(eng):
    n = len(eng); L = np.array([len(e[0]) for e in eng]); folds = list(GroupKFold(min(5, n)).split(np.arange(n), groups=np.arange(n))); se = {'age': [], 'ridge': []}; hcs = []
    for tri, tei in folds:
        P = rd.Prep(eng, tri, multi); tmax = int(np.quantile(L[tri], 0.9)); hc = rd.cond_entropy_fn(L[tri]); hmap = {t: hc(t) for t in range(T0, tmax+1)}
        Xn = {i: P.norm(eng[i][0], eng[i][1]) for i in range(n)}; hr = rd.fit_ridge_hi([Xn[i] for i in tri]); Hr = {i: hr(Xn[i]) for i in range(n)}
        sel = lambda i: np.arange(T0, min(L[i]-1, tmax)+1, STRIDE)-1
        tr = [(i, sel(i)) for i in tri]; te = [(i, sel(i)) for i in tei]
        if sum(len(r) for _, r in tr) < 40 or sum(len(r) for _, r in te) < 5: return None
        hcs += [hmap[t] for i, r in te for t in (r+1)]
        for k in se:
            F = lambda i: (np.arange(1, L[i]+1)[:, None].astype(float) if k == 'age' else np.c_[rd.feats_hi(Hr[i]), np.arange(1, L[i]+1)[:, None].astype(float)])
            m = rd.hgb().set_params(min_samples_leaf=10).fit(np.vstack([F(i)[r] for i, r in tr]), np.concatenate([L[i]-(r+1) for i, r in tr]))
            for i, r in te: se[k] += list((m.predict(F(i)[r])-(L[i]-(r+1)))**2)
    hb = np.mean(hcs); return {k: float(hb-0.5*np.log(2*np.pi*np.e*np.mean(v))) for k, v in se.items()}
for k in (5, 10, 20, 40, 80):
    rs = []
    for rep in range(10):
        idx = rng.choice(len(eng_all), k, replace=False); r = cert([eng_all[i] for i in idx])
        if r: rs.append(r)
    if rs:
        a = np.array([r['age'] for r in rs]); h = np.array([r['ridge'] for r in rs])
        out[k] = dict(age_mean=float(a.mean()), age_max=float(a.max()), ridge_mean=float(h.mean()), gain_mean=float((h-a).mean()), pass_control=float(np.mean(a <= 0.05)), informative=float(np.mean(h-a > 0.1)), valid=float(np.mean((h-a > 0.1) & (a <= 0.05))), n=len(rs))
        print(fd, k, {kk: round(v, 2) for kk, v in out[k].items()}, flush=True)
json.dump(out, open(f'../results/fleet_size_{fd}.json', 'w'), indent=1)
