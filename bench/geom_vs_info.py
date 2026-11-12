"""Do geometric scores track certified information on real data?  For each data set a family of 42 candidate HIs (40 random linear combinations of the
normalised sensors or charging features, the PCA-HI and the ridge HI) is built inside unit-grouped 5-fold cross-validation.  For every HI the
age-conditional certificate (as in realdata_checks.py / battery_checks.py) and the geometric score Mon + Tre + Pro on the held-out units are computed,
and the Spearman correlation between the two is reported.   Run from bench/:  python3 geom_vs_info.py FD001|FD002|FD003|FD004|HUST|MATR|Tongji"""
import sys, json, time, numpy as np
from scipy import stats
NAME = sys.argv[1]; K = 40
from sklearn.model_selection import GroupKFold
import realdata_checks as rd
ENG = NAME.startswith('FD')
if ENG:
    eng = rd.load(NAME); multi = NAME in ('FD002', 'FD004'); L = np.array([len(e[0]) for e in eng]); T0 = rd.T0; STRIDE = 3
    W = dict(lev=10, vel=30, l1=20, l2=50); cond = rd.cond_entropy_fn
else:
    sys.argv = [sys.argv[0], NAME]; import battery_checks as bc
    L, raw, rel, ids = bc.load(); T0 = bc.T0; STRIDE = bc.STRIDE; W = dict(lev=bc.W_LEV, vel=bc.W_VEL, l1=bc.LAG1, l2=bc.LAG2); cond = bc.cond_entropy_fn
n = len(L); folds = list(GroupKFold(5).split(np.arange(n), groups=np.arange(n))); rng = np.random.default_rng(0)
def lag(h, k): return np.r_[np.full(k, h[0]), h][:len(h)]
def feats(h): return np.c_[rd.tmean(h, W['lev']), rd.tslope(h, W['vel']), lag(h, W['l1']), lag(h, W['l2'])]
d_feat = 14 if ENG else 3; Wts = rng.standard_normal((K, d_feat)); names = [f'rand{k}' for k in range(K)]+['PCA-HI', 'ridge HI']
se = {m: [] for m in names}; hc_all = []; mons = {m: [] for m in names}; tres = {m: [] for m in names}; starts = {m: [] for m in names}; ends = {m: [] for m in names}; t0 = time.time()
for kf, (tri, tei) in enumerate(folds):
    tmax = int(np.quantile(L[tri], 0.9)); hc = cond(L[tri]); hmap = {t: hc(t) for t in range(T0, tmax+1)}
    if ENG:
        P = rd.Prep(eng, tri, multi); Xn = {i: P.norm(eng[i][0], eng[i][1]) for i in range(n)}
    else:
        Zc = np.concatenate([rel[i][:, bc.CHARGE] for i in tri]); mu, sd = Zc.mean(0), Zc.std(0)+1e-9; Xn = {i: (rel[i][:, bc.CHARGE]-mu)/sd for i in range(n)}
    HI = {}
    for k in range(K):
        w = Wts[k]; s = np.sign(np.mean([np.corrcoef(Xn[i]@w, np.arange(L[i]))[0, 1] for i in tri])) or 1.0; w = w*s
        fa = rd.affine([rd.tmean(Xn[i]@w, 5) for i in tri]); HI[f'rand{k}'] = {i: fa(rd.tmean(Xn[i]@w, 5)) for i in range(n)}
    hp = rd.fit_pca_hi([Xn[i] for i in tri]); hr = rd.fit_ridge_hi([Xn[i] for i in tri]); HI['PCA-HI'] = {i: hp(Xn[i]) for i in range(n)}; HI['ridge HI'] = {i: hr(Xn[i]) for i in range(n)}
    sel = lambda i: np.arange(T0, min(L[i]-1, tmax)+1, STRIDE)-1
    tri_sel = [(i, sel(i)) for i in tri]; tei_sel = [(i, sel(i)) for i in tei]
    hc_all += [hmap[t] for i, r in tei_sel for t in (r+1)]
    for m in names:
        F = {i: np.c_[feats(HI[m][i]), np.arange(1, L[i]+1)[:, None].astype(float)] for i in range(n)}
        Xtr = np.vstack([F[i][r] for i, r in tri_sel]); ytr = np.concatenate([L[i]-(r+1) for i, r in tri_sel]); mdl = rd.hgb().fit(Xtr, ytr)
        for i, r in tei_sel: se[m] += list((mdl.predict(F[i][r])-(L[i]-(r+1)))**2)
        for i in tei:
            h = HI[m][i]; mons[m].append(rd.mon(h)); tres[m].append(abs(np.corrcoef(h, np.arange(len(h)))[0, 1])); starts[m].append(h[:5].mean()); ends[m].append(h[-5:].mean())
    print(NAME, 'fold', kf, 'done %.0fs' % (time.time()-t0), flush=True)
hbar = float(np.mean(hc_all)); res = dict(name=NAME, n=n, h_cond=hbar, hi={})
for m in names:
    cert = hbar-0.5*np.log(2*np.pi*np.e*np.mean(se[m])); mo = float(np.mean(mons[m])); tr = float(np.mean(tres[m])); pro = float(np.exp(-np.std(ends[m])/max(np.mean(np.abs(np.array(ends[m])-np.array(starts[m]))), 1e-9)))
    res['hi'][m] = dict(cert=float(cert), mon=mo, tre=tr, pro=pro, score=mo+tr+pro, rmse=float(np.sqrt(np.mean(se[m]))))
c = np.array([res['hi'][m]['cert'] for m in names]); sc = np.array([res['hi'][m]['score'] for m in names]); mo = np.array([res['hi'][m]['mon'] for m in names])
res['spearman_score'] = float(stats.spearmanr(sc, c)[0]); res['spearman_mon'] = float(stats.spearmanr(mo, c)[0])
best = int(np.argmax(sc)); res['best_score_hi'] = names[best]; res['best_score_cert_rank'] = int((c > c[best]).sum()+1); res['best_cert'] = float(c.max()); res['cert_of_best_score'] = float(c[best]); res['median_cert'] = float(np.median(c)); res['n_hi'] = len(names)
json.dump(res, open(f'../results/geom_vs_info_{NAME}.json', 'w'), indent=1)
print(NAME, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != 'hi'}, flush=True)
