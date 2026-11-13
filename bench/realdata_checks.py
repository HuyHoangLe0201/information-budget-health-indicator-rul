"""Real-data checks of the theory on C-MAPSS (FD001-FD004), engine-grouped 5-fold cross-validation on the training engines (run-to-failure).
R1  age-conditional information lower bound with a non-parametric lifetime law (enough engines), negative control = age-only predictor;
R2  template shrinkage (Theorem 2) applied to a real HI: monotonicity versus information lower bound;
R3  stage-wise optimal linear scalar HI of (level, velocity) (Theorem 4, Corollaries 1-2): static versus stage-wise weights;
R4  regime-blind versus regime-aware linear estimator (Proposition 5).
Population: pairs (engine, age t) with 30 <= t <= t_max (t_max = 0.9 quantile of the training lifetimes of the fold); T = L - t.  Pooling all ages of all engines
gives the law T | t = L - t | L > t exactly, so h(T|t) can be estimated from the lifetime law (Gaussian KDE of log L, cross-fitted over engines).
Run from bench/:  python3 realdata_checks.py [FD001 FD002 ...]"""
import numpy as np, json, sys, time
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
D = '/home/user/data/cmapss_raw/6. Turbofan Engine Degradation Simulation Data Set'
SENS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]; T0 = 30; NF = 5; RNG = np.random.default_rng(0)
def load(fd):
    a = np.loadtxt(f'{D}/train_{fd}.txt'); cols = [4+s for s in SENS]
    return [(a[a[:, 0] == e][:, cols], a[a[:, 0] == e][:, 2:5]) for e in np.unique(a[:, 0])]
# ---------------------------------------------------------------- causal helpers
def tmean(y, w):
    c = np.cumsum(np.r_[0, y]); i = np.arange(1, len(y)+1); a = np.maximum(i-w, 0); return (c[i]-c[a])/(i-a)
def tslope(y, w):
    n = len(y); i = np.arange(n); c1 = np.cumsum(np.r_[0, y]); c2 = np.cumsum(np.r_[0, i*y]); out = np.zeros(n)
    for t in range(1, n):
        a = max(0, t-w+1); m = t-a+1
        if m < 2: continue
        sy = c1[t+1]-c1[a]; sxy = c2[t+1]-c2[a]; sx = (a+t)*m/2; sxx = (t*(t+1)*(2*t+1)-(a-1)*a*(2*a-1))/6 if a > 0 else t*(t+1)*(2*t+1)/6
        den = m*sxx-sx*sx; out[t] = (m*sxy-sx*sy)/den if den > 0 else 0.0
    return out
def mon(h): d = np.diff(h); return abs(np.sum(np.sign(d)))/max(len(d), 1)
def feats_hi(h): return np.c_[tmean(h, 10), tslope(h, 30), np.r_[np.full(20, h[0]), h][:len(h)], np.r_[np.full(50, h[0]), h][:len(h)]]
def feats_x(X): return np.hstack([np.column_stack([tmean(X[:, j], 10) for j in range(X.shape[1])]), np.column_stack([tslope(X[:, j], 30) for j in range(X.shape[1])])])
def affine(Hl):
    st = np.mean([h[:5].mean() for h in Hl]); en = np.mean([h[-5:].mean() for h in Hl]); s = 1.0 if en > st else -1.0
    return lambda h: (s*h-s*st)/(s*en-s*st)
# ---------------------------------------------------------------- lifetime law and conditional entropy
def cond_entropy_fn(Ltr):
    kde = stats.gaussian_kde(np.log(Ltr)); tau = np.arange(0.5, 1600, 0.5)
    def h(t):
        l = t+tau; f = kde(np.log(l))/l; S = np.trapezoid(f, tau); p = np.maximum(f/S, 1e-300); return float(-np.trapezoid(p*np.log(p), tau))
    return h
# ---------------------------------------------------------------- per-fold preprocessing and HI constructions
class Prep:
    def __init__(s, eng, tr_idx, multi):
        s.multi = multi; Z = np.concatenate([eng[i][0] for i in tr_idx]); S = np.concatenate([eng[i][1] for i in tr_idx])
        if multi:
            s.sm, s.ss = S.mean(0), S.std(0)+1e-9; s.km = KMeans(6, n_init=10, random_state=0).fit((S-s.sm)/s.ss); r = s.km.predict((S-s.sm)/s.ss)
            s.mu = np.array([Z[r == k].mean(0) for k in range(6)]); s.sd = np.array([Z[r == k].std(0)+1e-9 for k in range(6)])
        else: s.mu, s.sd = Z.mean(0)[None], (Z.std(0)+1e-9)[None]
    def reg(s, Sset): return s.km.predict((Sset-s.sm)/s.ss) if s.multi else np.zeros(len(Sset), int)
    def norm(s, X, Sset, regime_blind=False):
        if regime_blind: return X
        r = s.reg(Sset); return (X-s.mu[r])/s.sd[r]
def fit_ridge_hi(Xl, alpha=10.0):
    Z = np.concatenate(Xl); y = np.concatenate([np.arange(1, len(x)+1)/len(x) for x in Xl]); Za = np.c_[Z, np.ones(len(Z))]; w = np.linalg.solve(Za.T@Za+alpha*np.eye(Za.shape[1]), Za.T@y)
    f = lambda x: np.c_[x, np.ones(len(x))]@w; fa = affine([tmean(f(x), 5) for x in Xl]); return lambda x: fa(tmean(f(x), 5))
def fit_pca_hi(Xl):
    Z = np.concatenate(Xl); m = Z.mean(0); _, _, Vt = np.linalg.svd(Z-m, full_matrices=False); w = Vt[0]
    s = np.sign(np.mean([np.corrcoef(x@w, np.arange(len(x)))[0, 1] for x in Xl])); w = w*s
    fa = affine([tmean(x@w, 5) for x in Xl]); return lambda x: fa(tmean(x@w, 5))
def hgb(): return HGB(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
def dom(L, tmax): return np.arange(T0, min(L-1, tmax)+1)   # ages (1-based cycle counts) of the evaluation domain
GAMMAS = [0.0, 0.5, 0.9, 0.97, 0.99, 0.995]
def run(fd):
    t_start = time.time(); eng = load(fd); n = len(eng); multi = fd in ('FD002', 'FD004'); L = np.array([len(e[0]) for e in eng])
    folds = list(GroupKFold(NF).split(np.arange(n), groups=np.arange(n)))
    pairs = dict(eng=[], t=[], T=[], hc=[]); names = ['age only', 'ridge HI', 'PCA-HI', 'ridge level only', 'ridge slope only', 'sensors'] + [f'squeeze {g}' for g in GAMMAS]
    mon_pred = []; pred = {m: [] for m in names}; mon_sq = {g: {} for g in GAMMAS}; hi_oof = {}
    for kf, (tri, tei) in enumerate(folds):
        P = Prep(eng, tri, multi); tmax = int(np.quantile(L[tri], 0.9)); hc = cond_entropy_fn(L[tri]); hmap = {t: hc(t) for t in range(T0, tmax+1)}
        Xn = {i: P.norm(eng[i][0], eng[i][1]) for i in range(n)}; hr = fit_ridge_hi([Xn[i] for i in tri]); hp = fit_pca_hi([Xn[i] for i in tri])
        Hr = {i: hr(Xn[i]) for i in range(n)}; Hp = {i: hp(Xn[i]) for i in range(n)}
        for i in tei: hi_oof[i] = Hr[i]
        # template and noise level of the squeeze (Theorem 2): m_k = mean HI at cycle k over training engines; sigma_2 gives Mon_inf ~ 0.9
        Lm = int(L.max()); M = np.full(Lm, np.nan)
        for k in range(Lm):
            v = [Hr[i][k] for i in tri if L[i] > k]; M[k] = np.mean(v) if v else np.nan
        M = np.array(M); last = np.where(~np.isnan(M))[0].max(); M[last+1:] = M[last]
        dm = float(np.mean(np.diff(M[:int(np.quantile(L[tri], 0.5))]))); s2 = dm/(np.sqrt(2)*1.645); mon_pred.append(abs(float(np.mean(2*stats.norm.cdf(np.diff(M[:int(np.median(L[tri]))])/(np.sqrt(2)*s2))-1))))
        def build(kind, i, g=None):
            tt = np.arange(1, L[i]+1)[:, None].astype(float)
            if kind == 'age only': return tt
            if kind == 'ridge HI': return np.c_[feats_hi(Hr[i]), tt]
            if kind == 'PCA-HI': return np.c_[feats_hi(Hp[i]), tt]
            if kind == 'ridge level only': return np.c_[tmean(Hr[i], 10), tt]
            if kind == 'ridge slope only': return np.c_[tslope(Hr[i], 30), tt]
            if kind == 'sensors': return np.c_[feats_x(Xn[i]), tt]
            if kind == 'squeeze': 
                r = np.random.default_rng(1000*kf+int(g*1000)+i); Hs = M[:L[i]]+(1-g)*(Hr[i]-M[:L[i]])+s2*r.standard_normal(L[i]); return np.c_[feats_hi(Hs), tt], Hs
        sel = lambda i: np.arange(T0, min(L[i]-1, tmax)+1)-1     # 0-based rows of the domain ages
        tri_sel = [(i, sel(i)) for i in tri]; tei_sel = [(i, sel(i)) for i in tei]
        for i, r in tei_sel:
            pairs['eng'] += [i]*len(r); pairs['t'] += list(r+1); pairs['T'] += list(L[i]-(r+1)); pairs['hc'] += [hmap[t] for t in r+1]
        def fit_pred(Ftr_fn, Fte_fn):
            Xtr = np.vstack([Ftr_fn(i)[r] for i, r in tri_sel]); ytr = np.concatenate([L[i]-(r+1) for i, r in tri_sel]); m = hgb().fit(Xtr, ytr)
            return np.concatenate([m.predict(Fte_fn(i)[r]) for i, r in tei_sel])
        for kind in ['age only', 'ridge HI', 'PCA-HI', 'ridge level only', 'ridge slope only', 'sensors']:
            pred[kind] += list(fit_pred(lambda i: build(kind, i), lambda i: build(kind, i)))
        for g in GAMMAS:
            cache = {i: build('squeeze', i, g) for i in range(n) if i in set(tri) | set(tei)}
            pred[f'squeeze {g}'] += list(fit_pred(lambda i: cache[i][0], lambda i: cache[i][0]))
            for i in tei: mon_sq[g][i] = mon(cache[i][1])
        print(fd, 'fold', kf, 'done %.0fs' % (time.time()-t_start), flush=True)
    pairs = {k: np.array(v) for k, v in pairs.items()}; pred = {k: np.array(v) for k, v in pred.items()}
    res = dict(fd=fd, mon_inf_pred=float(np.mean(mon_pred)), n_engines=n, lifetimes=L.tolist(), n_pairs=int(len(pairs['T'])), cert={}, mon={g: float(np.mean(list(mon_sq[g].values()))) for g in GAMMAS})
    # bootstrap over engines
    ids = np.unique(pairs['eng']); idx_by = {i: np.where(pairs['eng'] == i)[0] for i in ids}
    def certs(sub):
        ii = np.concatenate([idx_by[i] for i in sub]); hbar = pairs['hc'][ii].mean(); out = {'h_cond': float(hbar)}
        out['h_marg'] = None
        for m in names: out[m] = float(hbar-0.5*np.log(2*np.pi*np.e*np.mean((pred[m][ii]-pairs['T'][ii])**2)))
        out['mse'] = {m: float(np.mean((pred[m][ii]-pairs['T'][ii])**2)) for m in names}; return out
    base = certs(ids); res['cert'] = base; boots = [certs(RNG.choice(ids, len(ids))) for _ in range(300)]
    res['cert_ci'] = {m: [float(np.percentile([b[m] for b in boots], 2.5)), float(np.percentile([b[m] for b in boots], 97.5))] for m in names}
    # ---- R3: linear scalar HIs of (level, velocity) from the out-of-fold ridge HI (features standardised globally; angle from the level axis).
    # Stages are defined (a) by the life fraction (oracle stage) and (b) by the age (observable stage, equal-count bins).
    rows = []
    for i, h in hi_oof.items():
        lev, vel = tmean(h, 10), tslope(h, 30); t = np.arange(1, L[i]+1); phi = t/L[i]
        for j in range(len(t)):
            if t[j] >= T0 and phi[j] >= 0.2 and phi[j] < 0.95: rows.append((i, phi[j], lev[j], vel[j], L[i]-t[j], t[j]))
    R = np.array(rows); sD, sV = R[:, 2].std(), R[:, 3].std(); Z1, Z2, Y = R[:, 2]/sD, R[:, 3]/sV, R[:, 4]; folds3 = list(GroupKFold(5).split(R, groups=R[:, 0])); angles = np.arange(0, 91, 5)
    def stage_analysis(stg, nst):
        def cv_stage(feat, s):
            m = stg == s; o = np.full(len(R), np.nan)
            for a, b in folds3:
                a2 = a[m[a]]; X = np.c_[feat, np.ones(len(R))]; w = np.linalg.lstsq(X[a2], Y[a2], rcond=None)[0]; o[b] = X[b]@w
            return float(np.mean((o[m]-Y[m])**2))
        cvang = np.array([[cv_stage((np.cos(np.radians(a))*Z1+np.sin(np.radians(a))*Z2)[:, None], s) for a in angles] for s in range(nst)]); a_star = int(angles[np.argmin(cvang.sum(0))]); out = []
        for s in range(nst):
            m = stg == s; both = cv_stage(np.c_[Z1, Z2], s); w = np.linalg.lstsq(np.c_[Z1[m], Z2[m], np.ones(m.sum())], Y[m], rcond=None)[0]
            out.append(dict(n=int(m.sum()), mean_T=float(Y[m].mean()), mse=dict(both=both, level=float(cvang[s, 0]), velocity=float(cvang[s, -1]), static=float(cvang[s, list(angles).index(a_star)])), angle_opt=float(np.degrees(np.arctan2(abs(w[1]), abs(w[0]))))))
        return dict(static_angle=a_star, stages=out)
    stg_phi = np.digitize(R[:, 1], [0.2, 0.35, 0.5, 0.65, 0.8])-1; stg_age = np.digitize(R[:, 5], np.quantile(R[:, 5], [0.2, 0.4, 0.6, 0.8]))
    res['stage_phi'] = stage_analysis(stg_phi, 5); res['stage_age'] = stage_analysis(stg_age, 5); res['stage_age']['age_edges'] = [float(x) for x in np.quantile(R[:, 5], [0.2, 0.4, 0.6, 0.8])]
    # ---- R4: regime-blind versus regime-aware linear read-out of the life fraction
    r4 = {'blind': [], 'norm': [], 'aware': []}; ref = []
    for kf, (tri, tei) in enumerate(folds):
        P = Prep(eng, tri, multi); tmax = int(np.quantile(L[tri], 0.9)); Zg = np.concatenate([eng[i][0] for i in tri]); gm, gs = Zg.mean(0), Zg.std(0)+1e-9
        def data(idx, mode):
            X, y, rg = [], [], []
            for i in idx:
                r = np.arange(T0, min(L[i]-1, tmax)+1)-1
                x = (eng[i][0]-gm)/gs if mode == 'blind' else P.norm(eng[i][0], eng[i][1]); X.append(x[r]); y.append((r+1)/L[i]); rg.append(P.reg(eng[i][1])[r])
            return np.vstack(X), np.concatenate(y), np.concatenate(rg)
        for mode in ['blind', 'norm']:
            Xa, ya, _ = data(tri, mode); Xb, yb, _ = data(tei, mode); Za = np.c_[Xa, np.ones(len(Xa))]; w = np.linalg.solve(Za.T@Za+10*np.eye(Za.shape[1]), Za.T@ya); r4[mode] += list((np.c_[Xb, np.ones(len(Xb))]@w-yb)**2)
        Xa, ya, ra = data(tri, 'norm'); Xb, yb, rb = data(tei, 'norm'); pb = np.zeros(len(yb))
        for g in np.unique(ra):
            ma = ra == g; Za = np.c_[Xa[ma], np.ones(ma.sum())]; w = np.linalg.solve(Za.T@Za+10*np.eye(Za.shape[1]), Za.T@ya[ma]); mb = rb == g; pb[mb] = np.c_[Xb[mb], np.ones(mb.sum())]@w
        r4['aware'] += list((pb-yb)**2)
    res['regime_rmse_phi'] = {k: float(np.sqrt(np.mean(v))) for k, v in r4.items()}
    res['time_s'] = time.time()-t_start; return res
if __name__ == '__main__':
    for fd in (sys.argv[1:] or ['FD001', 'FD002', 'FD003', 'FD004']):
        r = run(fd); json.dump(r, open(f'../results/realdata_{fd}.json', 'w'), indent=1); print(fd, 'cert (fuse):', {k: round(v, 2) for k, v in r['cert'].items() if k in ('h_cond', 'age only', 'ridge HI', 'PCA-HI', 'sensors')}, flush=True)
