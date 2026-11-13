"""Real-data tests of the theory on a second domain with many units: lithium-ion cells (HUST: 77 cells, one protocol per cell; MATR: 169 cells, fast-charging protocols; Tongji: 88 cells, three NCA/NMC chemistries).
Same design as realdata_checks.py (C-MAPSS): cell-grouped 5-fold cross-validation; population = pairs (cell, age t), T = L - t, T | t = L - t | L > t.
R1  age-conditional information lower bound (non-parametric lifetime law), negative control = age-only predictor;
R2  template shrinkage (Theorem 3) applied to a real HI;
R3  stage-wise optimal linear scalar HI of (level, velocity) (Theorem 4, Corollary 1): static versus stage-wise weights;
R4  protocol-blind versus own-reference normalisation (Proposition 1, with the protocol of a cell playing the role of the regime).
HIs use the charging-phase features only (charge duration, charge voltage at 0.3 and 0.8 Ah); every feature is expressed relative to the cell's own
cycles 2-11.  The discharge capacity, which defines the end of life, is used only by the reference predictor 'capacity'.
Run from bench/:  python3 battery_checks.py HUST|MATR|Tongji"""
import numpy as np, json, sys, time
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from realdata_checks import tmean, tslope, mon, affine, fit_ridge_hi, fit_pca_hi
NAME = sys.argv[1] if len(sys.argv) > 1 else 'HUST'
# time scales (cycles) follow the lifetimes of the data set: HUST median life 1700 cycles, MATR 760, Tongji 360
CFG = {'HUST': dict(T0=100, STRIDE=5, W_LEV=20, W_VEL=100, LAG1=100, LAG2=250, TAU=2.0), 'MATR': dict(T0=50, STRIDE=3, W_LEV=10, W_VEL=50, LAG1=50, LAG2=125, TAU=1.0), 'Tongji': dict(T0=30, STRIDE=2, W_LEV=10, W_VEL=40, LAG1=40, LAG2=100, TAU=1.0), 'TongjiAll': dict(T0=30, STRIDE=2, W_LEV=10, W_VEL=40, LAG1=40, LAG2=100, TAU=1.0)}[NAME]
T0, STRIDE, W_LEV, W_VEL, LAG1, LAG2, TAU = [CFG[k] for k in ('T0', 'STRIDE', 'W_LEV', 'W_VEL', 'LAG1', 'LAG2', 'TAU')]; NF = 5; RNG = np.random.default_rng(0)
CHARGE = [2, 3, 4]; CAP = [0]; GAMMAS = [0.0, 0.5, 0.9, 0.97, 0.99, 0.995]
def load():
    d = np.load(f'data/{NAME.lower()}_feats.npz'); L = d['L']; raw = [d[f'F{i}'].astype(float) for i in range(len(L))]; rel = []
    for f in raw:
        ref = f[1:11].mean(0); r = f-ref; r[:, [0, 1, 2]] = f[:, [0, 1, 2]]/ref[[0, 1, 2]]-1; rel.append(r)   # ratios for capacity and time, differences for voltages
    return L, raw, rel, [str(s) for s in d['ids']]
def lag(h, k): return np.r_[np.full(k, h[0]), h][:len(h)]
def feats_hi(h): return np.c_[tmean(h, W_LEV), tslope(h, W_VEL), lag(h, LAG1), lag(h, LAG2)]
def feats_x(X): return np.hstack([np.column_stack([tmean(X[:, j], W_LEV) for j in range(X.shape[1])]), np.column_stack([tslope(X[:, j], W_VEL) for j in range(X.shape[1])])])
def hgb(): return HGB(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
def cond_entropy_fn(Ltr):
    kde = stats.gaussian_kde(np.log(Ltr)); tau = np.arange(TAU/2, 3500, TAU)
    def h(t):
        l = t+tau; f = kde(np.log(l))/l; S = np.trapezoid(f, tau); p = np.maximum(f/S, 1e-300); return float(-np.trapezoid(p*np.log(p), tau))
    return h
def run():
    t_start = time.time(); L, raw, rel, ids = load(); n = len(L)
    folds = list(GroupKFold(NF).split(np.arange(n), groups=np.arange(n)))
    pairs = dict(eng=[], t=[], T=[], hc=[]); kinds = ['age only', 'ridge HI', 'PCA-HI', 'ridge level only', 'ridge slope only', 'charge sensors', 'capacity (reference)']
    names = kinds+[f'squeeze {g}' for g in GAMMAS]; pred = {m: [] for m in names}; mon_sq = {g: {} for g in GAMMAS}; mon_pred = []; hi_oof = {}
    for kf, (tri, tei) in enumerate(folds):
        tmax = int(np.quantile(L[tri], 0.9)); hc = cond_entropy_fn(L[tri]); hmap = {t: hc(t) for t in range(T0, tmax+1)}
        Zc = np.concatenate([rel[i][:, CHARGE] for i in tri]); mu, sd = Zc.mean(0), Zc.std(0)+1e-9
        Xn = {i: (rel[i][:, CHARGE]-mu)/sd for i in range(n)}
        hr = fit_ridge_hi([Xn[i] for i in tri]); hp = fit_pca_hi([Xn[i] for i in tri]); Hr = {i: hr(Xn[i]) for i in range(n)}; Hp = {i: hp(Xn[i]) for i in range(n)}
        for i in tei: hi_oof[i] = Hr[i]
        Lm = int(L.max()); M = np.full(Lm, np.nan)
        for k in range(Lm):
            v = [Hr[i][k] for i in tri if L[i] > k]; M[k] = np.mean(v) if v else np.nan
        last = np.where(~np.isnan(M))[0].max(); M[last+1:] = M[last]; half = int(np.median(L[tri])); dm = float(np.mean(np.diff(M[:int(np.quantile(L[tri], 0.5))]))); s2 = dm/(np.sqrt(2)*1.645)
        mon_pred.append(float(np.mean(2*stats.norm.cdf(np.diff(M[:half])/(np.sqrt(2)*s2))-1)))
        cm, cs = np.concatenate([rel[i][:, CAP] for i in tri]).mean(0), np.concatenate([rel[i][:, CAP] for i in tri]).std(0)+1e-9
        def build(kind, i, g=None):
            tt = np.arange(1, L[i]+1)[:, None].astype(float)
            if kind == 'age only': return tt
            if kind == 'ridge HI': return np.c_[feats_hi(Hr[i]), tt]
            if kind == 'PCA-HI': return np.c_[feats_hi(Hp[i]), tt]
            if kind == 'ridge level only': return np.c_[tmean(Hr[i], W_LEV), tt]
            if kind == 'ridge slope only': return np.c_[tslope(Hr[i], W_VEL), tt]
            if kind == 'charge sensors': return np.c_[feats_x(Xn[i]), tt]
            if kind == 'capacity (reference)': return np.c_[feats_x((rel[i][:, CAP]-cm)/cs), tt]
            if kind == 'squeeze':
                r = np.random.default_rng(1000*kf+int(g*1000)+i); Hs = M[:L[i]]+(1-g)*(Hr[i]-M[:L[i]])+s2*r.standard_normal(L[i]); return np.c_[feats_hi(Hs), tt], Hs
        sel = lambda i: np.arange(T0, min(L[i]-1, tmax)+1, STRIDE)-1
        tri_sel = [(i, sel(i)) for i in tri]; tei_sel = [(i, sel(i)) for i in tei]
        for i, r in tei_sel: pairs['eng'] += [i]*len(r); pairs['t'] += list(r+1); pairs['T'] += list(L[i]-(r+1)); pairs['hc'] += [hmap[t] for t in r+1]
        def fit_pred(Ftr, Fte):
            Xtr = np.vstack([Ftr(i)[r] for i, r in tri_sel]); ytr = np.concatenate([L[i]-(r+1) for i, r in tri_sel]); m = hgb().fit(Xtr, ytr)
            return np.concatenate([m.predict(Fte(i)[r]) for i, r in tei_sel])
        for kind in kinds: pred[kind] += list(fit_pred(lambda i: build(kind, i), lambda i: build(kind, i)))
        for g in GAMMAS:
            cache = {i: build('squeeze', i, g) for i in range(n)}
            pred[f'squeeze {g}'] += list(fit_pred(lambda i: cache[i][0], lambda i: cache[i][0]))
            for i in tei: mon_sq[g][i] = mon(cache[i][1])
        print('fold', kf, 'done %.0fs' % (time.time()-t_start), flush=True)
    pairs = {k: np.array(v) for k, v in pairs.items()}; pred = {k: np.array(v) for k, v in pred.items()}
    res = dict(fd=NAME, mon_inf_pred=float(np.mean(mon_pred)), n_engines=n, lifetimes=L.tolist(), n_pairs=int(len(pairs['T'])), cert={}, mon={g: float(np.mean(list(mon_sq[g].values()))) for g in GAMMAS})
    ids_u = np.unique(pairs['eng']); idx_by = {i: np.where(pairs['eng'] == i)[0] for i in ids_u}
    def certs(sub):
        ii = np.concatenate([idx_by[i] for i in sub]); hbar = pairs['hc'][ii].mean(); out = {'h_cond': float(hbar)}
        for m in names: out[m] = float(hbar-0.5*np.log(2*np.pi*np.e*np.mean((pred[m][ii]-pairs['T'][ii])**2)))
        out['mse'] = {m: float(np.mean((pred[m][ii]-pairs['T'][ii])**2)) for m in names}; return out
    base = certs(ids_u); res['cert'] = base; boots = [certs(RNG.choice(ids_u, len(ids_u))) for _ in range(300)]
    res['cert_ci'] = {m: [float(np.percentile([b[m] for b in boots], 2.5)), float(np.percentile([b[m] for b in boots], 97.5))] for m in names}
    # ---- R3 stage-wise direction (out-of-fold ridge HI)
    rows = []
    for i, h in hi_oof.items():
        lev, vel = tmean(h, W_LEV), tslope(h, W_VEL); t = np.arange(1, L[i]+1); phi = t/L[i]
        for j in range(T0-1, L[i], STRIDE):
            if 0.2 <= phi[j] < 0.95: rows.append((i, phi[j], lev[j], vel[j], L[i]-t[j], t[j]))
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
    # ---- R4: protocol-blind (population z-score of the raw charging features) versus own-reference (relative to cycles 2-11) linear read-out of the life fraction
    r4 = {'blind': [], 'norm': []}
    for kf, (tri, tei) in enumerate(folds):
        tmax = int(np.quantile(L[tri], 0.9))
        def data(idx, mode):
            src = raw if mode == 'blind' else rel; return np.vstack([src[i][:, CHARGE][np.arange(T0, min(L[i]-1, tmax)+1, STRIDE)-1] for i in idx]), np.concatenate([np.arange(T0, min(L[i]-1, tmax)+1, STRIDE)/L[i] for i in idx])
        for mode in ['blind', 'norm']:
            Xa, ya = data(tri, mode); Xb, yb = data(tei, mode); m_, s_ = Xa.mean(0), Xa.std(0)+1e-9; Za = np.c_[(Xa-m_)/s_, np.ones(len(Xa))]; w = np.linalg.solve(Za.T@Za+10*np.eye(Za.shape[1]), Za.T@ya); r4[mode] += list((np.c_[(Xb-m_)/s_, np.ones(len(Xb))]@w-yb)**2)
    res['regime_rmse_phi'] = {k: float(np.sqrt(np.mean(v))) for k, v in r4.items()}
    res['time_s'] = time.time()-t_start; return res
if __name__ == '__main__':
    r = run(); json.dump(r, open(f'../results/realdata_{NAME}.json', 'w'), indent=1)
    print('cert:', {k: round(v, 2) for k, v in r['cert'].items() if k not in ('mse',)}); print('CI:', {k: [round(a, 2) for a in v] for k, v in r['cert_ci'].items() if not k.startswith('squeeze')})
    print('mon', r['mon'], 'pred', r['mon_inf_pred']); print('regime', r['regime_rmse_phi']); print('stage_phi', r['stage_phi']['static_angle'], [round(s['angle_opt']) for s in r['stage_phi']['stages']], 'stage_age', r['stage_age']['static_angle'], [round(s['angle_opt']) for s in r['stage_age']['stages']])
