"""PGIB-KFR: regime-aware observation model (frozen before running; run once per subset).
Process model shared across regimes (positive velocity, physics prior); OBSERVATION model regime-dependent: level estimator l_k and filter gain kappa_k use regime-specific
linear heads on the encoder features, and the encoder also receives the one-hot regime.  Baseline added for fairness: regime-specific ridge HI.
Selection between kf and kfr, and of beta, uses only the NLL on the calibration engines.  FD001/FD003 (one regime) serve as a non-regression check.
Existing predictions (baselines, PGIB-KF) are loaded unchanged from out2/cmapss_{FD}_v3.pkl."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pickle, sys, time, os
import bench as B, bench2 as B2, run_cmapss4 as R4
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
class PGIBKR(B.PGIB):
    def __init__(s, ds, R, phys=True, hid=16):
        super().__init__(ds, phys, hid, arch='win'); s.ds, s.R = ds, R
        s.mlp = nn.Sequential(nn.Linear(2*ds+R, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        s.Wl = nn.Parameter(0.1*torch.randn(R, hid)); s.bl = nn.Parameter(torch.zeros(R)); s.Wk = nn.Parameter(0.1*torch.randn(R, hid)); s.bk = nn.Parameter(torch.full((R,), -2.0))
    def path(s, X, noise=True):
        S, oh = X[:, :s.ds], X[:, s.ds:]; Xp = torch.cat([S[:1].expand(2, -1), S]); xm = (Xp[2:]+Xp[1:-1]+Xp[:-2])/3; Xq = torch.cat([xm[:1].expand(8, -1), xm]); o = s.mlp(torch.cat([xm, xm-Xq[:-8], oh], 1))
        u = 0.02*F.softplus(s.fu(o))[:, 0]+1e-4; sd = 0.01*F.softplus(s.fs(o))[:, 0]+1e-4; l = (o*(oh@s.Wl)).sum(1)+oh@s.bl; kap = torch.sigmoid((o*(oh@s.Wk)).sum(1)+oh@s.bk); eps = torch.randn_like(u) if noise else torch.zeros_like(u)
        Hp = torch.zeros(()); Hs, ue, se = [], [], []
        for k in range(len(u)):
            Hn = (1-kap[k])*(Hp+u[k]+sd[k]*eps[k]) + kap[k]*l[k]; ue.append((1-kap[k])*u[k]+kap[k]*(l[k]-Hp)); se.append((1-kap[k])*sd[k]+1e-4); Hs.append(Hn); Hp = Hn
        return torch.stack(Hs), torch.stack(ue), torch.stack(se)
def train_kfr(Xtr, ds, R, beta, seed, steps):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); mod = PGIBKR(ds, R); opt = torch.optim.AdamW(mod.parameters(), lr=1e-2, weight_decay=1e-3); Xt = [torch.tensor(x) for x in Xtr]
    for s_ in range(steps):
        X = Xt[rng.integers(len(Xt))]; opt.zero_grad(); H, u, sd = mod.path(X, True); loss = B.nll_cell(H, sB2=torch.exp(2*mod.lsB)) + beta*mod.kl(H, u, sd) + 10*(H[-1]-1)**2; loss.backward(); nn.utils.clip_grad_norm_(mod.parameters(), 5.0); opt.step()
    return mod
def hi_ridge_reg(Xtr, Xte_list, ds, R, alpha=10.0):
    def design(x): return np.hstack([x[:, :ds], np.ones((len(x), 1))])
    W = []
    for r in range(R):
        rows = [x[x[:, ds+r] > 0.5] for x in Xtr]; Z = np.concatenate([design(a) for a in rows]); y = np.concatenate([np.linspace(0, 1, len(x))[x[:, ds+r] > 0.5] for x in Xtr]); W.append(np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@y))
    f = lambda x: np.array([design(x)[k]@W[int(np.argmax(x[k, ds:]))] for k in range(len(x))]); fa = B.affine_fit([f(x) for x in Xtr]); return [fa(f(x)) for x in Xte_list], [fa(f(x)) for x in Xtr]
if __name__ == '__main__':
    FD = sys.argv[1]; t0 = time.time(); res = pickle.load(open(f'out2/cmapss_{FD}_v3.pkl', 'rb')); tr, te, rul = R4.load_fd(FD); ids = np.arange(len(tr)); cal_ids = [i for i in ids if (i+1) % 5 == 0]; fit_ids = [i for i in ids if (i+1) % 5 != 0]
    multi = FD in ('FD002', 'FD004'); R = 6 if multi else 1; fs = np.concatenate([tr[i][0] for i in fit_ids])
    if multi:
        from sklearn.cluster import KMeans
        fo = np.concatenate([tr[i][1] for i in fit_ids]); om, osd = fo.mean(0), fo.std(0)+1e-9; km = KMeans(6, n_init=10, random_state=0).fit((fo-om)/osd); fr = km.labels_
        mus = np.array([fs[fr == r].mean(0) for r in range(6)]); sds = np.array([fs[fr == r].std(0)+1e-9 for r in range(6)]); regime = lambda p: km.predict((p[1]-om)/osd)
    else:
        mu0, sd0 = fs.mean(0), fs.std(0)+1e-9; mus = mu0[None]; sds = sd0[None]; regime = lambda p: np.zeros(len(p[0]), int)
    def nzr(p): r = regime(p); return np.hstack([(p[0]-mus[r])/sds[r], np.eye(R)[r]])
    ds = fs.shape[1]; Xfit = [nzr(tr[i]) for i in fit_ids]; Xcal = [nzr(tr[i]) for i in cal_ids]; Xte = [nzr(x) for x in te]; lives = [len(x)-1 for x in Xfit]; ap = B2.age_par(lives); grid = B2.make_grid(max(len(x[0]) for x in tr))
    print('data ok', FD, len(Xfit), len(Xcal), len(Xte), 'R', R, flush=True)
    def cal_paths(Hc, s2s): return [dict(H=h, sB2=s, age=ap) for h, s in zip(Hc, s2s)]
    def rec(name, Hte, sB2, params, seed=0):
        r = [R4.predict_last(h, sB2, params, ap, grid, rul[i]) for i, h in enumerate(Hte)]; res['methods'].setdefault(name, []).append(dict(seed=seed, med=np.array([a[0] for a in r]), lo=np.array([a[1] for a in r]), hi=np.array([a[2] for a in r]), crps=np.array([a[3] for a in r]), Hlast=np.array([h[-1] for h in Hte])))
    # regime-specific ridge baseline
    out, _ = hi_ridge_reg(Xfit, Xcal+Xte, ds, R); cal = B2.calibrate(cal_paths(out[:len(Xcal)], [None]*len(Xcal)), grid); res['cal']['Ridge-reg-HI'] = cal; rec('Ridge-reg-HI', out[len(Xcal):], None, cal['fuse']); rec('Ridge-reg-HI|nofuse', out[len(Xcal):], None, cal['nofuse']); print('ridge-reg', cal['fuse'], '%.0fs' % (time.time()-t0), flush=True)
    # PGIB-KFR
    sc, store = {}, {}
    for b in (0.03, 0.3, 3.0):
        mod = train_kfr(Xfit, ds, R, b, 100, 900); v, ps = [], []
        with torch.no_grad():
            s2 = float(torch.exp(2*mod.lsB))
            for x in Xcal: H = B.pgib_hi(mod, x); v.append(B.nll_cell(H, sB2=s2).item()); ps.append(H.numpy())
        sc[b] = float(np.nanmean(v)); store[b] = (ps, s2); print('PGIB-KFR', b, round(sc[b], 2), '%.0fs' % (time.time()-t0), flush=True)
    b = min(sc, key=sc.get); res['sel']['PGIB-KFR-HI'] = ('kfr', b, {str(k): v for k, v in sc.items()}); ps, s2 = store[b]; cal = B2.calibrate(cal_paths(ps, [s2]*len(ps)), grid); res['cal']['PGIB-KFR-HI'] = cal; Hs_, s2s_ = [], []
    for s in R4.SEEDS:
        mod = train_kfr(Xfit, ds, R, b, s, R4.STEPS); Ht = [B.pgib_hi(mod, x).numpy() for x in Xte]; s2 = float(torch.exp(2*mod.lsB)); Hs_.append(Ht); s2s_.append(s2); rec('PGIB-KFR-HI', Ht, s2, cal['fuse'], s); rec('PGIB-KFR-HI|nofuse', Ht, s2, cal['nofuse'], s)
    Hm = [np.mean([Hs_[k][i] for k in range(len(R4.SEEDS))], 0) for i in range(len(Xte))]; rec('PGIB-KFR-HI (ens)', Hm, float(np.mean(s2s_)), cal['fuse'])
    kf_nll = min(res['sel']['PGIB-KF-HI'][2].values()); kfr_nll = min(sc.values()); pick = 'PGIB-KFR-HI (ens)' if kfr_nll < kf_nll else 'PGIB-KF-HI (ens)'; res['val_selected'] = (pick, kf_nll, kfr_nll)
    res['methods']['PGIB-KF(R)-HI (val-selected, ens)'] = res['methods'][pick]; print('validation NLL KF %.3f KFR %.3f -> %s' % (kf_nll, kfr_nll, pick), '%.0fs' % (time.time()-t0), flush=True)
    pickle.dump(res, open(f'out2/cmapss_{FD}_v5.pkl', 'wb')); print('done %.0fs' % (time.time()-t0))
