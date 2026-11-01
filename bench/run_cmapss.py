"""C-MAPSS FD001 protocol (frozen before running).  100 official training engines (run to failure) are split by id into 80 'fit' engines and 20 'calibration'
engines (id % 5 == 0).  All HI constructions are trained on the 80 fit engines only.  Architecture/beta selection (bottleneck models) and predictor calibration
(c, w_h, w_a) use the 20 calibration engines.  The official 100 test engines (truncated) are predicted ONCE at their last observed cycle against the official RUL labels."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pickle, sys, time, os
import bench as B, bench2 as B2
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
D = '/home/user/data/cmapss_raw/6. Turbofan Engine Degradation Simulation Data Set'
SENS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]; STEPS = 1500; SEEDS = [0, 1, 2]
def load_fd(fd='FD001'):
    def read(f):
        a = np.loadtxt(f'{D}/{f}_{fd}.txt'); return a
    tr, te = read('train'), read('test'); rul = np.loadtxt(f'{D}/RUL_{fd}.txt')
    cols = [4+s for s in SENS]  # col 0 engine, 1 cycle, 2-4 settings, 5.. sensors (sensor k at column 4+k)
    def split(a): return [a[a[:, 0] == e][:, cols] for e in np.unique(a[:, 0])]
    return split(tr), split(te), rul
def train_steps(Xtr, phys, beta, seed, arch, steps=STEPS):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); mod = B.PGIB(Xtr[0].shape[1], phys, arch=arch); opt = torch.optim.AdamW(mod.parameters(), lr=1e-2, weight_decay=1e-3); Xt = [torch.tensor(x) for x in Xtr]
    for s in range(steps):
        X = Xt[rng.integers(len(Xt))]; opt.zero_grad(); H, u, sd = mod.path(X, True); loss = B.nll_cell(H, sB2=torch.exp(2*mod.lsB)) + beta*mod.kl(H, u, sd) + 10*(H[-1]-1)**2; loss.backward(); nn.utils.clip_grad_norm_(mod.parameters(), 5.0); opt.step()
    return mod
def vae_steps(Xtr, Xte_list, seed, steps=STEPS):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); mod = B.VAE(Xtr[0].shape[1]); opt = torch.optim.Adam(mod.parameters(), lr=1e-2); Xt = [torch.tensor(x) for x in Xtr]
    for s in range(steps):
        X = Xt[rng.integers(len(Xt))]; o, _ = mod.gru(X[None]); mu, lv = mod.mu(o)[0, :, 0], mod.lv(o)[0, :, 0]; z = mu+torch.randn_like(mu)*torch.exp(0.5*lv)
        loss = F.mse_loss(mod.dec(z[:, None]), X) + (-0.5*(1+lv-mu**2-lv.exp()).mean())*0.1; opt.zero_grad(); loss.backward(); opt.step()
    def h(X):
        with torch.no_grad(): o, _ = mod.gru(torch.tensor(X)[None]); return mod.mu(o)[0, :, 0].numpy()
    f = B.affine_fit([h(x) for x in Xtr]); return [f(h(x)) for x in Xte_list], [f(h(x)) for x in Xtr]
def hi_ridge(Xtr, Xte_list, alpha=10.0):
    Z = np.concatenate(Xtr); y = np.concatenate([np.linspace(0, 1, len(x)) for x in Xtr]); Za = np.hstack([Z, np.ones((len(Z), 1))]); w = np.linalg.solve(Za.T@Za + alpha*np.eye(Za.shape[1]), Za.T@y)
    f = lambda x: np.hstack([x, np.ones((len(x), 1))])@w; fa = B.affine_fit([f(x) for x in Xtr]); return [fa(f(x)) for x in Xte_list], [fa(f(x)) for x in Xtr]
def predict_last(H, sB2, params, ap, grid, y_true, epi=True):
    c, wh, wa = params; dg = float(grid[1]-grid[0]); idx = np.array([len(H)-1]); lhi, lag = B2.comps(H, sB2, c, ap, grid, idx, epi); logp, pdf = B2.pooled(lhi, lag, wh, wa, dg)
    cdf = torch.cumsum(pdf, 1)*dg; cdf = cdf/cdf[:, -1:]; q = lambda p: float(grid[(cdf < p).sum(1).clamp(max=len(grid)-1)])
    y = torch.tensor([[float(y_true)]]); crps = float((((cdf-(grid[None] >= y).double())**2).sum(1)*dg)); return q(0.5), q(0.025), q(0.975), crps
if __name__ == '__main__':
    t0 = time.time(); tr, te, rul = load_fd('FD001'); ids = np.arange(100); cal_ids = [i for i in ids if (i+1) % 5 == 0]; fit_ids = [i for i in ids if (i+1) % 5 != 0]
    allv = np.concatenate([tr[i] for i in fit_ids]); mu_, sd_ = allv.mean(0), allv.std(0)+1e-9; nz = lambda x: (x-mu_)/sd_
    Xfit = [nz(tr[i]) for i in fit_ids]; Xcal = [nz(tr[i]) for i in cal_ids]; Xte = [nz(x) for x in te]; lives = [len(x)-1 for x in Xfit]; ap = B2.age_par(lives); grid = B2.make_grid(max(len(x) for x in tr)); age_cal = B2.age_par(lives)
    res = dict(methods={}, cal={}, sel={}, y=rul, age=[len(x)-1 for x in Xte]); print('data ok', len(Xfit), len(Xcal), len(Xte), flush=True)
    def cal_paths(Hcal, sB2s): return [dict(H=h, sB2=s, age=ap) for h, s in zip(Hcal, sB2s)]
    def rec(name, Hte, sB2, params, ap_, seed=0, epi=True):
        r = [predict_last(h, sB2, params, ap_, grid, rul[i], epi) for i, h in enumerate(Hte)]; res['methods'].setdefault(name, []).append(dict(seed=seed, med=np.array([a[0] for a in r]), lo=np.array([a[1] for a in r]), hi=np.array([a[2] for a in r]), crps=np.array([a[3] for a in r]), Hlast=np.array([h[-1] for h in Hte])))
    # age-only
    rec('Age-only', [np.linspace(0, 1, len(x)) for x in Xte], None, (1.0, 0.0, 1.0), ap)
    for name, fn in [('PCA-HI', B.hi_pca), ('Mon-opt HI', B.hi_mon), ('Ridge-HI (supervised)', hi_ridge)]:
        out, _ = fn(Xfit, Xcal+Xte); Hc, Ht = out[:len(Xcal)], out[len(Xcal):]; cal = B2.calibrate(cal_paths(Hc, [None]*len(Hc)), grid); res['cal'][name] = cal; rec(name, Ht, None, cal['fuse'], ap); rec(name+'|nofuse', Ht, None, cal['nofuse'], ap); print(name, cal['fuse'], '%.0fs' % (time.time()-t0), flush=True)
    out, _ = vae_steps(Xfit, Xcal+Xte, 100); cal = B2.calibrate(cal_paths(out[:len(Xcal)], [None]*len(Xcal)), grid); res['cal']['beta-VAE-HI'] = cal
    for s in SEEDS:
        out, _ = vae_steps(Xfit, Xte, s); rec('beta-VAE-HI', out, None, cal['fuse'], ap, s); rec('beta-VAE-HI|nofuse', out, None, cal['nofuse'], ap, s)
    print('vae done %.0fs' % (time.time()-t0), flush=True)
    for name, phys in [('VIB-HI (no physics)', False), ('PGIB-HI', True)]:
        sc, store = {}, {}
        for a in ('gru', 'win'):
            for b in (0.03, 0.3, 3.0, 30.0):
                mod = train_steps(Xfit, phys, b, 100, a, steps=900); v = []; ps = []
                with torch.no_grad():
                    s2 = float(torch.exp(2*mod.lsB))
                    for x in Xcal: H = B.pgib_hi(mod, x); v.append(B.nll_cell(H, sB2=s2).item()); ps.append(H.numpy())
                sc[(a, b)] = float(np.nanmean(v)); store[(a, b)] = (ps, s2); print(name, a, b, round(sc[(a, b)], 2), '%.0fs' % (time.time()-t0), flush=True)
        (arch, b) = min(sc, key=sc.get); res['sel'][name] = (arch, b, {str(k): v for k, v in sc.items()}); ps, s2 = store[(arch, b)]; cal = B2.calibrate(cal_paths(ps, [s2]*len(ps)), grid); res['cal'][name] = cal; Hs_, s2s_ = [], []
        for s in SEEDS:
            mod = train_steps(Xfit, phys, b, s, arch); Ht = [B.pgib_hi(mod, x).numpy() for x in Xte]; s2 = float(torch.exp(2*mod.lsB)); Hs_.append(Ht); s2s_.append(s2)
            rec(name, Ht, s2, cal['fuse'], ap, s); rec(name+'|nofuse', Ht, s2, cal['nofuse'], ap, s)
            if phys:
                rec('PGIB-HI|noEpi', Ht, s2, cal['fuse'], ap, s, epi=False); m0 = train_steps(Xfit, True, 0.0, s, arch); rec('PGIB-HI|beta=0', [B.pgib_hi(m0, x).numpy() for x in Xte], float(torch.exp(2*m0.lsB)), cal['fuse'], ap, s)
        if phys: Hm = [np.mean([Hs_[k][i] for k in range(len(SEEDS))], 0) for i in range(len(Xte))]; rec('PGIB-HI (ens)', Hm, float(np.mean(s2s_)), cal['fuse'], ap); rec('PGIB-HI (ens)|nofuse', Hm, float(np.mean(s2s_)), cal['nofuse'], ap)
        print(name, 'selected', arch, b, cal['fuse'], '%.0fs' % (time.time()-t0), flush=True)
    os.makedirs('out2', exist_ok=True); pickle.dump(res, open('out2/cmapss_FD001.pkl', 'wb')); print('done %.0fs' % (time.time()-t0))
