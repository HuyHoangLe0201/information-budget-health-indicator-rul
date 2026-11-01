"""v2 protocol (frozen before the confirmation runs).  Differences from bench.py (v1):
 (i)  the predictor is calibrated on OUT-OF-SAMPLE HI paths (inner leave-one-training-unit-out), for every HI;
 (ii) the first-hitting-time density is fused with an age-based lifetime prior by log-linear pooling, weights chosen by inner validation;
 (iii) the tau-grid depends only on training lives (v1 used 3x the held-out unit's own life);
 (iv) PGIB-HI is also reported as a 3-seed ensemble (mean HI path).
All decisions use training units only."""
import numpy as np, torch, pickle, sys, os, time
import bench as B
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
CS = np.logspace(-2, 2, 9); W = [(1, 0), (0.5, 0), (0.25, 0), (1, 0.25), (1, 0.5), (1, 1), (0.5, 0.5), (0.5, 1), (0.25, 0.5), (0.25, 1)]
GN = 3000

def age_par(lives):
    l = np.log(np.asarray(lives, float)); return float(l.mean()), float(max(l.std(ddof=1) if len(l) > 1 else 0.3, 0.15))
def log_age(tau, age, mu, sd):
    x = age + tau; lf = -torch.log(x*sd*np.sqrt(2*np.pi)) - (torch.log(x)-mu)**2/(2*sd**2)
    S = 0.5*(1-torch.erf((torch.log(age.clamp(min=0.5))-mu)/(sd*np.sqrt(2)))); return lf - torch.log(S.clamp(min=1e-6))
def make_grid(maxlife): return torch.linspace(0.25, 3.0*maxlife, GN)
def comps(H, sB2, c, ap, grid, idx, epi=True):
    Hs, mu, var, wd = B.win_stats(torch.as_tensor(np.asarray(H, float))); base = var if sB2 is None else sB2*torch.ones_like(var); s2 = c*base
    a = (1-Hs[idx]).clamp(min=0.02)[:, None]; mus = mu[idx].clamp(min=1e-6)[:, None]; sm2 = (s2[idx]/wd[idx])[:, None] if epi else torch.zeros_like(a)
    lhi = B.logpdf(grid[None], a, mus, s2[idx][:, None], sm2); lag = log_age(grid[None], torch.as_tensor(idx, dtype=torch.float64)[:, None], *ap); return lhi, lag
def pooled(lhi, lag, wh, wa, dg):
    lp = torch.nan_to_num(wh*lhi + wa*lag, nan=-1e9, neginf=-1e9, posinf=-1e9); mx = lp.max(1, keepdim=True).values; p = torch.exp(lp-mx); Z = p.sum(1, keepdim=True)*dg
    return lp-mx-torch.log(Z), p/Z
def nll_at(logp, tau, grid, dg):
    f = ((tau-grid[0])/dg).clamp(0, len(grid)-1.001); i0 = f.floor().long(); w = f-i0; v = logp.gather(1, i0[:, None])[:, 0]*(1-w) + logp.gather(1, (i0+1)[:, None])[:, 0]*w; return -v.mean()
def calibrate(paths, grid):
    dg = float(grid[1]-grid[0]); res = {}; best = {'fuse': (1e18, None), 'nofuse': (1e18, None)}
    for c in CS:
        cp = []
        for p in paths:
            idx = B.train_idx(len(p['H'])); lhi, lag = comps(p['H'], p['sB2'], c, p['age'], grid, idx); tau = torch.tensor((len(p['H'])-1-idx).astype(float)).clamp(min=1.0); cp.append((lhi, lag, tau))
        for wh, wa in W:
            v = float(np.mean([nll_at(pooled(l1, l2, wh, wa, dg)[0], t, grid, dg).item() for l1, l2, t in cp]))
            if not np.isfinite(v): continue
            k = 'nofuse' if wa == 0 else 'fuse'
            if v < best[k][0]: best[k] = (v, (float(c), wh, wa))
            if wa == 0 and v < best['fuse'][0]: pass
    # 'fuse' search space includes the no-age entries, so fuse <= nofuse in-sample by construction
    for wh, wa in W:
        pass
    allbest = min(best.values(), key=lambda z: z[0])
    return dict(fuse=allbest[1], nofuse=best['nofuse'][1], nll_fuse=allbest[0], nll_nofuse=best['nofuse'][0])
def predict2(H, sB2, params, ap, grid, epi=True):
    c, wh, wa = params; dg = float(grid[1]-grid[0]); idx = B.eval_idx(len(H)); lhi, lag = comps(H, sB2, c, ap, grid, idx, epi); logp, pdf = pooled(lhi, lag, wh, wa, dg)
    cdf = torch.cumsum(pdf, 1)*dg; cdf = cdf/cdf[:, -1:]; q = lambda p: grid[(cdf < p).sum(1).clamp(max=len(grid)-1)]
    y = torch.tensor((len(H)-1-idx).astype(float))[:, None]; crps = (((cdf-(grid[None] >= y).double())**2).sum(1)*dg)
    return q(0.5).numpy(), q(0.025).numpy(), q(0.975).numpy(), crps.numpy(), y[:, 0].numpy()

def inner_set(n): return list(range(n)) if n <= 5 else [int(round(q)) for q in np.linspace(0, n-1, 3)]

def run_fold2(args):
    ds, test_name = args; t0 = time.time(); data = B.load(ds); names = list(data); trn = [n for n in names if n != test_name]
    Xraw = [data[n]['X'] for n in trn]; lives = [len(data[n]['X'])-1 for n in trn]; Z = np.concatenate(Xraw); mu_, sd_ = Z.mean(0), Z.std(0)+1e-9
    Xtr = [(x-mu_)/sd_ for x in Xraw]; Xte = (data[test_name]['X']-mu_)/sd_; stride = data[test_name]['stride']; grid = make_grid(max(lives)); inn = inner_set(len(Xtr))
    ap_all = age_par(lives); res = dict(test=test_name, ds=ds, stride=stride, m=len(Xte), methods={}, cal={}, betas={}, X_te=Xte)
    def ipaths(Hs, sB2s): return [dict(H=h, sB2=s2, age=age_par([lives[j] for j in range(len(lives)) if j != i])) for h, s2, i in zip(Hs, sB2s, inn)]
    def record(name, H, sB2, params, epi=True, seed=0):
        med, lo, hi, crps, y = predict2(H, sB2, params, ap_all, grid, epi); res['methods'].setdefault(name, []).append(dict(seed=seed, H=np.asarray(H, float), med=med, lo=lo, hi=hi, crps=crps, y=y))
    # age-only (no HI)
    Hd = np.linspace(0, 1, len(Xte)); record('Age-only', Hd, None, (1.0, 0.0, 1.0))
    # non-neural / unsupervised HIs
    for name, fn in [('PCA-HI', B.hi_pca), ('Mon-opt HI', B.hi_mon)]:
        Hin = [fn([Xtr[j] for j in range(len(Xtr)) if j != i], [Xtr[i]])[0][0] for i in inn]; cal = calibrate(ipaths(Hin, [None]*len(inn)), grid); res['cal'][name] = cal; Hte = fn(Xtr, [Xte])[0][0]
        record(name, Hte, None, cal['fuse']); record(name+'|nofuse', Hte, None, cal['nofuse'])
    Hin = [B.hi_vae([Xtr[j] for j in range(len(Xtr)) if j != i], [Xtr[i]], 100)[0][0] for i in inn]; cal = calibrate(ipaths(Hin, [None]*len(inn)), grid); res['cal']['beta-VAE-HI'] = cal
    for s in B.SEEDS: Hte = B.hi_vae(Xtr, [Xte], s)[0][0]; record('beta-VAE-HI', Hte, None, cal['fuse'], seed=s); record('beta-VAE-HI|nofuse', Hte, None, cal['nofuse'], seed=s)
    # bottleneck models
    for name, phys in [('VIB-HI (no physics)', False), ('PGIB-HI', True)]:
        sc, store = {}, {}
        for a in ('gru', 'win'):
            for b in (0.03, 0.3, 3.0, 30.0):
                v, ps = [], []
                for i in inn:
                    mod = B.train_pgib([Xtr[j] for j in range(len(Xtr)) if j != i], phys, b, 100, 120, a)
                    with torch.no_grad(): H = B.pgib_hi(mod, Xtr[i]); s2 = float(torch.exp(2*mod.lsB)); v.append(B.nll_cell(H, sB2=s2).item())
                    ps.append((H.numpy(), s2))
                sc[(a, b)] = float(np.nanmedian(v)); store[(a, b)] = ps
        (arch, b) = min(sc, key=sc.get); res['betas'][name] = (arch, b, {str(k): v for k, v in sc.items()}); cal = calibrate(ipaths([p[0] for p in store[(arch, b)]], [p[1] for p in store[(arch, b)]]), grid); res['cal'][name] = cal
        Hs_, s2s_ = [], []
        for s in B.SEEDS:
            mod = B.train_pgib(Xtr, phys, b, s, arch=arch); H = B.pgib_hi(mod, Xte).numpy(); s2 = float(torch.exp(2*mod.lsB)); Hs_.append(H); s2s_.append(s2)
            record(name, H, s2, cal['fuse'], seed=s); record(name+'|nofuse', H, s2, cal['nofuse'], seed=s)
            if phys:
                record('PGIB-HI|noEpi', H, s2, cal['fuse'], epi=False, seed=s); mod0 = B.train_pgib(Xtr, True, 0.0, s, arch=arch); record('PGIB-HI|beta=0', B.pgib_hi(mod0, Xte).numpy(), float(torch.exp(2*mod0.lsB)), cal['fuse'], seed=s)
        if phys: Hm = np.mean(Hs_, 0); sm = float(np.mean(s2s_)); record('PGIB-HI (ens)', Hm, sm, cal['fuse']); record('PGIB-HI (ens)|nofuse', Hm, sm, cal['nofuse'])
    res['time'] = time.time()-t0; os.makedirs('out2', exist_ok=True); pickle.dump(res, open(f'out2/{ds}_{test_name}.pkl', 'wb')); print(ds, test_name, 'done %.0fs' % res['time'], {k: v[:2] for k, v in res['betas'].items()}, {k: v['fuse'] for k, v in res['cal'].items()}, flush=True); return test_name

if __name__ == '__main__':
    from multiprocessing import Pool
    ds = sys.argv[1]; folds = [(ds, n) for n in B.CFG[ds]['cells'] if not os.path.exists(f'out2/{ds}_{n}.pkl')]
    if len(sys.argv) > 2 and sys.argv[2] == 'one': run_fold2(folds[0])
    else:
        with Pool(4) as p: p.map(run_fold2, folds, chunksize=1)
