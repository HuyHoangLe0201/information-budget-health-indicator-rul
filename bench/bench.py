"""Leave-one-unit-out benchmark of health-indicator constructions on NASA PCoE and CALCE CS2.
All HIs share the SAME downstream predictor (Wiener-drift first-hitting-time law with window drift estimate);
only the HI construction differs. Capacity is used only to define EOL (and hence RUL labels)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd, pickle, sys, time, os
from scipy.optimize import differential_evolution
from multiprocessing import Pool
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
CFG = {'nasa': dict(cells=['B0005', 'B0006', 'B0007', 'B0018'], fade=0.25, stride=1, sm=5),
       'calce': dict(cells=['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'], fade=0.20, stride=4, sm=15),
       'femto1': dict(cells=[f'Bearing1_{k}' for k in range(1, 8)], bearing=True, stride=20, sm=5, prefix='femto'),
       'femto2': dict(cells=[f'Bearing2_{k}' for k in range(1, 8)], bearing=True, stride=8, sm=5, prefix='femto'),
       'femto3': dict(cells=[f'Bearing3_{k}' for k in range(1, 4)], bearing=True, stride=10, sm=5, prefix='femto'),
       'ims': dict(cells=['IMS_T1B3', 'IMS_T1B4', 'IMS_T2B1', 'IMS_T3B3'], bearing=True, stride=20, sm=5, prefix='ims'),
       'calce6': dict(cells=['CS2_33', 'CS2_34', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'], fade=0.20, stride=4, sm=15, prefix='calce')}
LO, HI_ = 0.4, 0.9          # evaluation window (fraction of life)
SEEDS = [0, 1, 2]

def load(ds):
    c = CFG[ds]; out = {}
    for name in c['cells']:
        d = np.load(f"data/{c.get('prefix', ds)}_{name}.npz"); X = pd.DataFrame(d['X']).interpolate(limit_direction='both').values
        if c.get('bearing'): eol = len(X)-1      # run-to-failure: EOL is the last snapshot
        else:
            cap = d['cap']; sm = pd.Series(cap).rolling(c['sm'], center=True, min_periods=1).median().values
            eol = int(np.where(sm <= (1-c['fade'])*np.median(cap[:5]))[0][0])
        Xe = X[:eol+1]; s = c['stride']
        m = len(Xe)//s; Xe = Xe[len(Xe)-m*s:].reshape(m, s, -1).mean(1)
        out[name] = dict(X=Xe, eol=eol, stride=s)
    return out

# ---------------- predictor (shared) ----------------
def win_stats(H, sm=5):
    m = H.shape[0]; Hp = torch.cat([H[:1].expand(sm-1), H]); Hs = F.avg_pool1d(Hp[None, None], sm, 1)[0, 0]
    dH = torch.cat([torch.zeros(1), H[1:]-H[:-1]]); c1 = torch.cumsum(dH, 0); c2 = torch.cumsum(dH**2, 0)
    t = torch.arange(m); w = torch.clamp((0.25*t).long(), min=10); t0 = torch.clamp(t-w, min=0); wd = (t-t0).clamp(min=1).to(H.dtype)
    mu = (Hs-Hs[t0])/wd; s1 = c1-c1[t0]; s2 = c2-c2[t0]; var = ((s2-s1**2/wd)/(wd-1).clamp(min=1)).clamp(min=1e-12)
    return Hs, mu, var, wd

def logpdf(tau, a, mu, s2, sm2):
    v = s2 + sm2*tau; return torch.log(a) - 0.5*torch.log(2*np.pi*tau**3*v) - (a-mu*tau)**2/(2*tau*v)

def eval_idx(m): return np.arange(int(LO*m), int(HI_*m)+1)
def train_idx(m): return np.arange(int(0.2*m), int(0.92*m)+1)

def nll_cell(H, c=None, sB2=None, epistemic=True, idx=None):
    Hs, mu, var, wd = win_stats(H); idx = train_idx(len(H)) if idx is None else idx
    s2 = (c*var if sB2 is None else sB2*torch.ones_like(var)); sm2 = (s2/wd if epistemic else torch.zeros_like(s2))
    tau = torch.tensor((len(H)-1-idx).astype(float)).clamp(min=1.0); a = (1-Hs[idx]).clamp(min=0.02)
    return -logpdf(tau, a, mu[idx], s2[idx], sm2[idx]).mean()

def predict(H, m_steps, c=None, sB2=None, epistemic=True):
    """Return arrays over eval window: median, lo, hi, crps (all in steps) from the mixed-IG law."""
    with torch.no_grad():
        Hs, mu, var, wd = win_stats(H); idx = eval_idx(len(H)); s2 = (c*var if sB2 is None else sB2*torch.ones_like(var)); sm2 = s2/wd if epistemic else torch.zeros_like(s2)
        grid = torch.linspace(0.25, 3.0*len(H), 6000); dg = float(grid[1]-grid[0])
        a = (1-Hs[idx]).clamp(min=0.02)[:, None]; mus = mu[idx].clamp(min=1e-6)[:, None]
        pdf = torch.exp(logpdf(grid[None], a, mus, s2[idx][:, None], sm2[idx][:, None])); pdf = torch.nan_to_num(pdf)
        cdf = torch.cumsum(pdf, 1)*dg; cdf = cdf/cdf[:, -1:].clamp(min=1e-12)
        q = lambda p: grid[(cdf < p).sum(1).clamp(max=len(grid)-1)]
        y = torch.tensor((len(H)-1-idx).astype(float))[:, None]
        crps = (((cdf - (grid[None] >= y).double())**2).sum(1)*dg)
        return q(0.5).numpy(), q(0.025).numpy(), q(0.975).numpy(), crps.numpy(), y[:, 0].numpy()

def calibrate_c(Hs_train):
    best, bc = 1e9, 1.0
    for c in np.logspace(-3, 3, 25):
        v = float(np.mean([nll_cell(H, c=c).item() for H in Hs_train]))
        if v < best: best, bc = v, c
    return bc

# ---------------- HI methods ----------------
def mon_tre_pro(Hl):
    mon = np.mean([abs(np.sum(np.sign(np.diff(h))))/(len(h)-1) for h in Hl]); tre = np.mean([abs(np.corrcoef(h, np.arange(len(h)))[0, 1]) for h in Hl])
    fin = np.array([h[-1] for h in Hl]); ini = np.array([h[0] for h in Hl]); pro = np.exp(-np.std(fin)/max(np.mean(np.abs(fin-ini)), 1e-9)); return mon, tre, pro

def affine_fit(Hl):
    st = np.mean([h[:5].mean() for h in Hl]); en = np.mean([h[-5:].mean() for h in Hl]); sgn = 1.0 if en > st else -1.0
    return lambda h: (sgn*h - sgn*st)/(sgn*en - sgn*st)

def ma(h, k=5): return pd.Series(h).rolling(k, min_periods=1).mean().values

def hi_pca(Xtr, Xte):
    Z = np.concatenate(Xtr); U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); w = Vt[0]
    f = affine_fit([x@w for x in Xtr]); return [f(x@w) for x in Xte], [f(x@w) for x in Xtr]

def hi_mon(Xtr, Xte, seed=0):
    def fit(w):
        if np.linalg.norm(w) < 1e-6: return 1e3
        Hl = [ma(x@w/np.linalg.norm(w)) for x in Xtr]; sg = np.sign(np.mean([np.corrcoef(h, np.arange(len(h)))[0, 1] for h in Hl])) or 1.0
        mo, tr, pr = mon_tre_pro([sg*h for h in Hl]); return -(mo+tr+pr)
    r = differential_evolution(fit, [(-1, 1)]*Xtr[0].shape[1], seed=seed, maxiter=60, popsize=12, tol=1e-6, polish=False)
    w = r.x/np.linalg.norm(r.x); f = affine_fit([x@w for x in Xtr]); return [f(x@w) for x in Xte], [f(x@w) for x in Xtr]

class PGIB(nn.Module):
    def __init__(s, d, phys=True, hid=16, arch='gru'):
        super().__init__(); s.arch = arch; s.phys = phys; s.fu = nn.Linear(hid, 1); s.fs = nn.Linear(hid, 1)
        if arch == 'gru': s.gru = nn.GRU(d, hid, batch_first=True)
        else: s.mlp = nn.Sequential(nn.Linear(2*d, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        s.lt1 = nn.Parameter(torch.tensor(np.log(0.01))); s.t2 = nn.Parameter(torch.tensor(0.3)); s.lsB = nn.Parameter(torch.tensor(np.log(0.01)))
    def heads(s, X):
        if s.arch == 'gru': o, _ = s.gru(X[None]); o = o[0]
        else:   # causal feed-forward on (3-step moving average, 8-step difference): no recurrent memory
            Xp = torch.cat([X[:1].expand(2, -1), X]); xm = (Xp[2:]+Xp[1:-1]+Xp[:-2])/3; Xq = torch.cat([xm[:1].expand(8, -1), xm])
            o = s.mlp(torch.cat([xm, xm-Xq[:-8]], 1))
        return 0.02*F.softplus(s.fu(o))[:, 0]+1e-4, 0.01*F.softplus(s.fs(o))[:, 0]+1e-4
    def path(s, X, noise=True):
        u, sd = s.heads(X); eps = torch.randn_like(u) if noise else torch.zeros_like(u); return torch.cumsum(u+sd*eps, 0), u, sd
    def kl(s, H, u, sd):
        Hp = torch.cat([torch.zeros(1), H[:-1]]); sB2 = torch.exp(2*s.lsB); eta = torch.exp(s.lt1)*torch.exp(torch.clamp(s.t2*Hp, -3, 3)) if s.phys else torch.zeros_like(u)
        r = sd**2/sB2; return (0.5*(r-1-torch.log(r)+(u-eta)**2/sB2)).mean()

def train_pgib(Xtr, phys, beta, seed, epochs=200, arch='gru'):
    torch.manual_seed(seed); mod = PGIB(Xtr[0].shape[1], phys, arch=arch); opt = torch.optim.AdamW(mod.parameters(), lr=1e-2, weight_decay=1e-3)
    Xt = [torch.tensor(x) for x in Xtr]
    for ep in range(epochs):
        for X in Xt:
            opt.zero_grad(); H, u, sd = mod.path(X, True); loss = nll_cell(H, sB2=torch.exp(2*mod.lsB)) + beta*mod.kl(H, u, sd) + 10*(H[-1]-1)**2; loss.backward()
            nn.utils.clip_grad_norm_(mod.parameters(), 5.0); opt.step()
    return mod

def pgib_hi(mod, X):
    with torch.no_grad(): return mod.path(torch.tensor(X), False)[0]

def val_nll(mod, X):
    with torch.no_grad(): return nll_cell(pgib_hi(mod, X), sB2=torch.exp(2*mod.lsB)).item()

def select_cfg(Xtr, phys, betas=(0.03, 0.3, 3.0, 30.0), archs=('gru', 'win')):
    sc = {}
    for a in archs:
        for b in betas:   # inner leave-one-train-cell-out validation NLL; the test cell is never touched
            inner = list(range(len(Xtr))) if len(Xtr) <= 5 else [int(round(q)) for q in np.linspace(0, len(Xtr)-1, 3)]   # <=3 inner folds when >5 training cells
            v = [val_nll(train_pgib([Xtr[j] for j in range(len(Xtr)) if j != i], phys, b, 100, 120, a), Xtr[i]) for i in inner]; sc[(a, b)] = float(np.nanmedian(v))
    best = min(sc, key=sc.get); return best, {str(k): v for k, v in sc.items()}

class VAE(nn.Module):
    def __init__(s, d, hid=16):
        super().__init__(); s.gru = nn.GRU(d, hid, batch_first=True); s.mu = nn.Linear(hid, 1); s.lv = nn.Linear(hid, 1); s.dec = nn.Sequential(nn.Linear(1, hid), nn.Tanh(), nn.Linear(hid, d))

def hi_vae(Xtr, Xte, seed, beta=1.0, epochs=300):
    torch.manual_seed(seed); mod = VAE(Xtr[0].shape[1]); opt = torch.optim.Adam(mod.parameters(), lr=1e-2); Xt = [torch.tensor(x) for x in Xtr]
    for ep in range(epochs):
        for X in Xt:
            o, _ = mod.gru(X[None]); mu, lv = mod.mu(o)[0, :, 0], mod.lv(o)[0, :, 0]; z = mu+torch.randn_like(mu)*torch.exp(0.5*lv)
            loss = F.mse_loss(mod.dec(z[:, None]), X) + beta*(-0.5*(1+lv-mu**2-lv.exp()).mean())*0.1; opt.zero_grad(); loss.backward(); opt.step()
    def h(X):
        with torch.no_grad(): o, _ = mod.gru(torch.tensor(X)[None]); return mod.mu(o)[0, :, 0].numpy()
    f = affine_fit([h(x) for x in Xtr]); return [f(h(x)) for x in Xte], [f(h(x)) for x in Xtr]

# ---------------- one LOO fold ----------------
def run_fold(args):
    ds, test_name = args; t0 = time.time(); data = load(ds); names = list(data); trn = [n for n in names if n != test_name]
    Xraw_tr = [data[n]['X'] for n in trn]; Z = np.concatenate(Xraw_tr); mu_, sd_ = Z.mean(0), Z.std(0)+1e-9
    Xtr = [(x-mu_)/sd_ for x in Xraw_tr]; Xte = (data[test_name]['X']-mu_)/sd_; res = dict(test=test_name, ds=ds, stride=data[test_name]['stride'], m=len(Xte), methods={})
    def record(name, Hte, sB2=None, c=None, epi=True, seed=0):
        H = torch.tensor(np.asarray(Hte, float)); med, lo, hi, crps, y = predict(H, len(H), c=c, sB2=sB2, epistemic=epi)
        res['methods'].setdefault(name, []).append(dict(seed=seed, H=H.numpy(), med=med, lo=lo, hi=hi, crps=crps, y=y))
    for name, fn in [('PCA-HI', hi_pca), ('Mon-opt HI', hi_mon)]:
        Hte, Htr = fn(Xtr, [Xte]); c = calibrate_c([torch.tensor(h) for h in Htr]); record(name, Hte[0], c=c); record(name+'|noEpi', Hte[0], c=c, epi=False)
    for s in SEEDS:
        Hte, Htr = hi_vae(Xtr, [Xte], s); c = calibrate_c([torch.tensor(h) for h in Htr]); record('beta-VAE-HI', Hte[0], c=c, seed=s)
    betas = {}
    for name, phys in [('VIB-HI (no physics)', False), ('PGIB-HI', True)]:
        (arch, b), sc = select_cfg(Xtr, phys); betas[name] = (arch, b, sc)
        for s in SEEDS:
            mod = train_pgib(Xtr, phys, b, s, arch=arch); record(name, pgib_hi(mod, Xte).numpy(), sB2=float(torch.exp(2*mod.lsB)), seed=s)
            if phys:
                record('PGIB-HI|noEpi', pgib_hi(mod, Xte).numpy(), sB2=float(torch.exp(2*mod.lsB)), epi=False, seed=s)
                mod0 = train_pgib(Xtr, True, 0.0, s, arch=arch); record('PGIB-HI|beta=0', pgib_hi(mod0, Xte).numpy(), sB2=float(torch.exp(2*mod0.lsB)), seed=s)
    res['betas'] = betas; res['time'] = time.time()-t0; res['X_te'] = Xte
    pickle.dump(res, open(f'out/{ds}_{test_name}.pkl', 'wb')); print(ds, test_name, 'done %.0fs' % res['time'], betas, flush=True); return test_name

if __name__ == '__main__':
    os.makedirs('out', exist_ok=True); ds = sys.argv[1]; folds = [(ds, n) for n in CFG[ds]['cells']]
    if len(sys.argv) > 2 and sys.argv[2] == 'one': run_fold(folds[0])
    else:
        with Pool(4) as p: p.map(run_fold, folds, chunksize=1)
