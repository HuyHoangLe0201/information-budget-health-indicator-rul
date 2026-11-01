"""Exploratory: adds PGIB-KF-HI (observation-corrected level) to the v2 small-data runs.  Reuses the v2 calibration machinery; inner leave-one-training-unit-out
selection of beta in {0.03,0.3,3}; 3 seeds + ensemble.  Existing v2 results are copied unchanged into out3/."""
import numpy as np, torch, torch.nn as nn, pickle, sys, os, time
import bench as B, bench2 as B2
import run_cmapss3 as R
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
def train_kf(Xtr, beta, seed, epochs):
    torch.manual_seed(seed); mod = R.PGIBK(Xtr[0].shape[1], True); opt = torch.optim.AdamW(mod.parameters(), lr=1e-2, weight_decay=1e-3); Xt = [torch.tensor(x) for x in Xtr]
    for ep in range(epochs):
        for X in Xt:
            opt.zero_grad(); H, u, sd = mod.path(X, True); loss = B.nll_cell(H, sB2=torch.exp(2*mod.lsB)) + beta*mod.kl(H, u, sd) + 10*(H[-1]-1)**2; loss.backward(); nn.utils.clip_grad_norm_(mod.parameters(), 5.0); opt.step()
    return mod
def run(args):
    ds, test_name = args; t0 = time.time(); res = pickle.load(open(f'out2/{ds}_{test_name}.pkl', 'rb')); data = B.load(ds); names = list(data); trn = [n for n in names if n != test_name]
    Xraw = [data[n]['X'] for n in trn]; lives = [len(data[n]['X'])-1 for n in trn]; Z = np.concatenate(Xraw); mu_, sd_ = Z.mean(0), Z.std(0)+1e-9; Xtr = [(x-mu_)/sd_ for x in Xraw]; Xte = (data[test_name]['X']-mu_)/sd_
    grid = B2.make_grid(max(lives)); inn = B2.inner_set(len(Xtr)); ap_all = B2.age_par(lives)
    def ipaths(Hs, s2s): return [dict(H=h, sB2=s2, age=B2.age_par([lives[j] for j in range(len(lives)) if j != i])) for h, s2, i in zip(Hs, s2s, inn)]
    def record(name, H, sB2, params, seed=0):
        med, lo, hi, crps, y = B2.predict2(H, sB2, params, ap_all, grid); res['methods'].setdefault(name, []).append(dict(seed=seed, H=np.asarray(H, float), med=med, lo=lo, hi=hi, crps=crps, y=y))
    sc, store = {}, {}
    for b in (0.03, 0.3, 3.0):
        v, ps = [], []
        for i in inn:
            mod = train_kf([Xtr[j] for j in range(len(Xtr)) if j != i], b, 100, 120)
            with torch.no_grad(): H = B.pgib_hi(mod, Xtr[i]); s2 = float(torch.exp(2*mod.lsB)); v.append(B.nll_cell(H, sB2=s2).item())
            ps.append((H.numpy(), s2))
        sc[b] = float(np.nanmedian(v)); store[b] = ps
    b = min(sc, key=sc.get); cal = B2.calibrate(ipaths([p[0] for p in store[b]], [p[1] for p in store[b]]), grid); res['cal']['PGIB-KF-HI'] = cal; res['betas']['PGIB-KF-HI'] = ('kf', b, {str(k): v for k, v in sc.items()}); Hs_, s2s_ = [], []
    for s in B.SEEDS:
        mod = train_kf(Xtr, b, s, 200); H = B.pgib_hi(mod, Xte).numpy(); s2 = float(torch.exp(2*mod.lsB)); Hs_.append(H); s2s_.append(s2); record('PGIB-KF-HI', H, s2, cal['fuse'], s); record('PGIB-KF-HI|nofuse', H, s2, cal['nofuse'], s)
    Hm = np.mean(Hs_, 0); sm = float(np.mean(s2s_)); record('PGIB-KF-HI (ens)', Hm, sm, cal['fuse']); record('PGIB-KF-HI (ens)|nofuse', Hm, sm, cal['nofuse'])
    os.makedirs('out3', exist_ok=True); pickle.dump(res, open(f'out3/{ds}_{test_name}.pkl', 'wb')); print(ds, test_name, 'done %.0fs' % (time.time()-t0), 'beta', b, cal['fuse'], flush=True)
if __name__ == '__main__':
    from multiprocessing import Pool
    ds = sys.argv[1]; folds = [(ds, n) for n in B.CFG[ds]['cells'] if os.path.exists(f'out2/{ds}_{n}.pkl') and not os.path.exists(f'out3/{ds}_{n}.pkl')]
    with Pool(2) as p: p.map(run, folds, chunksize=1)
