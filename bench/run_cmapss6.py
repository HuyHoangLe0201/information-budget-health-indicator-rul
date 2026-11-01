"""P5: variance study. 10 fresh seeds (100..109) for the learned encoders on C-MAPSS.  Architecture, beta and predictor calibration (c, w_h, w_a) are FIXED to the values
selected earlier (P3/P4, stored in out2/cmapss_{FD}_v5.pkl); nothing is re-selected.  Records per-seed predictions and the 10-seed ensemble."""
import numpy as np, torch, pickle, sys, time, os
import bench as B, bench2 as B2, run_cmapss4 as R4, run_cmapss5 as R5
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
SEEDS = list(range(100, 110))
if __name__ == '__main__':
    FD = sys.argv[1]; t0 = time.time(); old = pickle.load(open(f'out2/cmapss_{FD}_v5.pkl', 'rb')); tr, te, rul = R4.load_fd(FD); ids = np.arange(len(tr)); cal_ids = [i for i in ids if (i+1) % 5 == 0]; fit_ids = [i for i in ids if (i+1) % 5 != 0]
    multi = FD in ('FD002', 'FD004'); R = 6 if multi else 1; fs = np.concatenate([tr[i][0] for i in fit_ids])
    if multi:
        from sklearn.cluster import KMeans
        fo = np.concatenate([tr[i][1] for i in fit_ids]); om, osd = fo.mean(0), fo.std(0)+1e-9; km = KMeans(6, n_init=10, random_state=0).fit((fo-om)/osd); fr = km.labels_
        mus = np.array([fs[fr == r].mean(0) for r in range(6)]); sds = np.array([fs[fr == r].std(0)+1e-9 for r in range(6)]); regime = lambda p: km.predict((p[1]-om)/osd)
    else:
        mu0, sd0 = fs.mean(0), fs.std(0)+1e-9; mus = mu0[None]; sds = sd0[None]; regime = lambda p: np.zeros(len(p[0]), int)
    def nzr(p): r = regime(p); return np.hstack([(p[0]-mus[r])/sds[r], np.eye(R)[r]])
    ds = fs.shape[1]; XfitA = [nzr(tr[i]) for i in fit_ids]; XteA = [nzr(x) for x in te]; XfitS = [x[:, :ds] for x in XfitA]; XteS = [x[:, :ds] for x in XteA]
    lives = [len(x)-1 for x in XfitA]; ap = B2.age_par(lives); grid = B2.make_grid(max(len(x[0]) for x in tr)); res = dict(methods={}, FD=FD, seeds=SEEDS)
    def rec(name, Hte, sB2, params, seed):
        r = [R4.predict_last(h, sB2, params, ap, grid, rul[i]) for i, h in enumerate(Hte)]; res['methods'].setdefault(name, []).append(dict(seed=seed, med=np.array([a[0] for a in r]), lo=np.array([a[1] for a in r]), hi=np.array([a[2] for a in r]), crps=np.array([a[3] for a in r])))
    def ens(name, Hs, s2s, params): Hm = [np.mean([Hs[k][i] for k in range(len(Hs))], 0) for i in range(len(te))]; rec(name+' (ens10)', Hm, float(np.mean(s2s)), params, -1)
    # fixed configurations from earlier selection
    cfg = {'PGIB-KF-HI': ('kf', old['sel']['PGIB-KF-HI'][1]), 'PGIB-KFR-HI': ('kfr', old['sel']['PGIB-KFR-HI'][1]), 'PGIB-HI': (old['sel']['PGIB-HI'][0], old['sel']['PGIB-HI'][1])}; print(FD, 'fixed configs', cfg, flush=True)
    for name, (arch, b) in cfg.items():
        params = old['cal'][name]['fuse']; Hs, s2s = [], []
        for s in SEEDS:
            if arch == 'kfr': mod = R5.train_kfr(XfitA, ds, R, b, s, R4.STEPS); Ht = [B.pgib_hi(mod, x).numpy() for x in XteA]
            else: mod = R4.train_steps(XfitS, True, b, s, arch); Ht = [B.pgib_hi(mod, x).numpy() for x in XteS]
            s2 = float(torch.exp(2*mod.lsB)); Hs.append(Ht); s2s.append(s2); rec(name, Ht, s2, params, s)
        ens(name, Hs, s2s, params); print(name, 'done %.0fs' % (time.time()-t0), flush=True)
    params = old['cal']['beta-VAE-HI']['fuse']
    for s in SEEDS: out, _ = R4.vae_steps(XfitS, XteS, s); rec('beta-VAE-HI', out, None, params, s)
    print('vae done %.0fs' % (time.time()-t0), flush=True)
    pickle.dump(res, open(f'out2/cmapss_{FD}_v6.pkl', 'wb')); print('done %.0fs' % (time.time()-t0))
