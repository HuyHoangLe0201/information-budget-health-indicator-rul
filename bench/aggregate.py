import numpy as np, pickle, glob, json, sys, torch
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import rankdata
import bench, os
OUTD = os.environ.get('OUTDIR', 'out'); OUTJ = os.environ.get('OUTJSON', '../results/bench.json'); NOMI = os.environ.get('NOMI') == '1'
torch.set_default_dtype(torch.float64)

def ksg(x, y, k=5):
    x = np.asarray(x).reshape(len(x), -1); y = np.asarray(y).reshape(len(y), -1); n = len(x)
    r = lambda a: np.apply_along_axis(lambda c: (rankdata(c)-0.5)/len(c), 0, a); x, y = r(x), r(y)
    z = np.hstack([x, y]); d, _ = cKDTree(z).query(z, k=k+1, p=np.inf); eps = d[:, -1]-1e-12
    tx, ty = cKDTree(x), cKDTree(y)
    nx = np.array([len(tx.query_ball_point(x[i], eps[i], p=np.inf))-1 for i in range(n)]); ny = np.array([len(ty.query_ball_point(y[i], eps[i], p=np.inf))-1 for i in range(n)])
    return float(digamma(k)+digamma(n)-np.mean(digamma(nx+1)+digamma(ny+1)))

def feats(H):
    Hs, mu, var, wd = bench.win_stats(torch.tensor(np.asarray(H, float))); idx = bench.eval_idx(len(H)); return np.stack([Hs.numpy()[idx], mu.numpy()[idx]], 1)

def aggregate(ds):
    folds = [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(f'{OUTD}/{ds}_*.pkl'))]; stride = folds[0]['stride']; methods = list(folds[0]['methods'])
    rows = {}
    nseed = lambda mth: len(folds[0]['methods'][mth])
    for mth in methods:
        per_seed = []
        for si in range(nseed(mth)):
            Y, M, LO, HI, C, W = [], [], [], [], [], []; hq = []; Hs = []; fi_ = []
            for f in folds:
                x = f['methods'][mth][si]; Y.append(x['y']*stride); M.append(x['med']*stride); LO.append(x['lo']*stride); HI.append(x['hi']*stride); C.append(x['crps']*stride)
                W.append(np.mean(x['hi']-x['lo'])/(x['y'].max()-x['y'].min()+1e-9)); H = x['H']; Hs.append(H)
                hq.append((abs(np.sum(np.sign(np.diff(H))))/(len(H)-1), abs(np.corrcoef(H, np.arange(len(H)))[0, 1])))
            y, med, lo, hi, cr = map(np.concatenate, (Y, M, LO, HI, C)); fin = np.array([h[-1] for h in Hs]); ini = np.array([h[0] for h in Hs])
            Tp = np.concatenate([x_['methods'][mth][si]['y']*stride for x_ in folds]); Hf = np.concatenate([feats(h) for h in Hs])
            per_seed.append(dict(rmse=float(np.sqrt(np.mean((med-y)**2))), mae=float(np.mean(np.abs(med-y))), picp=float(100*np.mean((y >= lo) & (y <= hi))), pinaw=float(np.mean(W)), crps=float(np.mean(cr)),
                                 mon=float(np.mean([a for a, b in hq])), tre=float(np.mean([b for a, b in hq])), pro=float(np.exp(-np.std(fin)/max(np.mean(np.abs(fin-ini)), 1e-9))), mi=(0.0 if NOMI else ksg(Hf, Tp))))
        rows[mth] = {k: (float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed]))) for k in per_seed[0]}
    # raw-sensor reference MI
    Xf, Tp = [], []
    for f in folds:
        Xs = f['X_te']; m = len(Xs); Hs_ = [bench.win_stats(torch.tensor(Xs[:, j].copy())) for j in range(Xs.shape[1])]; idx = bench.eval_idx(m)
        Xf.append(np.stack([h[0].numpy()[idx] for h in Hs_]+[h[1].numpy()[idx] for h in Hs_], 1)); Tp.append(f['methods']['PCA-HI'][0]['y']*stride)
    Iraw = 0.0 if NOMI else ksg(np.concatenate(Xf), np.concatenate(Tp)); percell = {}
    for f in folds:
        percell[f['test']] = {mth: float(np.sqrt(np.mean([np.mean(((x['med']-x['y'])*stride)**2) for x in f['methods'][mth]]))) for mth in methods}
    return dict(rows=rows, Iraw=Iraw, percell=percell, stride=stride, m={f['test']: f['m'] for f in folds}, cfg={f['test']: {k: v[:2] for k, v in f['betas'].items()} for f in folds})

if __name__ == '__main__':
    out = {ds: aggregate(ds) for ds in sys.argv[1:]}; json.dump(out, open(OUTJ, 'w'), indent=1)
    for ds, a in out.items():
        print(ds, 'I(T;X12) raw =', round(a['Iraw'], 3), 'cfg', a['cfg'])
        for mth, r in a['rows'].items(): print('  %-22s' % mth, ' '.join('%s %.2f±%.2f' % (k, v[0], v[1]) for k, v in r.items()))
