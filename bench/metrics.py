import numpy as np, pickle, glob
def cell_metrics(r, cycles_per_step):
    out = {}
    for name, runs in r['methods'].items():
        L = []
        for x in runs:
            y, med, lo, hi, crps = x['y'], x['med'], x['lo'], x['hi'], x['crps']; k = cycles_per_step
            L.append(dict(rmse=np.sqrt(np.mean(((med-y)*k)**2)), mae=np.mean(np.abs(med-y))*k, picp=100*np.mean((y >= lo) & (y <= hi)),
                          pinaw=np.mean(hi-lo)/(y.max()-y.min()+1e-9), crps=np.mean(x['crps'])*k))
        out[name] = L
    return out
if __name__ == '__main__':
    import sys
    for f in sorted(glob.glob('out/*.pkl')):
        r = pickle.load(open(f, 'rb')); mt = cell_metrics(r, r['stride']); print(f, 'm=', r['m'])
        for n, L in mt.items(): print('  %-24s RMSE %.2f MAE %.2f PICP %.0f PINAW %.2f CRPS %.2f  (n=%d)' % (n, np.mean([l['rmse'] for l in L]), np.mean([l['mae'] for l in L]), np.mean([l['picp'] for l in L]), np.mean([l['pinaw'] for l in L]), np.mean([l['crps'] for l in L]), len(L)))
