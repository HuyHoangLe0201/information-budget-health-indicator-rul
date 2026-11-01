"""FEMTO/PRONOSTIA bearings: per-snapshot vibration features (2 channels x [log RMS, kurtosis, crest factor, spectral entropy]).
Each snapshot = 2560 samples at 25.6 kHz every 10 s. Failure = end of run (all 17 bearings in Learning_set + Full_Test_Set run to failure)."""
import numpy as np, pandas as pd, glob, os, sys
from scipy.stats import kurtosis
from multiprocessing import Pool
B = '/home/user/data/femto_raw/10. FEMTO Bearing'; OUT = '/home/user/paper/bench/data'; os.makedirs(OUT, exist_ok=True)
LEARN = {'Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2'}
def folder(n): return f'{B}/Training_set/Learning_set/{n}' if n in LEARN else f'{B}/Validation_Set/Full_Test_Set/{n}'
def feats(x):
    rms = np.sqrt(np.mean(x**2)); sp = np.abs(np.fft.rfft(x - x.mean()))**2; p = sp/sp.sum(); ent = -np.sum(p*np.log(p+1e-20))/np.log(len(p))
    return [np.log(rms+1e-9), kurtosis(x, fisher=False), np.max(np.abs(x))/(rms+1e-9), ent]
def job(n):
    fs = sorted(glob.glob(folder(n) + '/acc_*.csv')); X = []
    for f in fs:
        d = pd.read_csv(f, header=None)
        if d.shape[1] < 6: d = pd.read_csv(f, header=None, sep=';')
        h, v = d[4].values.astype(float), d[5].values.astype(float); X.append(feats(h) + feats(v))
    X = np.array(X); np.savez(f'{OUT}/femto_{n}.npz', X=X); return n, X.shape, int(np.isnan(X).sum())
if __name__ == '__main__':
    names = [f"Bearing{c}_{k}" for c, ks in [(3, range(1, 4))] for k in ks]
    with Pool(4) as p:
        for r in p.imap_unordered(job, names): print(r, flush=True)
