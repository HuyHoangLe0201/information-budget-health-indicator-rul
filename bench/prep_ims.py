"""NASA IMS bearings (confirmation set, never used during development). Failed bearings: Test1-B3 (ch5), Test1-B4 (ch7), Test2-B1 (ch1), Test3-B3 (ch3).
One accelerometer channel per bearing, 4 features per 1-s snapshot (20 kHz): log RMS, kurtosis, crest factor, spectral entropy. Failure = end of the run."""
import numpy as np, pandas as pd, glob, os, subprocess
from scipy.stats import kurtosis
from multiprocessing import Pool
R = '/home/user/data/ims_raw/ims7'; OUT = '/home/user/paper/bench/data'
def feats(x):
    rms = np.sqrt(np.mean(x**2)); sp = np.abs(np.fft.rfft(x - x.mean()))**2; p = sp/sp.sum(); ent = -np.sum(p*np.log(p+1e-20))/np.log(len(p))
    return [np.log(rms+1e-9), kurtosis(x, fisher=False), np.max(np.abs(x))/(rms+1e-9), ent]
def one(args):
    f, cols = args; d = pd.read_csv(f, sep='\t', header=None).values.astype(float); return [feats(d[:, c]) for c in cols]
if __name__ == '__main__':
    plan = [('1st_test', 'x1', {'IMS_T1B3': 4, 'IMS_T1B4': 6}), ('2nd_test', 'x2', {'IMS_T2B1': 0}), ('3rd_test', 'x3', {'IMS_T3B3': 2})]
    for rar, d, units in plan:
        if all(os.path.exists(f'{OUT}/ims_{u}.npz') for u in units): continue
        os.makedirs(f'{R}/{d}', exist_ok=True); subprocess.run(['unrar-free', '-x', f'{R}/{rar}.rar', f'{R}/{d}'], capture_output=True)
        fs = sorted(f for f in glob.glob(f'{R}/{d}/**/*', recursive=True) if os.path.isfile(f)); cols = list(units.values())
        with Pool(4) as p: res = np.array(p.map(one, [(f, cols) for f in fs], chunksize=20))
        for k, name in enumerate(units): np.savez(f'{OUT}/ims_{name}.npz', X=res[:, k, :]); print(name, res.shape[0], flush=True)
