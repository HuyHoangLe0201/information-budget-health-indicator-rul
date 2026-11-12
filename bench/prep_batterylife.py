"""Per-cycle features of the lithium-ion cells of two BatteryLife data sets (HUST: 77 cells; MATR: 169 cells, 'processed' format, Zenodo record 19688272).
The zip archive is read member by member; pickles are read with a restricted unpickler (safe_pickle.py).  End of life L: HUST, first cycle at which the 5-cycle running median of the
discharge capacity is below 80% of the mean of the first 5 cycles (cells that never reach it are excluded); MATR, the life label of the source
(80% of the nominal capacity), at which the recordings end (cells recorded for fewer than L-10 cycles are excluded).
Features (8 per cycle): discharge capacity, charge capacity, charge duration, charge voltage at 0.3 and 0.8 Ah, mean discharge voltage,
discharge voltage at 0.3 and 0.8 Ah.   Output: data/<name>_feats.npz.   Run from bench/:  python3 prep_batterylife.py HUST|MATR"""
import numpy as np, os, sys, zipfile
from multiprocessing import Pool
import safe_pickle as sp
import json
NAME = sys.argv[1] if len(sys.argv) > 1 else 'HUST'; LAB = json.load(open(f'/home/user/data/batterylife/meta/Life labels/{NAME}_labels.json')); ZIP = f'/home/user/data/batterylife/{NAME}.zip'; QS = (0.3, 0.8)
NAMES = ['Qd', 'Qc', 't_charge', 'Vc_0.3', 'Vc_0.8', 'Vd_mean', 'Vd_0.3', 'Vd_0.8']
def at(q, v, qstar):
    i = np.argmax(q >= qstar) if np.any(q >= qstar) else -1
    return v[i] if i >= 0 else np.nan
def cell(member):
    with zipfile.ZipFile(ZIP) as z: d = sp.loads(z.read(member))
    out = []
    for z in d['cycle_data']:
        I = np.asarray(z['current_in_A'], float); V = np.asarray(z['voltage_in_V'], float); qc = np.asarray(z['charge_capacity_in_Ah'], float); qd = np.asarray(z['discharge_capacity_in_Ah'], float); t = np.asarray(z['time_in_s'], float)
        ch = I > 0.01; dc = I < -0.01
        if ch.sum() < 5 or dc.sum() < 5: out.append([np.nan]*8); continue
        Qd = qd[dc].max(); Vd = V[dc]; qdd = qd[dc]
        vmean = np.trapezoid(Vd, qdd)/max(qdd[-1]-qdd[0], 1e-9)
        out.append([Qd, qc[ch].max(), t[ch][-1]-t[ch][0], at(qc[ch], V[ch], QS[0]), at(qc[ch], V[ch], QS[1]), vmean, at(qdd, Vd, QS[0]), at(qdd, Vd, QS[1])])
    F = np.array(out); 
    for j in range(F.shape[1]):   # forward fill invalid values
        m = np.isnan(F[:, j]); 
        if m.all(): continue
        idx = np.where(~m, np.arange(len(F)), 0); np.maximum.accumulate(idx, out=idx); F[:, j] = F[idx, j]
    return os.path.basename(member)[:-4], F
if __name__ == '__main__':
    with zipfile.ZipFile(ZIP) as z: files = sorted(n for n in z.namelist() if n.endswith('.pkl'))
    with Pool(4) as p: res = p.map(cell, files, chunksize=2)
    ids, feats, lifes = [], [], []
    for name, F in res:
        q = F[:, 0]
        if NAME == 'MATR':   # the recordings end at the end of life of the source (80% of the nominal capacity); L is the life label of the source
            L = LAB[name+'.pkl']
            if np.isnan(F).any() or len(q) < L-10: print('excluded', name, len(q), L); continue
            L = min(L, len(q))
        else:
            ref = np.mean(q[:5]); med = np.array([np.median(q[max(0, i-4):i+1]) for i in range(len(q))]); below = np.where(med < 0.8*ref)[0]
            if len(below) == 0 or np.isnan(F).any(): print('excluded', name, len(q), np.isnan(F).any()); continue
            L = int(below[0])+1
        ids.append(name); feats.append(F[:L].astype(np.float32)); lifes.append(L)
    print(len(ids), 'cells; life min/median/max', min(lifes), int(np.median(lifes)), max(lifes))
    np.savez_compressed(f'data/{NAME.lower()}_feats.npz', ids=np.array(ids), L=np.array(lifes), names=np.array(NAMES), **{f'F{i}': f for i, f in enumerate(feats)})
