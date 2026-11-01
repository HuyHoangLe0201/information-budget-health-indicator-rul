"""Per-cycle feature extraction for NASA PCoE (B0005/6/7/18) and CALCE CS2_35-38.
Capacity is the EOL label only; it is NOT an input feature (no discharge-time/capacity features)."""
import numpy as np, scipy.io as sio, glob, sys, os, pandas as pd
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
OUT = '/home/user/paper/bench/data'; os.makedirs(OUT, exist_ok=True)

def ic_feat(t, v, i, vlo=3.75, vhi=4.12):
    """Incremental-capacity peak (height, voltage) from the CC charge segment."""
    m = i > 0.5*np.nanmax(i); idx = np.where(m)[0]
    if len(idx) < 6: return np.nan, np.nan
    t, v, i = t[idx], v[idx], i[idx]; q = np.concatenate([[0], np.cumsum(0.5*(i[1:]+i[:-1])*np.diff(t))])/3600.0
    vv = np.maximum.accumulate(v); keep = np.concatenate([[True], np.diff(vv) > 1e-6]); vv, q = vv[keep], q[keep]
    if len(vv) < 6 or vv[-1] < vhi-0.05: return np.nan, np.nan
    grid = np.linspace(vlo, vhi, 120); qg = np.interp(grid, vv, q); qg = gaussian_filter1d(qg, 3)
    dq = np.gradient(qg, grid); k = np.argmax(dq); return float(dq[k]), float(grid[k])

def nasa_cell(name, folder):
    m = sio.loadmat(f'/home/user/data/nasa_raw/5. Battery Data Set/{folder}/{name}.mat', struct_as_record=False, squeeze_me=True)
    cyc = [c for c in m[name].cycle if c.type in ('charge', 'discharge')]
    X, cap = [], []; last_ch = None
    for c in cyc:
        d = c.data
        if c.type == 'charge': last_ch = d; continue
        if last_ch is None: continue
        t, v, i = last_ch.Time, last_ch.Voltage_measured, last_ch.Current_measured
        cc = i > 1.4; cc_time = t[cc][-1]-t[cc][0] if cc.sum() > 2 else np.nan
        cv_time = t[-1] - (t[cc][-1] if cc.sum() > 2 else t[0])
        ph, pv = ic_feat(t, v, i)
        td, vd = d.Time, d.Voltage_measured
        X.append([cc_time, cv_time, ph, pv, float(np.mean(d.Temperature_measured)), float(np.interp(600, td, vd))])
        cap.append(float(d.Capacity)); last_ch = None
    return np.array(X, float), np.array(cap, float)

def calce_cell(cell):
    files = sorted(sorted(glob.glob(f'/home/user/data/calce_raw/{cell}/**/*.xlsx', recursive=True)))
    dfs = []
    for fi, f in enumerate(files):
        x = pd.ExcelFile(f)
        for sh in x.sheet_names:
            if sh.startswith('Channel'):
                d = x.parse(sh, usecols=['Test_Time(s)', 'Date_Time', 'Cycle_Index', 'Current(A)', 'Voltage(V)']); d['fid'] = fi; dfs.append(d)
    dfs.sort(key=lambda d: pd.to_datetime(d['Date_Time'].iloc[0]))
    for k, d in enumerate(dfs): d['fid'] = k
    df = pd.concat(dfs).reset_index(drop=True)
    X, cap, prot = [], [], []
    for cy, g in df.groupby(['fid', 'Cycle_Index'], sort=True):
        t = (pd.to_datetime(g['Date_Time']) - pd.to_datetime(g['Date_Time']).iloc[0]).dt.total_seconds().values
        v, i = g['Voltage(V)'].values, g['Current(A)'].values
        ch, dc = i > 0.02, i < -0.02
        if np.nanmax(v) < 4.15 or np.nanmin(v[dc] if dc.any() else v) > 2.8: continue  # incomplete (split) cycle
        if ch.sum() < 10 or dc.sum() < 10: continue
        td = t[dc]; qd = np.trapezoid(-i[dc], td)/3600.0 if len(td) > 1 else np.nan
        tc, vc, ic = t[ch], v[ch], i[ch]
        cc = ic > 0.8*np.max(ic); cc_time = tc[cc][-1]-tc[cc][0] if cc.sum() > 2 else np.nan
        cv_time = tc[-1] - (tc[cc][-1] if cc.sum() > 2 else tc[0])
        ph, pv = ic_feat(tc, vc, ic, 3.7, 4.12)
        vd = v[dc]; dv_on = float(v[ch][-1] - vd[0]) if len(vd) else np.nan
        if 0.3 < qd < 1.5: prot.append([float(np.max(ic)), float(np.median(-i[dc])), float(np.max(v))]); X.append([cc_time, cv_time, ph, pv, dv_on, float(np.interp(1200, td-td[0], vd))]); cap.append(qd)
    return np.array(X, float), np.array(cap, float), np.array(prot, float)

def job(a):
    try: return job_(a)
    except Exception as e: return ('FAILED', a[1], repr(e)[:150])

def job_(a):
    kind, name, folder = a
    if kind == 'nasa': X, cap = nasa_cell(name, folder); prot = np.zeros((0, 3))
    else: X, cap, prot = calce_cell(name)
    np.savez(f'{OUT}/{kind}_{name}.npz', X=X, cap=cap, prot=prot); return kind, name, X.shape, np.isnan(X).mean(0).round(3).tolist(), float(cap[0]), float(cap.min())

if __name__ == '__main__':
    jobs = [('calce', f'CS2_{k}', '') for k in map(int, sys.argv[1:])]
    with Pool(8) as p:
        for r in p.imap_unordered(job, jobs): print(r, flush=True)
