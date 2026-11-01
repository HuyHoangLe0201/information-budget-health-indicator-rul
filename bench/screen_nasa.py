"""Pre-specified inclusion screen for NASA cells (same protocol as B0005/6/7/18):
(1) ambient 24C; (2) >=100 discharge cycles; (3) >=90% of discharges are ~1.5-2 A constant-current (|median I| in [1.4,2.1]) and run to cutoff (min V < 3.0 V);
(4) initial capacity in [1.5,2.1] Ah; (5) 5-pt rolling-median capacity reaches 75% of initial."""
import scipy.io as sio, glob, numpy as np, pandas as pd, json
base = '/home/user/data/nasa_raw/5. Battery Data Set'; rows = {}
for f in sorted(glob.glob(base + '/x_*/B00*.mat')):
    n = f.split('/')[-1][:-4]
    if n in rows: continue
    c = sio.loadmat(f, struct_as_record=False, squeeze_me=True)[n].cycle; dc = [x for x in c if x.type == 'discharge']
    if len(dc) < 5: rows[n] = dict(n=len(dc), fail=['<5 discharges']); continue
    bad = sum(np.ndim(x.data.Capacity) != 0 for x in dc); dc = [x for x in dc if np.ndim(x.data.Capacity) == 0]
    if len(dc) < 5: rows[n] = dict(n=len(dc), fail=['no scalar capacity records']); continue
    caps = np.array([float(x.data.Capacity) for x in dc]); I = np.array([abs(np.median(x.data.Current_measured)) for x in dc]); Ve = np.array([np.min(x.data.Voltage_measured) for x in dc])
    T = float(np.median([x.ambient_temperature for x in dc])); c0 = float(np.median(caps[:5])); sm = pd.Series(caps).rolling(5, center=True, min_periods=1).median().values
    chk = {'24C': T == 24, '>=100 cycles': len(dc) >= 100, '~2A CC discharge': np.mean((I > 1.4) & (I < 2.1)) >= 0.9, 'discharged to cutoff': np.mean(Ve < 3.0) >= 0.9, 'cap0 in [1.5,2.1]': 1.5 <= c0 <= 2.1, 'reaches 75%': bool((sm <= 0.75*c0).any())}
    rows[n] = dict(n=len(dc), T=T, cap0=round(c0, 2), fail=[k for k, v in chk.items() if not v])
for n, r in rows.items(): print(n, 'PASS' if not r['fail'] else 'fails: ' + ', '.join(r['fail']), r.get('n'), r.get('T'), r.get('cap0'))
json.dump(rows, open('../results/nasa_screen.json', 'w'), indent=1)
print('included:', [n for n, r in rows.items() if not r['fail']])
