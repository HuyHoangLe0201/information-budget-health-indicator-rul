"""Pre-specified inclusion screen for CALCE CS2 cells (same protocol as CS2_35-38):
>=400 valid cycles; initial capacity in [1.0,1.25] Ah; median CC charge current in [0.5,0.6] A; median discharge current in [1.0,1.2] A;
5->15-pt rolling-median capacity reaches 80% of initial."""
import numpy as np, pandas as pd, glob, json
rows = {}
for f in sorted(glob.glob('data/calce_*.npz')):
    n = f.split('calce_')[1][:-4]; d = np.load(f); cap = d['cap']; prot = d['prot'] if 'prot' in d.files else np.zeros((0, 3))
    if len(cap) < 5: rows[n] = dict(n=len(cap), fail=['no valid cycles']); continue
    sm = pd.Series(cap).rolling(15, center=True, min_periods=1).median().values; c0 = float(np.median(cap[:5]))
    ich, idis = (float(np.median(prot[:, 0])), float(np.median(prot[:, 1]))) if len(prot) else (np.nan, np.nan)
    chk = {'>=400 cycles': len(cap) >= 400, 'cap0 in [1.0,1.25]': 1.0 <= c0 <= 1.25, 'charge ~0.55A': 0.5 <= ich <= 0.6, 'discharge ~1.1A': 1.0 <= idis <= 1.2, 'reaches 80%': bool((sm <= 0.8*c0).any())}
    rows[n] = dict(n=len(cap), cap0=round(c0, 3), Ich=round(ich, 2), Idis=round(idis, 2), fail=[k for k, v in chk.items() if not v])
for n, r in rows.items(): print(n, 'PASS' if not r['fail'] else 'fails: ' + ', '.join(r['fail']), {k: v for k, v in r.items() if k != 'fail'})
json.dump(rows, open('../results/calce_screen.json', 'w'), indent=1); print('included:', [n for n, r in rows.items() if not r['fail']])
