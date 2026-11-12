"""Stylised maintenance-decision evaluation on the official C-MAPSS test engines (one replacement decision per engine at its last observed cycle).
Risk-matched policy: the replacement is scheduled at median_i + m, where the margin m (the same for all engines of a method) is chosen so that a target fraction alpha of the
test engines fail before the scheduled time (oracle calibration on the test engines, identical for all methods); the unused life is max(true RUL - scheduled time, 0), averaged over engines.
alpha = 5% and 10%.  A comparison at a fixed predictive quantile is also stored (realised failure rates then differ between methods because of miscalibration).
Reads out2/cmapss_*_v5.pkl and *_v6.pkl.   Run from bench/ -> ../results/decision_eval.json, ../results/tab_decision.tex"""
import pickle, json, numpy as np
M = [('Age-only', 'v5', 'Age law'), ('PCA-HI', 'v5', 'PCA-HI'), ('Mon-opt HI', 'v5', 'Geo-opt HI'), ('Ridge-HI (supervised)', 'v5', 'Ridge HI'), ('PGIB-HI (ens10)', 'v6', 'PGIB, integrating'), ('PGIB-KF-HI (ens10)', 'v6', 'PGIB-KF')]
out = {}
for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    r5 = pickle.load(open(f'out2/cmapss_{fd}_v5.pkl', 'rb')); r6 = pickle.load(open(f'out2/cmapss_{fd}_v6.pkl', 'rb')); y = np.asarray(r5['y'], float); out[fd] = {}
    for k, ver, nm in M:
        v = (r5 if ver == 'v5' else r6)['methods'][k]; med = np.mean([m['med'] for m in v], 0); lo = np.mean([m['lo'] for m in v], 0); e = {}
        for a in (0.05, 0.10):
            tau = med+np.quantile(y-med, a); e[f'unused_{int(a*100)}'] = float(np.mean(np.maximum(y-tau, 0))); e[f'fail_{int(a*100)}'] = float((y < tau).mean())
        e['fail_q025'] = float((y < lo).mean()); e['unused_q025'] = float(np.mean(np.where(y < lo, 0, y-lo))); out[fd][nm] = e
json.dump(out, open('../results/decision_eval.json', 'w'), indent=1)
names = [m[2] for m in M]
with open('../results/tab_decision.tex', 'w') as f:
    f.write("\\begin{tabular}{l c c c c c c c c}\n\\hline\n& \\multicolumn{4}{c}{5\\% of engines fail} & \\multicolumn{4}{c}{10\\% of engines fail} \\\\\nMethod & " + " & ".join(out) + " & " + " & ".join(out) + " \\\\\n\\hline\n")
    for nm in names:
        f.write(nm + " & " + " & ".join(f"{out[fd][nm]['unused_5']:.0f}" for fd in out) + " & " + " & ".join(f"{out[fd][nm]['unused_10']:.0f}" for fd in out) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(open('../results/tab_decision.tex').read())
for fd in out: print(fd, 'fail at 2.5% quantile', {k: round(100*v['fail_q025']) for k, v in out[fd].items()})
