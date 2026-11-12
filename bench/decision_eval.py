"""Stylised maintenance-decision evaluation on the official C-MAPSS test engines (one replacement decision per engine at its last observed cycle).
Policy: the replacement is scheduled at median_i + m, where the margin m (common to all engines of a method) is the alpha-quantile of the residuals (true minus predicted RUL), alpha = 5% or 10%,
so that about this fraction of engines fails.  The margin is CROSS-FITTED: five-fold over engines, m estimated on the four training folds and applied to the held-out fold (20 random repetitions);
unused life = max(true RUL - scheduled time, 0).  95% intervals: bootstrap over engines (400 replicates, one cross-fit each).  Also stored: the failure fraction at the predictive 2.5% quantile.
Reads out2/cmapss_*_v5.pkl and *_v6.pkl.   Run from bench/ -> ../results/decision_eval.json, ../results/tab_decision.tex"""
import pickle, json, numpy as np
M = [('Age-only', 'v5', 'Age law'), ('PCA-HI', 'v5', 'PCA-HI'), ('Mon-opt HI', 'v5', 'Geo-opt HI'), ('Ridge-HI (supervised)', 'v5', 'Ridge HI'), ('PGIB-HI (ens10)', 'v6', 'PGIB, integrating'), ('PGIB-KF-HI (ens10)', 'v6', 'PGIB-KF')]
rng = np.random.default_rng(0)
def crossfit(y, med, a, folds):
    n = len(y); idx = rng.permutation(n); parts = np.array_split(idx, folds); un = np.zeros(n); fl = np.zeros(n)
    for p in parts:
        tr = np.setdiff1d(idx, p); m = np.quantile(y[tr]-med[tr], a); tau = med[p]+m; un[p] = np.maximum(y[p]-tau, 0); fl[p] = (y[p] < tau)
    return un, fl
out = {}
for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    r5 = pickle.load(open(f'out2/cmapss_{fd}_v5.pkl', 'rb')); r6 = pickle.load(open(f'out2/cmapss_{fd}_v6.pkl', 'rb')); y = np.asarray(r5['y'], float); n = len(y); out[fd] = {}; meds = {}; los = {}
    for k, ver, nm in M:
        v = (r5 if ver == 'v5' else r6)['methods'][k]; meds[nm] = np.mean([m['med'] for m in v], 0); los[nm] = np.mean([m['lo'] for m in v], 0)
    for a in (0.05, 0.10):
        ak = int(a*100); base = {}
        for nm in meds:
            un = np.mean([crossfit(y, meds[nm], a, 5)[0].mean() for _ in range(20)]); fl = np.mean([crossfit(y, meds[nm], a, 5)[1].mean() for _ in range(20)])
            out[fd].setdefault(nm, {})[f'unused_{ak}'] = float(un); out[fd][nm][f'fail_{ak}'] = float(fl)
        bs = {nm: [] for nm in meds}
        for _ in range(400):
            ii = rng.integers(0, n, n); yb = y[ii]
            for nm in meds: bs[nm].append(crossfit(yb, meds[nm][ii], a, 5)[0].mean())
        for nm in meds:
            red = 1-np.array(bs[nm])/np.array(bs['Age law']); out[fd][nm][f'unused_{ak}_ci'] = [float(np.percentile(bs[nm], 2.5)), float(np.percentile(bs[nm], 97.5))]
            out[fd][nm][f'reduction_{ak}'] = float(1-out[fd][nm][f'unused_{ak}']/out[fd]['Age law'][f'unused_{ak}']); out[fd][nm][f'reduction_{ak}_ci'] = [float(np.percentile(red, 2.5)), float(np.percentile(red, 97.5))]
    for nm in meds: out[fd][nm]['fail_q025'] = float((y < los[nm]).mean())
json.dump(out, open('../results/decision_eval.json', 'w'), indent=1)
names = [m[2] for m in M]
with open('../results/tab_decision.tex', 'w') as f:
    f.write("\\begin{tabular}{l c c c c c c c c}\n\\hline\n& \\multicolumn{4}{c}{5\\% of engines fail} & \\multicolumn{4}{c}{10\\% of engines fail} \\\\\nMethod & " + " & ".join(out) + " & " + " & ".join(out) + " \\\\\n\\hline\n")
    for nm in names:
        f.write(nm + " & " + " & ".join(f"{out[fd][nm]['unused_5']:.0f}" for fd in out) + " & " + " & ".join(f"{out[fd][nm]['unused_10']:.0f}" for fd in out) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(open('../results/tab_decision.tex').read())
for fd in out:
    print(fd, {nm: (round(v['reduction_5']*100), [round(100*c) for c in v['reduction_5_ci']]) for nm, v in out[fd].items() if nm != 'Age law'})
