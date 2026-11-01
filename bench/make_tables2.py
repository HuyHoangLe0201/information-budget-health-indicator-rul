import json, os, numpy as np
R = {}
for f in ['bench3_a', 'bench2_calce', 'bench2_femto', 'bench3_calce']:
    p = f'../results/{f}.json'
    if os.path.exists(p):
        for ds, a in json.load(open(p)).items(): R[ds] = a   # later files override (bench3_* include KF rows)
AGE = json.load(open('../results/age_only.json'))
order = [('Age-only', 'Age-only (no sensors)'), ('PCA-HI', 'PCA-HI'), ('Mon-opt HI', 'Mon-optimised HI'), ('beta-VAE-HI', '$\\beta$-VAE-HI'), ('VIB-HI (no physics)', 'VIB-HI (no physics)'), ('PGIB-HI (ens)', 'PGIB-HI (ens.)'), ('PGIB-KF-HI (ens)', 'PGIB-KF-HI (ens.)')]
names = {'nasa': 'NASA PCoE (4 cells)', 'calce': 'CALCE CS2 (4 cells)', 'calce6': 'CALCE CS2 (6 cells)', 'femto1': 'FEMTO cond. 1 (7 bearings)', 'femto2': 'FEMTO cond. 2 (7 bearings)', 'femto3': 'FEMTO cond. 3 (3 bearings)', 'ims': 'NASA IMS (4 bearings)'}
def f(v, d=1): return f"{v[0]:.{d}f}" + (f"$\\pm${v[1]:.{d}f}" if v[1] > 1e-9 else "")
def table(dss, fn, cap_unit):
    t = open(f'../results/{fn}.tex', 'w')
    for k, ds in enumerate(dss):
        if ds not in R: continue
        rows = R[ds]['rows']; best = min(rows[m]['rmse'][0] for m, _ in order if m in rows)
        t.write(f"\\begin{{tabular}}{{l c c c c c}}\n\\hline\n\\multicolumn{{6}}{{l}}{{{names[ds]}}}\\\\\n Method & RMSE & MAE & PICP [\\%] & PINAW & CRPS \\\\\n\\hline\n")
        for m, lab in order:
            if m not in rows: continue
            r = rows[m]; rm = f(r['rmse']); rm = f"\\textbf{{{rm}}}" if abs(r['rmse'][0]-best) < 1e-9 else rm
            t.write(f"{lab} & {rm} & {f(r['mae'])} & {f(r['picp'],0)} & {f(r['pinaw'],2)} & {f(r['crps'])} \\\\\n")
        t.write("\\hline\n\\end{tabular}\n" + ("\\\\[4pt]\n" if k < len(dss)-1 else ""))
    t.close()
table(['nasa', 'calce', 'calce6', 'femto1', 'femto2'], 'tab2_dev', 'cycles'); table(['femto3', 'ims'], 'tab2_conf', 'snapshots')
# fusion effect table: RMSE with / without fusion for PCA and PGIB
short = {'nasa': 'NASA', 'calce': 'CALCE-4', 'calce6': 'CALCE-6', 'femto1': 'FEMTO-1', 'femto2': 'FEMTO-2', 'femto3': 'FEMTO-3', 'ims': 'IMS'}
fl = ['PCA-HI', 'Mon-opt HI', 'beta-VAE-HI', 'VIB-HI (no physics)', 'PGIB-HI (ens)', 'PGIB-KF-HI (ens)']; dss = [d for d in ['nasa', 'calce', 'calce6', 'femto1', 'femto2', 'femto3', 'ims'] if d in R]
t = open('../results/tab2_fusion.tex', 'w'); t.write("\\begin{tabular}{l " + "c "*len(dss) + "}\n\\hline\nHI & " + " & ".join(short[d] for d in dss).replace('&', '&') + " \\\\\n\\hline\n")
for m in fl:
    cells = []
    for d in dss:
        rows = R[d]['rows']
        if m in rows and m+'|nofuse' in rows: cells.append(f"{rows[m]['rmse'][0]:.0f} / {rows[m+'|nofuse']['rmse'][0]:.0f}")
        else: cells.append('--')
    t.write(f"{dict(order).get(m, m)} & " + " & ".join(cells) + " \\\\\n")
t.write("\\hline\n\\end{tabular}\n"); t.close(); print(open('../results/tab2_fusion.tex').read()[:900])
