import json, pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
B = json.load(open('../results/bench.json')); order = ['PCA-HI', 'Mon-opt HI', 'beta-VAE-HI', 'VIB-HI (no physics)', 'PGIB-HI']
lab = {'PCA-HI': 'PCA-HI', 'Mon-opt HI': 'Mon-optimised HI', 'beta-VAE-HI': '$\\beta$-VAE-HI', 'VIB-HI (no physics)': 'VIB-HI (no physics prior)', 'PGIB-HI': 'PGIB-HI'}
def f(v, d=1): return f"{v[0]:.{d}f}" + (f"$\\pm${v[1]:.{d}f}" if v[1] > 1e-9 else "")
AGE = json.load(open('../results/age_only.json'))
tab = open('../results/tab_bench.tex', 'w'); abl = open('../results/tab_abl.tex', 'w'); pc = open('../results/tab_percell.tex', 'w'); geo = {}
DSL = [('nasa', 'NASA PCoE'), ('calce', 'CALCE CS2 (4 cells, 1C/1C)'), ('calce6', 'CALCE CS2 (6 cells, incl. 0.5C-discharge CS2\\_33/34)'), ('femto1', 'FEMTO bearings, condition 1 (7 units)'), ('femto2', 'FEMTO bearings, condition 2 (7 units)')]
for ds, nm in DSL:
    a = B[ds]; R = a['rows']
    tab.write(f"\\begin{{tabular}}{{l c c c c c c c c}}\n\\hline\n\\multicolumn{{9}}{{l}}{{{nm} (leave-one-unit-out)}}\\\\\n Method & Mon & Tre & Pro & RMSE & MAE & PICP [\\%] & PINAW & CRPS \\\\\n\\hline\n")
    tab.write(f"Age-only reference (no sensors) & -- & -- & -- & {AGE[ds]['rmse']:.1f} & {AGE[ds]['mae']:.1f} & -- & -- & -- \\\\\n")
    for m in order:
        r = R[m]; tab.write(f"{lab[m]} & {f(r['mon'],2)} & {f(r['tre'],2)} & {f(r['pro'],2)} & {f(r['rmse'])} & {f(r['mae'])} & {f(r['picp'],0)} & {f(r['pinaw'],2)} & {f(r['crps'])} \\\\\n")
    tab.write("\\hline\n\\end{tabular}\n%%SPLIT%%\n")
    geo[ds] = (spearmanr([R[m]['mon'][0]+R[m]['tre'][0]+R[m]['pro'][0] for m in order], [R[m]['rmse'][0] for m in order])[0])
    abl.write(f"\\begin{{tabular}}{{l c c c c}}\n\\hline\n\\multicolumn{{5}}{{l}}{{{nm}}}\\\\\n Variant & RMSE & MAE & PICP [\\%] & PINAW \\\\\n\\hline\n")
    for m, l in [('PGIB-HI', 'Full PGIB-HI'), ('PGIB-HI|beta=0', 'w/o bottleneck ($\\beta=0$)'), ('VIB-HI (no physics)', 'w/o physics prior (VIB-HI)'), ('PGIB-HI|noEpi', 'w/o epistemic drift variance')]:
        r = R[m]; abl.write(f"{l} & {f(r['rmse'])} & {f(r['mae'])} & {f(r['picp'],0)} & {f(r['pinaw'],2)} \\\\\n")
    abl.write("\\hline\n\\end{tabular}\n%%SPLIT%%\n")
    pc.write(f"\\begin{{tabular}}{{l c c c c c}}\n\\hline\n\\multicolumn{{6}}{{l}}{{{nm}: RMSE per held-out cell}}\\\\\n Cell & PCA-HI & Mon-opt. & $\\beta$-VAE & VIB-HI & PGIB-HI \\\\\n\\hline\n")
    for c, d in a['percell'].items(): pc.write(f"{c.replace('_','\\_')} & " + " & ".join(f"{d[m]:.1f}" for m in order) + " \\\\\n")
    pc.write("\\hline\n\\end{tabular}\n%%SPLIT%%\n")
tab.close(); abl.close(); pc.close()
for fn in ['tab_bench', 'tab_abl', 'tab_percell']:
    parts = open(f'../results/{fn}.tex').read().split('%%SPLIT%%\n')
    for k, nm in enumerate([d for d, _ in DSL]): open(f'../results/{fn}_{nm}.tex', 'w').write(parts[k])
json.dump(dict(spearman_geo_vs_rmse=geo), open('../results/geo_corr.json', 'w')); print('Spearman(geometric score, RMSE):', geo)
# figure: RUL trajectories
fig, ax = plt.subplots(3, 2, figsize=(7.2, 7.6), sharex='row')
for i, (ds, cell) in enumerate([('nasa', 'B0005'), ('calce', 'CS2_35'), ('femto1', 'Bearing1_1')]):
    r = pickle.load(open(f'out/{ds}_{cell}.pkl', 'rb')); s = r['stride']
    for j, m in enumerate(['PCA-HI', 'PGIB-HI']):
        x = r['methods'][m][0]; k = np.arange(len(x['y']))*s
        ax[i, j].fill_between(k, x['lo']*s, x['hi']*s, color='0.8', label='95% interval'); ax[i, j].plot(k, x['med']*s, 'k--', label='median prediction'); ax[i, j].plot(k, x['y']*s, 'k-', label='true RUL')
        ax[i, j].set_title(f"({'abcdef'[2*i+j]}) {cell}, {m}", loc='left', fontsize=9); ax[i, j].set_ylim(0, min(max(x['hi']*s)*1.05, 4*max(x['y']*s)))
        ax[i, j].set_xlabel('Steps since inspection window start' if False else 'Position in evaluation window [cycles / snapshots]'); ax[i, j].set_ylabel('RUL [cycles / snapshots]')
ax[0, 0].legend(fontsize=7, loc='upper right'); plt.tight_layout(); plt.savefig('../figs/fig_bench.pdf'); print('fig ok')
