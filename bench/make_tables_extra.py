"""Tables for the additional analyses: RMSE versus certificate across data sets, geometric scores versus certified information, Tongji inclusion-rule sensitivity.  Run from bench/."""
import json, os, numpy as np
S = ['FD001', 'FD002', 'FD003', 'FD004', 'HUST', 'MATR', 'Tongji']
R = {s: json.load(open(f'../results/realdata_{s}.json')) for s in S}
with open('../results/tab_cmp.tex', 'w') as f:
    f.write("\\begin{tabular}{l c c c c c c c}\n\\hline\nData set & Median life & \\multicolumn{3}{c}{RMSE [cycles]} & \\multicolumn{3}{c}{Certificate [nats]} \\\\\n & [cycles] & age only & ridge HI & ratio & age only & ridge HI & gain \\\\\n\\hline\n")
    for s in S:
        r = R[s]; m = r['cert']['mse']; ra = np.sqrt(m['age only']); rr = np.sqrt(m['ridge HI']); ca = r['cert']['age only']; cr = r['cert']['ridge HI']
        f.write(f"{s} & {np.median(r['lifetimes']):.0f} & {ra:.0f} & {rr:.0f} & {rr/ra:.2f} & {ca:.2f} & {cr:.2f} & {cr-ca:.2f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
G = {s: json.load(open(f'../results/geom_vs_info_{s}.json')) for s in S if os.path.exists(f'../results/geom_vs_info_{s}.json')}
if G:
    with open('../results/tab_geom.tex', 'w') as f:
        f.write("\\begin{tabular}{l c c c c c}\n\\hline\nData set & HIs & $\\rho_{\\rm S}$ (score) & $\\rho_{\\rm S}$ (Mon) & rank of the highest-score HI & certificate: highest-score HI / best HI \\\\\n\\hline\n")
        for s in S:
            if s in G: g = G[s]; f.write(f"{s} & {g['n_hi']} & {g['spearman_score']:.2f} [{g['spearman_score_ci'][0]:.2f}, {g['spearman_score_ci'][1]:.2f}] & {g['spearman_mon']:.2f} [{g['spearman_mon_ci'][0]:.2f}, {g['spearman_mon_ci'][1]:.2f}] & {g['best_score_cert_rank']} & {g['cert_of_best_score']:.2f} / {g['best_cert']:.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
if os.path.exists('../results/realdata_TongjiAll.json'):
    A = json.load(open('../results/realdata_TongjiAll.json')); B = R['Tongji']
    def st(r):
        out = []
        for k in ('stage_phi', 'stage_age'): out.append(max(np.sqrt(x['mse']['static']/x['mse']['both']) for x in r[k]['stages']))
        return max(out)
    with open('../results/tab_sens_tongji.tex', 'w') as f:
        f.write("\\begin{tabular}{l c c}\n\\hline\n & Main (88 cells) & All cells (119 cells) \\\\\n\\hline\n")
        rows = [('Age-only control [nats]', lambda r: f"{r['cert']['age only']:.2f} [{r['cert_ci']['age only'][0]:.2f}, {r['cert_ci']['age only'][1]:.2f}]"), ('Ridge HI certificate [nats]', lambda r: f"{r['cert']['ridge HI']:.2f} [{r['cert_ci']['ridge HI'][0]:.2f}, {r['cert_ci']['ridge HI'][1]:.2f}]"),
                ('Charging features [nats]', lambda r: f"{r['cert']['charge sensors']:.2f}"), ('Monotonicity: $\\gamma=0$, $0.995$ (predicted)', lambda r: f"{r['mon']['0.0']:.2f}, {r['mon']['0.995']:.2f} ({r['mon_inf_pred']:.2f})"),
                ('Certificate after squeeze, $\\gamma=0.995$ [nats]', lambda r: f"{r['cert']['squeeze 0.995']:.2f}"), ('Worst-stage RMSE ratio, static / free', lambda r: f"{st(r):.3f}"),
                ('Protocol-blind / normalised read-out RMSE', lambda r: f"{r['regime_rmse_phi']['blind']:.3f} / {r['regime_rmse_phi']['norm']:.3f}")]
        for nm, fn in rows: f.write(f"{nm} & {fn(B)} & {fn(A)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(open('../results/tab_sens_tongji.tex').read())
print(open('../results/tab_cmp.tex').read())
# ---- end-to-end context table
E2 = json.load(open('../results/e2e_baseline.json')); CC = json.load(open('../results/cert_age_cmapss.json')); FDs = ['FD001', 'FD002', 'FD003', 'FD004']
def best_simple(fd):
    d = json.load(open(f'../results/cmapss_{fd}_v5.json' if fd == 'FD001' else f'../results/cmapss_{fd}.json'))['rows']; return min(d[k]['rmse'][0] for k in ('PCA-HI', 'Mon-opt HI', 'Ridge-HI (supervised)'))
with open('../results/tab_e2e.tex', 'w') as f:
    f.write("\\begin{tabular}{l c c c c}\n\\hline\n & " + " & ".join(FDs) + " \\\\\n\\hline\n")
    f.write("RMSE, best of PCA-HI, Geo-opt HI, ridge HI & " + " & ".join(f"{best_simple(fd):.1f}" for fd in FDs) + " \\\\\n")
    f.write("RMSE, all sensors, true-RUL target & " + " & ".join(f"{E2[fd]['unclipped']['rmse']:.1f}" for fd in FDs) + " \\\\\n")
    f.write("RMSE, all sensors, target clipped at 125 & " + " & ".join(f"{E2[fd]['clipped 125']['rmse']:.1f}" for fd in FDs) + " \\\\\n")
    f.write("Certificate [nats], best of the three HIs & " + " & ".join(f"{max(CC[fd]['cert'][k] for k in ('PCA-HI', 'Mon-opt HI', 'Ridge-HI (supervised)')):.2f}" for fd in FDs) + " \\\\\n")
    f.write("Certificate [nats], all sensors, clipped target & " + " & ".join(f"{E2[fd]['clipped 125']['cert']:.2f}" for fd in FDs) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(open('../results/tab_e2e.tex').read())
