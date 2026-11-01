import json, sys, os
order = [('Age-only', 'Age-only (no sensors)'), ('PCA-HI', 'PCA-HI'), ('Mon-opt HI', 'Mon-optimised HI'), ('Ridge-HI (supervised)', 'Ridge-HI (supervised)'), ('Ridge-reg-HI', 'Ridge-HI, regime-specific'), ('beta-VAE-HI', '$\\beta$-VAE-HI'),
         ('VIB-HI (no physics)', 'VIB-HI (no physics prior)'), ('PGIB-HI', 'PGIB-HI (GRU/window encoder)'), ('PGIB-HI (ens)', 'PGIB-HI, 3-seed ensemble'), ('PGIB-KF-HI', 'PGIB-KF-HI'), ('PGIB-KF-HI (ens)', 'PGIB-KF-HI, 3-seed ensemble'), ('PGIB-KFR-HI', 'PGIB-KFR-HI (regime-aware)'), ('PGIB-KFR-HI (ens)', 'PGIB-KFR-HI, 3-seed ensemble')]
def tab(fd, fn):
    d = json.load(open(f'../results/{fn}.json')); R = d['rows']; t = f"\\setlength{{\\tabcolsep}}{{4pt}}\\begin{{tabular}}{{l c c c c c c c}}\n\\hline\n Method & RMSE & RMSE seed-avg. [95\\% CI] & MAE & Score & PICP [\\%] & PINAW & CRPS \\\\\n\\hline\n"
    for k, lab in order:
        if k not in R: continue
        r = R[k]; sd = lambda x, p=1: f"{x[0]:.{p}f}" + (f"$\\pm${x[1]:.{p}f}" if x[1] > 1e-9 else "")
        t += f"{lab} & {sd(r['rmse'])} & {r['ci'][0]:.1f}--{r['ci'][1]:.1f} & {sd(r['mae'])} & {r['score'][0]:.0f} & {r['picp'][0]:.0f} & {r['pinaw'][0]:.2f} & {sd(r['crps'])} \\\\\n"
    return t + "\\hline\n\\end{tabular}\n", d
for fd, lab in [('FD001', 'FD001'), ('FD002', 'FD002'), ('FD003', 'FD003'), ('FD004', 'FD004')]:
    t, d = tab(lab, f'cmapss_{fd}_v5'); open(f'../results/tab_cmapss_{fd.lower()}.tex', 'w').write(t)
t3 = open('../results/tab_cmapss_fd003.tex').read(); d3 = {'paired_rmse_diff_pgib_ens_minus': {}}
print(t3); print(d3['paired_rmse_diff_pgib_ens_minus'])
