"""Ablation table of the integrating PGIB on C-MAPSS FD003 (three seeds, with age fusion): reads ../results/cmapss_FD003.json, writes ../results/tab_abl_fd003.tex.  Run from bench/."""
import json
d = json.load(open('../results/cmapss_FD003.json'))['rows']
rows = [('Full PGIB', 'PGIB-HI'), ('Without bottleneck ($\\beta=0$)', 'PGIB-HI|beta=0'), ('Without physics prior (VIB)', 'VIB-HI (no physics)'), ('Without epistemic drift variance', 'PGIB-HI|noEpi')]
def pm(v, nd=1): return f'{v[0]:.{nd}f}$\\pm${v[1]:.{nd}f}'
with open('../results/tab_abl_fd003.tex', 'w') as f:
    f.write("\\begin{tabular}{l c c c c c}\n\\hline\nVariant & RMSE & MAE & PICP [\\%] & PINAW & CRPS \\\\\n\\hline\n")
    for nm, k in rows:
        r = d[k]; f.write(f"{nm} & {pm(r['rmse'])} & {pm(r['mae'])} & {pm(r['picp'], 0)} & {pm(r['pinaw'], 2)} & {pm(r['crps'])} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(open('../results/tab_abl_fd003.tex').read())
