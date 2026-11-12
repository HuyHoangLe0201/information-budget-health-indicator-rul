"""LaTeX tables of the real-data checks (reads ../results/realdata_FD00x.json and cert_age_cmapss.json).  Run from bench/."""
import json, numpy as np
FD = ['FD001', 'FD002', 'FD003', 'FD004']; R = {fd: json.load(open(f'../results/realdata_{fd}.json')) for fd in FD}; C = json.load(open('../results/cert_age_cmapss.json'))
f2 = lambda x: f'{x:.2f}'
# ---- T1: cross-validated age-conditional certificates
cols = [('age only', 'Age only (control)'), ('ridge HI', 'Ridge HI'), ('PCA-HI', 'PCA-HI'), ('ridge level only', 'Level only'), ('ridge slope only', 'Slope only'), ('sensors', 'Sensors')]
with open('../results/tab_rd_cert.tex', 'w') as f:
    f.write("\\begin{tabular}{l c " + "c "*len(cols) + "}\n\\hline\nSubset & $\\bar h(T\\mid t)$ & " + " & ".join(c for _, c in cols) + " \\\\\n\\hline\n")
    for fd in FD:
        r = R[fd]; c = r['cert']; ci = r['cert_ci']; f.write(f"{fd} & {c['h_cond']:.2f} & " + " & ".join(f"{c[k]:.2f} [{ci[k][0]:.2f}, {ci[k][1]:.2f}]" for k, _ in cols) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
# ---- T2: squeeze
G = ['0.0', '0.5', '0.9', '0.97', '0.99', '0.995']
with open('../results/tab_rd_squeeze.tex', 'w') as f:
    f.write("\\begin{tabular}{l l " + "c "*len(G) + "c}\n\\hline\nSubset & & " + " & ".join(f"$\\gamma={g}$" for g in G) + " & $\\mathrm{Mon}_\\infty$ (predicted) \\\\\n\\hline\n")
    for fd in FD:
        r = R[fd]; f.write(f"{fd} & $\\mathrm{{Mon}}$ & " + " & ".join(f2(r['mon'][g]) for g in G) + f" & {r['mon_inf_pred']:.2f} \\\\\n")
        f.write(f" & certificate & " + " & ".join(f2(r['cert'][f'squeeze {g}']) for g in G) + " & \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
# ---- T3: stage-wise static versus free direction
def summ(S):
    st = S['stages']; ang = [s['angle_opt'] for s in st]; ratio = max(np.sqrt(s['mse']['static']/s['mse']['both']) for s in st); lvl = [np.sqrt(s['mse']['level']/s['mse']['both']) for s in st]; vel = [np.sqrt(s['mse']['velocity']/s['mse']['both']) for s in st]
    return min(ang), max(ang), S['static_angle'], ratio, max(lvl), max(vel)
with open('../results/tab_rd_stage.tex', 'w') as f:
    f.write("\\begin{tabular}{l l c c c c c}\n\\hline\nSubset & stages & optimal angle [deg] & static angle [deg] & static / free & level only / free & velocity only / free \\\\\n\\hline\n")
    for fd in FD:
        for key, nm in [('stage_phi', 'life fraction'), ('stage_age', 'age')]:
            a, b, sa, rt, lv, ve = summ(R[fd][key]); f.write(f"{fd} & {nm} & {a:.0f}--{b:.0f} & {sa} & {rt:.3f} & {lv:.2f} & {ve:.2f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
# ---- T4: regime read-out
with open('../results/tab_rd_regime.tex', 'w') as f:
    f.write("\\begin{tabular}{l c c c c}\n\\hline\nSubset & regimes & regime-blind & regime-normalised & regime-specific weights \\\\\n\\hline\n")
    for fd in FD:
        r = R[fd]['regime_rmse_phi']; f.write(f"{fd} & {6 if fd in ('FD002', 'FD004') else 1} & {r['blind']:.3f} & {r['norm']:.3f} & {r['aware']:.3f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
# ---- T5: certificates on the official test engines
rows = [('Age-only', 'Age law (control)'), ('PCA-HI', 'PCA-HI'), ('Mon-opt HI', 'Geo-opt HI'), ('Ridge-HI (supervised)', 'Ridge HI'), ('beta-VAE-HI', '$\\beta$-VAE'), ('VIB-HI (no physics)', 'VIB'), ('PGIB-HI (ens10)', 'PGIB, integrating (10 seeds)'), ('PGIB-KF-HI (ens10)', 'PGIB-KF (10 seeds)'), ('PGIB-KFR-HI (ens10)', 'PGIB-KFR (10 seeds)')]
with open('../results/tab_rd_test.tex', 'w') as f:
    f.write("\\begin{tabular}{l " + "c "*4 + "}\n\\hline\nMethod & FD001 & FD002 & FD003$^\\dagger$ & FD004 \\\\\n\\hline\n")
    f.write("$\\bar h(T\\mid t)$ & " + " & ".join(f"{C[fd]['h_cond']:.2f}" for fd in FD) + " \\\\\n\\hline\n")
    for k, nm in rows: f.write(f"{nm} & " + " & ".join(f"{C[fd]['cert'][k]:.2f} [{C[fd]['ci'][k][0]:.2f}, {C[fd]['ci'][k][1]:.2f}]" for fd in FD) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(open('../results/tab_rd_stage.tex').read()); print(open('../results/tab_rd_regime.tex').read())
