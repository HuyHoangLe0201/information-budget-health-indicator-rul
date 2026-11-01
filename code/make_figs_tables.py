import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 7.5})
d1 = json.load(open('../results/exp1.json'))
gs = list(d1['res'].keys()); R = d1['res']
ms = lambda g, k: (np.mean([x[k] for x in R[g]['runs']]), np.std([x[k] for x in R[g]['runs']]))
mon = np.array([ms(g, 'mon')[0] for g in gs]); gam = np.array([float(g) for g in gs])
# ---- Fig 1
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].plot(gam, mon, 'k-o', ms=4, label=r'Monotonicity $\mathrm{Mon}$')
ax[0].plot(gam, [R[g]['eta'] for g in gs], 'k--s', ms=4, mfc='w', label=r'Efficiency $\eta_{\rm info}$ (target $V_t$)')
ax[0].set_xlabel(r'Squeezing strength $\gamma$'); ax[0].set_ylabel('Dimensionless score'); ax[0].set_ylim(0, 1.05); ax[0].legend(loc='center left'); ax[0].set_title('(a)', loc='left')
ax[1].plot(gam, [ms(g, 'rmse_bound')[0] for g in gs], 'k-o', ms=4, label=r'Entropy-power bound $\sqrt{\mathcal{B}}$ (Thm. 1)')
ax[1].plot(gam, [ms(g, 'rmse_postvar')[0] for g in gs], 'k--^', ms=4, mfc='w', label='Bayes-optimal RMSE (posterior)')
ax[1].set_xlabel(r'Squeezing strength $\gamma$'); ax[1].set_ylabel('RUL RMSE [cycles]'); ax[1].legend(loc='upper left'); ax[1].set_title('(b)', loc='left')
plt.tight_layout(); plt.savefig('../figs/fig_paradox.pdf'); plt.close()
# ---- Fig 2 (Exp 2b)
d2b = json.load(open('../results/exp2b.json')); rb = d2b['rows']; eta = np.array([r['eta'] for r in rb])
fig, ax = plt.subplots(figsize=(3.6, 3.0))
ax.plot(eta, [r['rmse_bound'] for r in rb], 'k-o', ms=4, label='Thm. 3 bound (pointwise)')
ax.plot(eta, [r['rmse_bound_jensen'] for r in rb], 'k-.s', ms=4, mfc='w', label='Thm. 3 bound (Jensen form)')
ax.plot(eta, [r['rmse_bayes_post'] for r in rb], 'k--^', ms=4, mfc='w', label='Bayes-optimal RMSE')
ax.axhline(np.sqrt(rb[0]['ale']), color='gray', ls=':', label='Aleatoric floor')
ax.set_xlabel(r'$\eta_m = I(m;H)/I(m;X)$'); ax.set_ylabel('RUL RMSE [cycles]'); ax.legend(loc='upper right', fontsize=6.5); plt.tight_layout(); plt.savefig('../figs/fig_bound.pdf'); plt.close()
# ---- tables
with open('../results/tables.tex', 'w') as f:
    f.write("\\begin{tabular}{c c c c c c c c}\n\\hline\n$\\gamma$ & $\\mathrm{Mon}$ & $\\mathcal{I}_{\\rm state}$ & $\\mathcal{I}_{\\rm kin}$ & $\\eta_{\\rm PIE}$ & $\\sqrt{\\mathcal{B}_{\\rm info}}$ & Bayes RMSE & Empirical RMSE \\\\\n & & [nats] & [nats] & & [cycles] & [cycles] & [cycles] \\\\\n\\hline\n")
    for g in gs:
        r = R[g]; b = ms(g, 'rmse_bound'); p = ms(g, 'rmse_postvar'); e = ms(g, 'rmse')
        f.write(f"{float(g):.3f} & {ms(g,'mon')[0]:.3f} & {r['Istate']:.3f} & {r['Ikin']:.3f} & {r['eta']:.2f} & {b[0]:.2f} & {p[0]:.2f} & {e[0]:.1f}$\\pm${e[1]:.1f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
    f.write("%%SPLIT%%\n\\begin{tabular}{c c c c c c c c c}\n\\hline\n$\\kappa$ & $\\eta_m$ & $I(m;H)$ & $\\mathcal{E}_{\\rm ale}$ & $\\mathcal{E}_{\\rm epi}$ (exact) & $\\mathcal{E}_{\\rm epi}$ (pointwise) & $\\mathcal{E}_{\\rm epi}$ (Jensen) & $\\sqrt{\\rm bound}$ & Bayes RMSE \\\\\n & & [nats] & & & & & [cycles] & [cycles] \\\\\n\\hline\n")
    for r in rb: f.write(f"{r['kappa']:.0f} & {r['eta']:.2f} & {r['I_mH']:.2f} & {r['ale']:.1f} & {r['epi_exact']:.1f} & {r['epi_pointwise']:.1f} & {r['epi_jensen']:.1f} & {r['rmse_bound']:.2f} & {r['rmse_bayes_post']:.2f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(open('../results/tables.tex').read())
