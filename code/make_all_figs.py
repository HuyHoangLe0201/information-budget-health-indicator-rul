"""Regenerates every data figure of the paper from the stored result files (no re-simulation except cheap closed forms).
Run from /home/user/paper/code :  python3 make_all_figs.py"""
import json, os, pickle, re, sys
import numpy as np, matplotlib, matplotlib.patches
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
J = lambda p: json.load(open(os.path.join(ROOT, p)))
OUT = os.path.join(ROOT, 'figs')

BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YEL, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#6b6b6b'
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.3, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'lines.linewidth': 1.5, 'figure.dpi': 150, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})
W = 6.0  # text width in inches (153 mm)

def panel(ax, s):
    ax.text(-0.16, 1.08, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf')); plt.close(fig); print('wrote', name)

# =====================================================================  Fig: squeezing (Thm 3)
def fig_paradox():
    import exp1_paradox as E
    d = J('results/exp1.json'); R = d['res']; gs = list(R)
    x = np.arange(len(gs)); xt = ['0', '.5', '.9', '.97', '.99', '.995']
    arr = lambda k: np.array([[r[k] for r in R[g]['runs']] for g in gs])
    mon, rb, rp, re_ = arr('mon'), arr('rmse_bound'), arr('rmse_postvar'), arr('rmse')
    fig = plt.figure(figsize=(W, 4.9)); gsp = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42, height_ratios=[1, 1.08])
    # ---- top row: sample paths of the squeezed HI
    rng = np.random.default_rng(11); N = 14; D, T = E.simulate(N, rng); K = E.t; ks = np.arange(1, K + 1)
    m = np.mean(E.simulate(400, np.random.default_rng(3))[0][:, 1:K + 1], axis=0)
    xi1 = E.s1 * rng.standard_normal((N, K)); xi2 = E.s2 * rng.standard_normal((N, K)); Hraw = D[:, 1:K + 1] + xi1
    cmap = plt.cm.viridis; norm = plt.Normalize(T.min(), T.max())
    for j, g in enumerate([0.0, 0.9, 0.99]):
        ax = fig.add_subplot(gsp[0, j]); H = m + (1 - g) * (Hraw - m) + xi2
        for i in np.argsort(-T): ax.plot(ks, H[i], color=cmap(norm(T[i])), lw=1.0, alpha=.9)
        ax.plot(ks, m, 'k--', lw=1.2, label='template mean')
        if j == 0: panel(ax, '(a)'); ax.set_ylabel('health indicator $\\tilde H_k$')
        gi = gs.index(str(g)) if str(g) in gs else gs.index(f'{g:g}')
        ax.set_title(fr'$\gamma={g:g}$:  monotonicity $={mon[gi].mean():.2f}$,  $\eta_{{\mathrm{{info}}}}={R[gs[gi]]["eta"]:.2f}$', fontsize=8, pad=3)
        ax.set_xlabel('time step $k$'); ax.set_xlim(1, K)
        if j == 2:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.03); cb.set_label('true RUL at $t=50$', fontsize=7); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
        if j == 0: ax.legend(loc='upper left', fontsize=7)
    # ---- bottom row
    a = fig.add_subplot(gsp[1, 0]); panel(a, '(b)')
    a.fill_between(x, mon.mean(1) - mon.std(1), mon.mean(1) + mon.std(1), color=BLUE, alpha=.18, lw=0)
    a.plot(x, mon.mean(1), '-o', color=BLUE, ms=4); a.plot(x, [R[g]['eta'] for g in gs], '-s', color=RED, ms=4, mfc='w')
    a.axhline(0.981, color=BLUE, ls=':', lw=1); a.text(x[5], 1.02, r'$\mathrm{Mon}_\infty=0.981$', color=BLUE, fontsize=7, va='bottom', ha='right')
    a.text(x[4] + .3, 0.45, r'$\eta_{\mathrm{info}}$' + '\ndecreases', color=RED, fontsize=7.5, ha='left'); a.text(x[3] + .6, 0.86, 'monotonicity saturates', color=BLUE, fontsize=7.5, va='top', ha='center')
    a.set_ylim(0, 1.15); a.set_ylabel('score'); a.set_xlabel(r'shrinkage strength $\gamma$'); a.set_xticks(x); a.set_xticklabels(xt)
    b = fig.add_subplot(gsp[1, 1]); panel(b, '(c)')
    b.plot(x, [R[g]['Itot'] for g in gs], '-o', color='k', ms=4); b.plot(x, [R[g]['Istate'] for g in gs], '-s', color=BLUE, ms=4, mfc='w'); b.plot(x, [R[g]['Ikin'] for g in gs], '-^', color=ORANGE, ms=4.5, mfc='w')
    b.text(x[0] + .1, 0.93, 'total', fontsize=7.5, color='k'); b.text(x[1] + .6, 0.47, 'state', fontsize=7.5, color=BLUE, ha='center')
    b.annotate('kinematic\n(non-monotone)', xy=(x[3] + .05, 0.4), xytext=(x[4] + .2, 0.66), fontsize=7.2, color=ORANGE, ha='center', va='bottom', arrowprops=dict(arrowstyle='->', color=ORANGE, lw=.7))
    b.set_ylim(0, 1.0); b.set_ylabel('information about $V_t$ [nats]'); b.set_xlabel(r'shrinkage strength $\gamma$'); b.set_xticks(x); b.set_xticklabels(xt)
    c = fig.add_subplot(gsp[1, 2]); panel(c, '(d)')
    c.errorbar(x, re_.mean(1), yerr=re_.std(1), fmt='s', color='#9a9a9a', ms=3.2, capsize=2, lw=.8, alpha=.8)
    c.plot(x, rb.mean(1), '-o', color=BLUE, ms=4); c.plot(x, rp.mean(1), '--^', color=RED, ms=4.5, mfc='w')
    c.text(x[1] + .5, 27.5, 'empirical RMSE\n(mean $\\pm$ standard\ndeviation, 3 seeds)', fontsize=7, color=GREY, va='bottom', ha='center'); c.annotate('Bayes-optimal', xy=(x[4] + .05, 15.5), xytext=(x[2] + .35, 19.5), fontsize=7.5, color=RED, ha='center', arrowprops=dict(arrowstyle='->', color=RED, lw=.7)); c.text(x[4] + .1, 4.6, r'bound $\sqrt{\mathcal{B}_{\mathrm{info}}}$', fontsize=7.5, color=BLUE, ha='center')
    c.set_ylim(3, 35); c.set_ylabel('RUL RMSE [cycles]'); c.set_xlabel(r'shrinkage strength $\gamma$'); c.set_xticks(x); c.set_xticklabels(xt)
    save(fig, 'fig_paradox')

# =====================================================================  Fig: physics bound (Thm 4, Prop 5)
def fig_bound():
    d = J('results/exp2b.json'); rows = d['rows']; eta = np.array([r['eta'] for r in rows]); o = np.argsort(eta); eta = eta[o]
    g = lambda k: np.array([rows[i][k] for i in o]); ale = g('ale')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.32))
    a = ax[0]; panel(a, '(a)')
    a.fill_between(eta, 0, np.sqrt(ale), color='#e4e4e4'); a.text(0.97, 1.0, r'aleatoric floor $\sqrt{\mathcal{E}_{\mathrm{aleatoric}}}$ (independent of the HI)', fontsize=7, color=GREY, va='bottom')
    a.plot(eta, g('rmse_bound_jensen'), '-.', color=ORANGE, marker='D', ms=3.5, label='bound, Jensen form')
    a.plot(eta, g('rmse_bound'), '-o', color=BLUE, ms=4, label='bound, pointwise form')
    a.plot(eta, g('rmse_bayes_post'), '--^', color=RED, ms=4.5, mfc='w', label='Bayes-optimal RMSE')
    a.plot(eta, g('rmse_realised'), ':s', color='k', ms=3.5, mfc='w', lw=1, label='empirical RMSE')
    a.set_xlabel(r'efficiency about the drift state, $\eta_m=I(m;H)/I(m;\mathbf{X})$'); a.set_ylabel('RUL RMSE [cycles]'); a.set_ylim(0, 26); a.legend(loc='upper left', fontsize=6.8)
    a.invert_xaxis()
    b = ax[1]; panel(b, '(b)')
    ex, pw, jn = g('epi_exact'), g('epi_pointwise'), g('epi_jensen')
    b.fill_between(eta, pw, ex, color=RED, alpha=.25, lw=0); b.annotate('non-Gaussianity gap', xy=(0.75, 125), xytext=(0.97, 215), fontsize=7, color=RED, arrowprops=dict(arrowstyle='->', color=RED, lw=.7))
    b.plot(eta, ex, '--^', color=RED, ms=4.5, mfc='w', label=r'exact $\mathbb{E}\,\mathrm{Var}(m\mid H)$')
    b.plot(eta, pw, '-o', color=BLUE, ms=4, label=r'pointwise entropy power')
    b.plot(eta, jn, '-.D', color=ORANGE, ms=3.5, label='Jensen form')
    b.set_xlabel(r'$\eta_m$'); b.set_ylabel(r'epistemic term $\mathcal{E}_{\mathrm{epistemic}}$ [cycles$^2$]'); b.invert_xaxis(); b.legend(loc='lower right', fontsize=6.8)
    save(fig, 'fig_bound')

# =====================================================================  Fig: angular law + rotation (Thm 5, Cor 1-2)
def fig_angular():
    import exp1_paradox as E
    d = J('results/exp3b.json'); TT = np.array(d['TT']); cur = d['curve']; mc = d['mc']
    keys = ['level only', 'velocity only', 'static (aligned at t*=50)', 'phase-adaptive']
    lab = {'level only': r'level only $\mathbf{u}=(1,0)$', 'velocity only': r'velocity only $\mathbf{u}=(0,1)$', 'static (aligned at t*=50)': r'static, aligned at $t^\ast=50$', 'phase-adaptive': r'stage-adaptive $\mathbf{u}_t=\mathbf{g}_t$'}
    col = {'level only': BLUE, 'velocity only': RED, 'static (aligned at t*=50)': ORANGE, 'phase-adaptive': GREEN}
    mk = {'level only': 'o', 'velocity only': 's', 'static (aligned at t*=50)': '^', 'phase-adaptive': 'v'}; ls = {'level only': '-', 'velocity only': '--', 'static (aligned at t*=50)': '-.', 'phase-adaptive': '-'}
    Df, vbar, s1 = E.Dfail, E.vbar, E.s1; TMAX = int(TT.max()); Ks, Ps = E.kalman_cov(1.0, s1 ** 2, TMAX); PC = E.prior_cov(TMAX)
    tstar = 50; ustat = np.array([-1 / vbar, -(Df - vbar * tstar) / vbar ** 2])
    psi = {k: [] for k in keys}; rho = []
    for t in TT:
        Sig = PC[t]; P = Ps[t - 1]; Lam = Sig - P; gv = np.array([-1 / vbar, -(Df - vbar * t) / vbar ** 2]); rho.append((gv @ Lam @ gv) / (gv @ P @ gv))
        for k, u in [('level only', np.array([1., 0.])), ('velocity only', np.array([0., 1.])), ('static (aligned at t*=50)', ustat), ('phase-adaptive', gv)]:
            c = (gv @ Lam @ u) / np.sqrt((gv @ Lam @ gv) * (u @ Lam @ u)); psi[k].append(np.degrees(np.arccos(np.clip(abs(c), 0, 1))))
    fig, ax = plt.subplots(1, 3, figsize=(W, 3.2), gridspec_kw=dict(wspace=0.5))
    a = ax[0]; panel(a, '(a)')
    for k in keys:
        a.plot(TT, cur[k], ls[k], color=col[k], label=lab[k])
        mk_ = 'phase-adaptive (stage-fixed)' if k == 'phase-adaptive' else k
        a.plot([int(t) for t in mc], [mc[t][mk_]['mc'] for t in mc], mk[k], color=col[k], mfc='w', ms=5, lw=0)
        a.plot([int(t) for t in mc], [mc[t][mk_]['pred'] for t in mc], '+', color=col[k], ms=6, mew=1.1, lw=0)
    a.axvline(tstar, color='#aaaaaa', ls=':', lw=1); a.text(tstar + 1, 0.93, r'$t^\ast$', color=GREY, fontsize=8)
    a.annotate(r'$5\times$ at $t=75$', xy=(75, mc['75']['velocity only']['mc']), xytext=(30, 6.2), fontsize=7.2, color=RED, arrowprops=dict(arrowstyle='->', color=RED, lw=.7))
    a.set_yscale('log'); a.set_xlabel('inspection time $t$'); a.set_ylabel(r'bound increase factor $\mathcal{B}(h_{\mathbf{u}})/\mathcal{B}_{\min}$'); a.set_ylim(0.9, 20)
    b = ax[1]; panel(b, '(b)')
    for k in keys: b.plot(TT, psi[k], ls[k], color=col[k])
    b.axvline(tstar, color='#aaaaaa', ls=':', lw=1); b.set_xlabel('inspection time $t$'); b.set_ylabel(r'$\boldsymbol{\Lambda}$-angle $\psi_t$ to $\mathbf{g}_t$ [deg]'); b.set_ylim(-0.5, 18)
    c = ax[2]; panel(c, '(c)')
    for k in keys:
        eta = [1 - np.log(1 + r_ * np.sin(np.radians(p_)) ** 2) / np.log(1 + r_) for r_, p_ in zip(rho, psi[k])]; c.plot(TT, eta, ls[k], color=col[k])
    c.axvline(tstar, color='#aaaaaa', ls=':', lw=1); c.set_xlabel('inspection time $t$'); c.set_ylabel(r'information efficiency $\eta_m(h_{\mathbf{u}})$'); c.set_ylim(-0.02, 1.05)
    fig.legend(handles=[Line2D([], [], color=col[k], ls=ls[k], lw=1.5, label=lab[k]) for k in keys]+[Line2D([], [], color='k', lw=1.4, label='theory, fixed operating point'), Line2D([], [], color='k', marker='+', lw=0, ms=6, mew=1.1, label='theory, unit-specific gradient'), Line2D([], [], color='k', marker='o', mfc='w', lw=0, label='Monte Carlo, exact target')], loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=2.2)
    fig.subplots_adjust(bottom=0.33)
    save(fig, 'fig_angular')

# =====================================================================  Fig: regime leakage (Prop 7)
def fig_regime():
    rng = np.random.default_rng(1); Rn = 6; pi = np.full(Rn, 1 / Rn); a = np.array([0.6, 0.8, 1.0, 1.0, 1.3, 1.6]); b = np.array([-0.5, -0.2, 0.0, 0.1, 0.3, 0.6]); muD, v = 0.5, 0.08
    ED2 = v + muD ** 2; Ea, Ea2, Eb = (pi * a).sum(), (pi * a ** 2).sum(), (pi * b).sum(); Vb = (pi * b ** 2).sum() - Eb ** 2; Cab = (pi * a * b).sum() - Ea * Eb
    sig = np.linspace(0, 0.12, 25)
    blind = np.array([v - (Ea * v) ** 2 / (Ea2 * ED2 - (Ea * muD) ** 2 + Vb + 2 * muD * Cab + s ** 2) for s in sig]); aware = np.array([(pi * s ** 2 / a ** 2).sum() for s in sig])
    mcs = [0.0, 0.02, 0.05, 0.1]; mb, ma = [], []
    for s in mcs:
        n = 200000; D = rng.normal(muD, np.sqrt(v), n); r = rng.integers(0, Rn, n); x = a[r] * D + b[r] + s * rng.standard_normal(n)
        A = np.c_[x, np.ones(n)]; w = np.linalg.lstsq(A, D, rcond=None)[0]; mb.append(np.mean((A @ w - D) ** 2))
        ma.append(np.mean([np.mean((np.c_[x[r == k], np.ones((r == k).sum())] @ np.linalg.lstsq(np.c_[x[r == k], np.ones((r == k).sum())], D[r == k], rcond=None)[0] - D[r == k]) ** 2) for k in range(Rn)]))
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.1]))
    p = ax[0]; panel(p, '(a)'); cm = plt.cm.viridis(np.linspace(0.05, 0.9, Rn)); n = 1200; D = rng.normal(muD, np.sqrt(v), n); r = rng.integers(0, Rn, n); x = a[r] * D + b[r] + 0.02 * rng.standard_normal(n)
    for k in range(Rn):
        p.scatter(D[r == k], x[r == k], s=4, color=cm[k], alpha=.7, lw=0, label=f'regime {k + 1}')
    p.set_xlabel('latent level $D$'); p.set_ylabel('sensor output $x=c_rD+o_r+\\varepsilon$'); p.legend(ncol=2, fontsize=6.3, markerscale=2.5, loc='upper left', columnspacing=.8, handletextpad=.2)
    q = ax[1]; panel(q, '(b)')
    q.plot(sig, blind, '-', color=RED, label=r'regime-blind linear estimator'); q.plot(mcs, mb, 'o', color=RED, mfc='w', ms=5, label='Monte Carlo (blind)')
    q.plot(sig, aware, '-', color=GREEN, label=r'regime-aware estimator'); q.plot(mcs, ma, 's', color=GREEN, mfc='w', ms=4.5, label='Monte Carlo (aware)')
    q.axhline(v, color=GREY, ls=':', lw=1); q.text(0.001, v * 1.01, r'$\mathrm{Var}(D)$', ha='left', fontsize=7, color=GREY)
    q.annotate('non-vanishing\nerror floor', xy=(0.0, blind[0]), xytext=(0.03, 0.018), fontsize=7, color=RED, arrowprops=dict(arrowstyle='->', color=RED, lw=.7))
    q.set_xlabel(r'sensor noise $\sigma$'); q.set_ylabel('MSE of the level estimate'); q.set_ylim(0, 0.09); q.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.6, columnspacing=1.0, handlelength=1.8)
    save(fig, 'fig_regime')

# =====================================================================  Fig: seven small sets (heat-map of RMSE relative to age-only baseline)
def fig_small():
    txt = open(os.path.join(ROOT, 'results/tab_summary_small.tex')).read(); rows = []
    for ln in txt.split('\n'):
        if '&' in ln and 'Data set' not in ln:
            cells = [c.strip().rstrip('\\').strip() for c in ln.replace('\\\\', '').split('&')]; name = cells[0].replace('$^\\dagger$', '*')
            vals = []
            for c in cells[1:]:
                m = re.search(r'([0-9]+\.?[0-9]*)', c); vals.append(float(m.group(1)) if m and c != '--' else np.nan)
            rows.append((name, vals))
    meth = ['Age-only', 'PCA-HI', 'Geo-opt. HI', r'$\beta$-VAE', 'VIB', 'PGIB', 'PGIB-KF']
    M = np.array([v for _, v in rows]); rel = M / M[:, [0]]
    fig, ax = plt.subplots(figsize=(W, 2.9)); ax.grid(False)
    im = ax.imshow(np.log2(np.clip(rel, 0.25, 4)), cmap='RdBu_r', vmin=-1.6, vmax=1.6, aspect='auto')
    for i in range(M.shape[0]):
        best = np.nanargmin(M[i])
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]): ax.text(j, i, '--', ha='center', va='center', fontsize=7.5, color=GREY); continue
            s = f'{M[i, j]:.0f}' if M[i, j] >= 100 else f'{M[i, j]:.1f}'
            ax.text(j, i, s, ha='center', va='center', fontsize=7.8, fontweight='bold' if j == best else 'normal', color='k')
            if j == best: ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='k', lw=1.4))
    ax.set_xticks(range(len(meth))); ax.set_xticklabels(meth); ax.xaxis.tick_top(); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[-1.6, -1, 0, 1, 1.6]); cb.ax.set_yticklabels(['', r'$2\times$ better', 'age-only baseline', r'$2\times$ worse', '']); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=0)
    save(fig, 'fig_small')

# =====================================================================  Fig: C-MAPSS, seeds and baselines
def fig_cmapss():
    V = J('results/variance10.json'); sub = ['FD001', 'FD002', 'FD003', 'FD004']; rng = np.random.default_rng(0)
    rows = [('Age-only baseline', 'base', 'Age-only', GREY), ('PCA-HI', 'base', 'PCA-HI', SKY), ('Geo-opt. HI', 'base', 'Mon-opt HI', BLUE), ('Ridge (supervised)', 'base', 'Ridge-HI (supervised)', PURPLE),
            (r'$\beta$-VAE (10 seeds)', 'seed', 'beta-VAE-HI', ORANGE), ('PGIB, integrating (10 seeds)', 'seed', 'PGIB-HI', RED), ('PGIB-KF (10 seeds)', 'seed', 'PGIB-KF-HI', GREEN), ('PGIB-KFR (10 seeds)', 'seed', 'PGIB-KFR-HI', '#1b7a5a')]
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.2), gridspec_kw=dict(hspace=0.45, wspace=0.08), sharey=True)
    for ax, fd, lab in zip(axs.ravel(), sub, ['(a) FD001, 1 regime', '(b) FD002, 6 regimes', '(c) FD003, 1 regime', '(d) FD004, 6 regimes']):
        d = V[fd]; best = min(d['baselines'][k] for k in ['PCA-HI', 'Mon-opt HI', 'Ridge-HI (supervised)'])
        ax.axvline(best, color='k', ls=':', lw=.9); ax.grid(axis='y', visible=False)
        for i, (nm, kind, key, c) in enumerate(rows):
            y = len(rows) - 1 - i
            if kind == 'base': ax.plot(d['baselines'][key], y, 's', color=c, ms=6.5)
            else:
                s = np.array(d[key]['seed_rmse']); ax.plot(s, y + rng.uniform(-.17, .17, len(s)), 'o', color=c, ms=3, alpha=.55, mfc=c, lw=0)
                ax.errorbar(s.mean(), y, xerr=s.std(ddof=1), fmt='|', color='k', ms=9, capsize=3, lw=1.3, zorder=4)
                if 'ens' in d[key]: ax.plot(d[key]['ens'], y, 'D', color=c, mec='k', ms=6, zorder=5)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1]); ax.set_title(lab, loc='left', fontsize=8.5); ax.set_xlabel('RMSE [cycles]')
        ax.set_ylim(-.6, len(rows) - .4); lo_, hi_ = ax.get_xlim(); pad = 0.04 * (hi_ - lo_); ax.set_xlim(lo_ - pad, hi_ + pad)
    for ax in axs[:, 1]: ax.tick_params(axis='y', length=0)
    h = [Line2D([], [], marker='s', color=GREY, mfc=GREY, lw=0, ms=6, label='deterministic HI'), Line2D([], [], marker='o', color='#777', lw=0, ms=3.5, alpha=.7, label='single model (one seed)'),
         Line2D([], [], marker='|', color='k', lw=0, ms=9, mew=1.3, label=r'mean $\pm$ standard deviation'), Line2D([], [], marker='D', mfc='w', mec='k', color='k', lw=0, ms=6, label='10-seed ensemble'), Line2D([], [], color='k', ls=':', label='best simple/supervised HI')]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=7.2, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    fig.subplots_adjust(bottom=0.16)
    save(fig, 'fig_cmapss')

# =====================================================================  Fig: example RUL trajectories (Appendix, protocol P1)
def fig_bench():
    B = os.path.join(ROOT, 'bench'); cells = [('nasa', 'B0005', 'NASA PCoE'), ('calce', 'CS2_35', 'CALCE CS2'), ('femto1', 'Bearing1_1', 'FEMTO cond. 1')]
    fig, ax = plt.subplots(3, 2, figsize=(W, 6.4), gridspec_kw=dict(hspace=0.55, wspace=0.28))
    for i, (ds, cell, nm) in enumerate(cells):
        r = pickle.load(open(os.path.join(B, f'out/{ds}_{cell}.pkl'), 'rb')); s = r['stride']
        for j, m in enumerate(['PCA-HI', 'PGIB-HI']):
            x = r['methods'][m][0]; k = np.arange(len(x['y'])) * s; a = ax[i, j]
            rm = float(np.sqrt(np.mean((x['med'] * s - x['y'] * s) ** 2))); cov = float(np.mean((x['y'] >= x['lo']) & (x['y'] <= x['hi'])))
            a.fill_between(k, x['lo'] * s, x['hi'] * s, color=BLUE if j == 0 else GREEN, alpha=.2, lw=0, label='95 % interval')
            a.plot(k, x['med'] * s, '--', color=BLUE if j == 0 else GREEN, lw=1.5, label='median prediction'); a.plot(k, x['y'] * s, '-', color='k', lw=1.4, label='true RUL')
            a.set_ylim(0, min(max(x['hi'] * s) * 1.05, 4 * max(x['y'] * s))); a.set_title(f"({'abcdef'[2 * i + j]}) {nm}, {cell}, {m}", loc='left', fontsize=8.3)
            a.text(0.97, 0.95, f'RMSE {rm:.0f}\ncoverage {100 * cov:.0f} %', transform=a.transAxes, ha='right', va='top', fontsize=7, bbox=dict(boxstyle='round,pad=.25', fc='w', ec='#cccccc', lw=.6))
            a.set_xlabel('position in evaluation window [' + ('cycles' if ds != 'femto1' else 'snapshots') + ']'); a.set_ylabel('RUL [' + ('cycles' if ds != 'femto1' else 'snapshots') + ']')
    h = [Line2D([], [], color='k', lw=1.4, label='true RUL'), Line2D([], [], color=BLUE, ls='--', lw=1.5, label='median prediction (PCA-HI)'), Line2D([], [], color=GREEN, ls='--', lw=1.5, label='median prediction (PGIB-HI)'),
         matplotlib.patches.Patch(color=BLUE, alpha=.25, label='95 % interval (PCA-HI)'), matplotlib.patches.Patch(color=GREEN, alpha=.25, label='95 % interval (PGIB-HI)')]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0)); fig.subplots_adjust(bottom=0.11)
    save(fig, 'fig_bench')

# =====================================================================  Fig: real-data checks on C-MAPSS (certificates, squeeze, stages)
def fig_realdata():
    import os
    FD4 = ['FD001', 'FD002', 'FD003', 'FD004']; FDs = FD4+[b for b in ['HUST', 'MATR', 'Tongji'] if os.path.exists(f'results/realdata_{b}.json')]; Rr = {fd: J(f'results/realdata_{fd}.json') for fd in FDs}; Cc = J('results/cert_age_cmapss.json'); cols = [BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY]; nS = len(FDs)
    G = ['0.0', '0.5', '0.9', '0.97', '0.99', '0.995']; xg = np.arange(len(G)); xt = ['0', '.5', '.9', '.97', '.99', '.995']
    # ---- Figure A: cross-validated certificates and squeeze
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.7), gridspec_kw=dict(wspace=0.45, width_ratios=[1.7, 1, 1]))
    a = ax[0]; panel(a, '(a)'); preds = [('age only', 'age\n(control)'), ('ridge HI', 'ridge\nHI'), ('PCA-HI', 'PCA\nHI'), ('ridge level only', 'level\nonly'), ('ridge slope only', 'slope\nonly'), ('sensors', 'raw\nsensors')]
    for j, fd in enumerate(FDs):
        c = Rr[fd]['cert']; ci = Rr[fd]['cert_ci']; xs = np.arange(len(preds))+(j-(nS-1)/2)*0.115; kk = [('charge sensors' if (k == 'sensors' and 'charge sensors' in c) else k) for k, _ in preds]; y = np.array([c[k] for k in kk]); lo = y-np.array([ci[k][0] for k in kk]); hi = np.array([ci[k][1] for k in kk])-y
        a.errorbar(xs, y, yerr=[lo, hi], fmt='o', color=cols[j], ms=3.5, capsize=1.5, lw=.9, label=fd)
    a.axhline(0, color='k', lw=.8); a.set_ylim(-0.5, 0.9); a.set_xticks(range(len(preds))); a.set_xticklabels([l for _, l in preds], fontsize=6.6); a.set_ylabel(r'lower bound on $I(T;H\mid t)$ [nats]');     b = ax[1]; panel(b, '(b)')
    for j, fd in enumerate(FDs):
        b.plot(xg, [Rr[fd]['mon'][g] for g in G], '-o', color=cols[j], ms=3.5, label=fd); b.axhline(Rr[fd]['mon_inf_pred'], color=cols[j], ls=':', lw=1)
    b.set_xticks(xg); b.set_xticklabels(xt, fontsize=7); b.set_xlabel(r'shrinkage strength $\gamma$'); b.set_ylabel('monotonicity')
    c_ = ax[2]; panel(c_, '(c)')
    for j, fd in enumerate(FDs): c_.plot(xg, [Rr[fd]['cert'][f'squeeze {g}'] for g in G], '-s', color=cols[j], ms=3.5, mfc='w')
    c_.axhline(0, color='k', lw=.8); c_.set_xticks(xg); c_.set_xticklabels(xt, fontsize=7); c_.set_xlabel(r'shrinkage strength $\gamma$'); c_.set_ylabel(r'lower bound on $I(T;H\mid t)$ [nats]')
    fig.subplots_adjust(bottom=0.27); fig.legend(handles=[Line2D([], [], marker='o', color=cols[j], lw=0, ms=4, label=fd) for j, fd in enumerate(FDs)], loc='lower center', ncol=nS, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    save(fig, 'fig_realdata_a')
    # ---- Figure B: official-test certificates and stage-wise directions
    fig, ax = plt.subplots(1, 3, figsize=(W, 3.2), gridspec_kw=dict(wspace=0.55, width_ratios=[1.5, 1, 1]))
    fig.subplots_adjust(bottom=0.24)
    a = ax[0]; panel(a, '(a)'); rows = [('Age-only', 'age-only baseline (control)'), ('PCA-HI', 'PCA-HI'), ('Mon-opt HI', 'Geo-opt HI'), ('Ridge-HI (supervised)', 'ridge HI'), ('beta-VAE-HI', r'$\beta$-VAE'), ('PGIB-HI (ens10)', 'PGIB, integrating'), ('PGIB-KF-HI (ens10)', 'PGIB-KF'), ('PGIB-KFR-HI (ens10)', 'PGIB-KFR')]
    for j, fd in enumerate(FD4):
        for i, (k, _) in enumerate(rows):
            y = len(rows)-1-i+(j-1.5)*0.16; v = Cc[fd]['cert'][k]; ci = Cc[fd]['ci'][k]; a.errorbar([v], [y], xerr=[[v-ci[0]], [ci[1]-v]], fmt='o', color=cols[j], mfc='w' if fd == 'FD003' else cols[j], ms=3.3, capsize=1.3, lw=.8, label=fd if i == 0 else None)
    a.axvline(0, color='k', lw=.8); a.set_yticks(range(len(rows))); a.set_yticklabels([l for _, l in rows][::-1], fontsize=7); a.set_xlabel(r'lower bound on $I(T;H\mid t)$ [nats]'); a.grid(axis='y', visible=False); a.set_title('turbofan test engines', fontsize=7, loc='right')
    fig.legend(handles=[Line2D([], [], marker='o', color=cols[j], mfc='w' if fd == 'FD003' else cols[j], lw=0, ms=4, label=fd+(' (control fails)' if fd == 'FD003' else '')) for j, fd in enumerate(FDs)], loc='lower center', ncol=nS, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    b = ax[1]; panel(b, '(b)')
    for j, fd in enumerate(FDs):
        st = Rr[fd]['stage_age']['stages']; b.plot(np.arange(1, 6), [s['angle_opt'] for s in st], '-o', color=cols[j], ms=3.5); b.axhline(Rr[fd]['stage_age']['static_angle'], color=cols[j], ls=':', lw=1)
    b.set_xlabel('age stage (equal-count bins)'); b.set_ylabel('optimal direction [deg from level axis]'); b.set_ylim(0, 90)
    c = ax[2]; panel(c, '(c)'); x0 = np.arange(len(FDs)); wd = 0.26
    def worst(fd, key): st = Rr[fd]['stage_age']['stages']; return max(np.sqrt(s['mse'][key]/s['mse']['both']) for s in st)
    for k, (key, lab, col) in enumerate([('static', 'static direction', GREEN), ('level', 'level only', BLUE), ('velocity', 'velocity only', ORANGE)]):
        c.bar(x0+(k-1)*wd, [worst(fd, key) for fd in FDs], wd, color=col, label=lab)
    c.axhline(1, color='k', lw=.8); c.set_xticks(x0); c.set_xticklabels(FDs, fontsize=6, rotation=45, ha='right'); c.set_ylim(0.95, 1.6); c.set_ylabel('worst-stage RMSE ratio to stage-wise direction'); c.legend(fontsize=6.3, loc='upper left')
    save(fig, 'fig_realdata_b')

if __name__ == '__main__':
    todo = sys.argv[1:] or ['paradox', 'bound', 'angular', 'regime', 'small', 'cmapss', 'bench', 'realdata']
    for t in todo: globals()['fig_' + t]()
