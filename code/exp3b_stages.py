"""Exp 3b: a static scalar HI cannot be optimal over the whole life: predicted inflation 1+rho sin^2(psi_t) vs inspection time t, with Monte-Carlo checks at t=25,50,75."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import exp1_paradox as E
rng = np.random.default_rng(5); Df, vbar, s1 = E.Dfail, E.vbar, E.s1; TT = list(range(10, 91, 5)); TMAX = max(TT)
Ks, Ps = E.kalman_cov(1.0, s1**2, TMAX); PC = E.prior_cov(TMAX)
def vasicek(x, m):
    x = np.sort(x); n = len(x); lo = np.clip(np.arange(n)-m, 0, n-1); hi = np.clip(np.arange(n)+m, 0, n-1); return np.mean(np.log(n/(2*m)*np.maximum(x[hi]-x[lo], 1e-12)))
tstar = 50; u_static = np.array([-1/vbar, -(Df-vbar*tstar)/vbar**2])
def stats(t):
    Sig = PC[t]; P = Ps[t-1]; return Sig, P, Sig-P, np.array([vbar*t, vbar])
def pred_curve():
    out = {k: [] for k in ['level only', 'velocity only', 'static (aligned at t*=50)', 'phase-adaptive']}
    for t in TT:
        Sig, P, Lam, th0 = stats(t); g = np.array([-1/vbar, -(Df-vbar*t)/vbar**2]); rho = (g @ Lam @ g)/(g @ P @ g)
        for k, u in [('level only', np.array([1., 0.])), ('velocity only', np.array([0., 1.])), ('static (aligned at t*=50)', u_static), ('phase-adaptive', g)]:
            cos = (g @ Lam @ u)/np.sqrt((g @ Lam @ g)*(u @ Lam @ u)); out[k].append(1+rho*(1-cos**2))
    return out
curve = pred_curve()
# Monte Carlo at selected t
N, S = 300, 1500; D, _ = E.simulate(N, rng); X = D[:, 1:TMAX+1] + s1*rng.standard_normal((N, TMAX)); s = np.zeros((N, 2)); s[:, 1] = vbar; snap = {}
for k in range(TMAX):
    sp = s @ E.F.T; s = sp + (X[:, k]-sp[:, 0])[:, None]*Ks[k][None, :]
    if k+1 in (25, 50, 75): snap[k+1] = s.copy()
def cs(mean, cov, n):
    L = np.linalg.cholesky(cov + 1e-12*np.eye(2)); z = mean + rng.standard_normal((4*n, 2)) @ L.T; ok = (z[:, 1] > 0.5) & (z[:, 0] < Df-1); z = z[ok][:n]; return (Df-z[:, 0])/z[:, 1]
mc = {}
for t, th in snap.items():
    Sig, P, Lam, th0 = stats(t); res = {k: [] for k in curve}; pr = {k: [] for k in curve}
    for i in range(N):
        a = Df-th[i, 0]
        if a < 5: continue
        g = np.array([-1/th[i, 1], -a/th[i, 1]**2]); rho = (g @ Lam @ g)/(g @ P @ g); NX = np.exp(2*vasicek(cs(th[i], P, S), 40))
        for k, u in [('level only', np.array([1., 0.])), ('velocity only', np.array([0., 1.])), ('static (aligned at t*=50)', u_static), ('phase-adaptive', g)]:
            lu = u @ Lam @ u; c = Lam @ u; cov = Sig - np.outer(c, c)/lu; mh = th0 + c*(u @ th[i]-u @ th0)/lu; cos = (g @ Lam @ u)/np.sqrt((g @ Lam @ g)*lu)
            res[k].append(np.exp(2*vasicek(cs(mh, cov, S), 40))/NX); pr[k].append(1+rho*(1-cos**2))
    mc[t] = {k: dict(mc=float(np.mean(v)), pred=float(np.mean(pr[k]))) for k, v in res.items()}; print(t, {k: (round(v['pred'], 3), round(v['mc'], 3)) for k, v in mc[t].items()}, flush=True)
fig, ax = plt.subplots(figsize=(3.8, 3.0)); sty = {'level only': ('k-', 'o'), 'velocity only': ('k--', 's'), 'static (aligned at t*=50)': ('k-.', '^'), 'phase-adaptive': ('k:', 'v')}
for k, ys in curve.items(): ax.plot(TT, ys, sty[k][0], label=k, lw=1.3)
for t, d in mc.items():
    for k, v in d.items(): ax.plot([t], [v['mc']], sty[k][1], color='k', mfc='w', ms=4.5)
ax.set_xlabel('Inspection time $t$'); ax.set_ylabel(r'Bound inflation $\mathcal{B}(h_u)/\mathcal{B}_{\min}$'); ax.set_yscale('log'); ax.legend(fontsize=6.5); plt.tight_layout(); plt.savefig('../figs/fig_angular.pdf')
json.dump(dict(TT=TT, curve=curve, mc=mc), open('/home/user/paper/results/exp3b.json', 'w'), indent=1)
print({k: [round(v, 2) for v in ys[::3]] for k, ys in curve.items()})
