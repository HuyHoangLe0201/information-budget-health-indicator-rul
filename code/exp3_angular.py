"""Exp 3: angular loss law for static scalar HIs. Latent theta=(D_t, V_t) from the local-linear-trend model of Exp 1 (time-varying velocity),
raw sensor X_k = D_k + N(0, s1^2).  Target m(theta)=(Dfail-D_t)/V_t (frozen-velocity mean RUL). vartheta_hat=E[theta|X] is 2-d sufficient.
Static scalar HI h_u = u^T vartheta_hat. Prediction: B(h_u)/B_min = 1 + rho sin^2(psi) (linearised); Monte-Carlo uses exact conditional Gaussian sampling + Vasicek entropy of the exact nonlinear m."""
import numpy as np, json
import exp1_paradox as E
rng = np.random.default_rng(3); t, Df, vbar, s1 = E.t, E.Dfail, E.vbar, E.s1
def vasicek(x, m):
    x = np.sort(x); n = len(x); lo = np.clip(np.arange(n)-m, 0, n-1); hi = np.clip(np.arange(n)+m, 0, n-1); return np.mean(np.log(n/(2*m)*np.maximum(x[hi]-x[lo], 1e-12)))
Sig = E.prior_cov(t)[t]; th0 = np.array([vbar*t, vbar]); Ks, Ps = E.kalman_cov(1.0, s1**2, t); P = Ps[-1]; Lam = Sig - P
print('eig(Lam)', np.linalg.eigvalsh(Lam), 'eig(P)', np.linalg.eigvalsh(P))
N, S = 500, 2000; D, T_ = E.simulate(N, rng); Dp = D[:, 1:t+1]; X = Dp + s1*rng.standard_normal(Dp.shape)
s = np.zeros((N, 2)); s[:, 1] = vbar
for k in range(t): sp = s @ E.F.T; s = sp + (X[:, k]-sp[:, 0])[:, None]*Ks[k][None, :]
th_hat = s
def cond_samples(mean, cov, n):
    L = np.linalg.cholesky(cov + 1e-12*np.eye(2)); z = mean + rng.standard_normal((4*n, 2)) @ L.T; ok = (z[:, 1] > 0.5) & (z[:, 0] < Df-1); z = z[ok][:n]; return (Df-z[:, 0])/z[:, 1]
cases = {'level only u=(1,0)': lambda g: np.array([1., 0.]), 'velocity only u=(0,1)': lambda g: np.array([0., 1.]), 'static, u = gradient at prior mean': lambda g: np.array([-1/vbar, -(Df-vbar*t)/vbar**2]), 'phase-adaptive u_t = g_t': lambda g: g}
rows = {k: dict(pred=[], mc=[], psi=[]) for k in cases}
for i in range(N):
    Dh, vh = th_hat[i]; a = Df - Dh; g = np.array([-1/vh, -a/vh**2]); rho = (g @ Lam @ g)/(g @ P @ g); N_X = np.exp(2*vasicek(cond_samples(th_hat[i], P, S), 40))
    for k, fu in cases.items():
        u = fu(g); lu = u @ Lam @ u; cos = (g @ Lam @ u)/np.sqrt((g @ Lam @ g)*lu); c = Lam @ u; cov = Sig - np.outer(c, c)/lu; mean_h = th0 + c*(u @ th_hat[i] - u @ th0)/lu
        N_h = np.exp(2*vasicek(cond_samples(mean_h, cov, S), 40)); rows[k]['pred'].append(1 + rho*(1-cos**2)); rows[k]['mc'].append(N_h/N_X); rows[k]['psi'].append(np.degrees(np.arccos(min(1, abs(cos)))))
res = {k: dict(pred_mean=float(np.mean(v['pred'])), mc_mean=float(np.mean(v['mc'])), psi_mean_deg=float(np.mean(v['psi'])), pred_median=float(np.median(v['pred'])), mc_median=float(np.median(v['mc']))) for k, v in rows.items()}
for k, v in res.items(): print('%-40s psi %.1f deg  predicted %.3f (med %.3f)  Monte Carlo %.3f (med %.3f)' % (k, v['psi_mean_deg'], v['pred_mean'], v['pred_median'], v['mc_mean'], v['mc_median']))
json.dump(dict(res=res, Lam=Lam.tolist(), P=P.tolist(), Sig=Sig.tolist()), open('/home/user/paper/results/exp3.json', 'w'), indent=1)
