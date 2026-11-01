"""Exp 2b: physics-constrained bound with UNKNOWN inspection state (latent theta=(D_t, mu)).
D_k = D_{k-1} + mu + sB*eps_k, D_0=0, mu ~ N(mu0, smu^2) truncated (mu>0.3, a=Dfail-D_t>1).  Sensor X_k = D_k + xi_k.
HI family H^kappa_k = X_k + kappa*zeta_k (DPI-degraded versions of X). Posterior of (D_t, mu) is Gaussian (Kalman) => exact sampling.
m(theta) = E[T|theta] = a/mu ; T | theta ~ IG(a/mu, a^2/sB^2).  Bayes-optimal MSE = E[Var(T|theta)] + E[Var(m|H)]."""
import numpy as np, json
from scipy import stats
rng = np.random.default_rng(7)
mu0, smu, sB, sx, Df, t = 1.0, 0.22, 0.85, 1.5, 100.0, 50
def vasicek(x, m):
    x = np.sort(x); n = len(x); lo = np.clip(np.arange(n)-m, 0, n-1); hi = np.clip(np.arange(n)+m, 0, n-1)
    return np.mean(np.log(n/(2*m)*np.maximum(x[hi]-x[lo], 1e-12)))
def units(N):
    out = []
    while sum(len(o[0]) for o in out) < N:
        mu = rng.normal(mu0, smu, 2*N); D = np.cumsum(mu[:, None] + sB*rng.standard_normal((2*N, t)), axis=1); ok = (mu > 0.3) & (D[:, -1] < Df-1); out.append((mu[ok], D[ok]))
    return np.concatenate([o[0] for o in out])[:N], np.concatenate([o[1] for o in out])[:N]
def kalman(Y, s2):
    """State z=[D, mu]; D_k = D_{k-1}+mu+sB eps; mu const; prior D_0=0, mu~N(mu0,smu^2); obs Y_k = D_k + noise(s2)."""
    F = np.array([[1., 1.], [0., 1.]]); Q = np.diag([sB**2, 0.]); Hm = np.array([[1., 0.]]); P = np.diag([0., smu**2]); N = Y.shape[0]
    z = np.tile([0., mu0], (N, 1))
    for k in range(Y.shape[1]):
        zp = z @ F.T; Pp = F @ P @ F.T + Q; S = Pp[0, 0] + s2; K = Pp[:, 0]/S; z = zp + (Y[:, k]-zp[:, 0])[:, None]*K[None, :]; P = Pp - np.outer(K, Pp[0, :])
    return z, P
N, S = 1500, 1500
mu_t, D_t = units(N); mu_t_, Dpath = mu_t, D_t; a_true = Df - Dpath[:, -1]
X = Dpath + sx*rng.standard_normal(Dpath.shape)
Tt = stats.invgauss.rvs(mu=sB**2/(mu_t*a_true), scale=a_true**2/sB**2, random_state=11)   # realised RUL
# prior entropy of m
p_mu, p_D = units(200000); p_a = Df - p_D[:, -1]; h_prior = vasicek(p_a/p_mu, 450)
rows = []; I_X = None
for kappa in [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    Hobs = X + kappa*rng.standard_normal(X.shape); s2 = sx**2 + kappa**2; z, P = kalman(Hobs, s2); L = np.linalg.cholesky(P + 1e-12*np.eye(2))
    hs, Nb, ale, epi, That = [], [], [], [], []
    for i in range(N):
        smp = z[i] + rng.standard_normal((4*S, 2)) @ L.T; ok = (smp[:, 1] > 0.3) & (smp[:, 0] < Df-1); smp = smp[ok][:S]; a = Df - smp[:, 0]; mm = a/smp[:, 1]
        hs.append(vasicek(mm, 40)); ale.append(np.mean(a*sB**2/smp[:, 1]**3)); epi.append(np.var(mm)); That.append(mm.mean())
    hs = np.array(hs); Nb = np.exp(2*hs)/(2*np.pi*np.e); That = np.array(That); ale = np.array(ale); epi = np.array(epi)
    I_mH = h_prior - hs.mean()
    if kappa == 0.0: I_X = I_mH
    rows.append(dict(kappa=kappa, I_mH=float(I_mH), eta=float(I_mH/I_X), ale=float(ale.mean()), epi_exact=float(epi.mean()), epi_pointwise=float(Nb.mean()),
                     epi_jensen=float(np.exp(2*hs.mean())/(2*np.pi*np.e)), rmse_bound=float(np.sqrt(ale.mean()+Nb.mean())), rmse_bound_jensen=float(np.sqrt(ale.mean()+np.exp(2*hs.mean())/(2*np.pi*np.e))),
                     rmse_bayes_post=float(np.sqrt(ale.mean()+epi.mean())), rmse_realised=float(np.sqrt(np.mean((Tt-That)**2)))))
    print({k: round(v, 3) for k, v in rows[-1].items()}, flush=True)
json.dump(dict(h_prior=float(h_prior), I_mX=float(I_X), rows=rows, params=dict(mu0=mu0, smu=smu, sB=sB, sx=sx, Df=Df, t=t, N=N, S=S)), open('/home/user/paper/results/exp2b.json', 'w'), indent=1)
