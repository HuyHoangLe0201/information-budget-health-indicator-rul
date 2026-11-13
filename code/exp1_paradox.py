"""Exp 1: Monotonicity paradox in an exactly solvable linear-Gaussian degradation model.
Latent: D_k = D_{k-1} + V_{k-1} + sB*w_k ; V_k = V_{k-1} + sV*z_k ; V_0 ~ N(vbar, sv0^2); D_0=0.
Raw HI: H_k = D_k + xi1_k.  Shrunk HI: Ht_k = m_k + g (H_k - m_k) + xi2_k, g = 1-gamma, m_k=E[D_k].
Gaussian => mutual informations about V_t are exact (Kalman). For T (non-Gaussian) the Bayes-optimal MSE and
the entropy-power bound are evaluated from exact posterior samples of the latent state."""
import numpy as np, json, sys
vbar, sv0, sV, sB, s1, s2 = 2.0, 0.25, 0.03, 0.5, 2.0, 0.6
Dfail, t, HMAX = 200.0, 50, 500
F = np.array([[1., 1.], [0., 1.]]); Q = np.diag([sB**2, sV**2])

def prior_cov(t):
    S = np.diag([0., sv0**2]); out = [S]
    for _ in range(t): S = F @ S @ F.T + Q; out.append(S)
    return out

def kalman_cov(g, R, t):
    P = np.diag([0., sv0**2]); Hm = np.array([[g, 0.]]); Ks, Ps = [], []
    for _ in range(t):
        Pp = F @ P @ F.T + Q; S = Hm @ Pp @ Hm.T + R; K = Pp @ Hm.T / S
        P = (np.eye(2) - K @ Hm) @ Pp; Ks.append(K[:, 0]); Ps.append(P.copy())
    return Ks, Ps

def info_V(g, R):
    Sig = prior_cov(t)[t]; Ks, Ps = kalman_cov(g, R, t)
    I_all = 0.5*np.log(Sig[1, 1]/Ps[-1][1, 1])
    vy = g**2*Sig[0, 0] + R; cvy = g*Sig[0, 1]
    I_state = 0.5*np.log(Sig[1, 1]/(Sig[1, 1] - cvy**2/vy))
    return I_state, I_all - I_state, I_all

def simulate(N, rng):
    V = np.empty((N, t+HMAX+1)); D = np.zeros((N, t+HMAX+1)); V[:, 0] = rng.normal(vbar, sv0, N)
    for k in range(1, t+HMAX+1):
        D[:, k] = D[:, k-1] + V[:, k-1] + sB*rng.standard_normal(N)
        V[:, k] = V[:, k-1] + sV*rng.standard_normal(N)
    T = np.full(N, float(HMAX))
    for i in range(N):
        fut = D[i, t:]; idx = np.where(fut >= Dfail)[0]
        if len(idx):
            j = idx[0]; T[i] = (j-1) + (Dfail-fut[j-1])/(fut[j]-fut[j-1]) if j > 0 else 0.0
    return D, T

def vasicek(x, m):
    x = np.sort(x); n = len(x); lo = np.clip(np.arange(n)-m, 0, n-1); hi = np.clip(np.arange(n)+m, 0, n-1)
    return np.mean(np.log(n/(2*m)*np.maximum(x[hi]-x[lo], 1e-12))) 

def run(gamma, D, T, rng, S=300):
    N = D.shape[0]; g = 1-gamma; R = g**2*s1**2 + s2**2
    ks = np.arange(1, t+1); m = vbar*ks
    H = D[:, 1:t+1] + s1*rng.standard_normal((N, t)); Ht = m + g*(H-m) + s2*rng.standard_normal((N, t))
    d = np.diff(Ht, axis=1); mon = np.mean(np.abs(np.sum(np.sign(d), axis=1))/(t-1))
    z = Ht - (1-g)*m
    Ks, Ps = kalman_cov(g, R, t); s = np.zeros((N, 2)); s[:, 1] = vbar
    for k in range(t):
        sp = s @ F.T; innov = z[:, k] - g*sp[:, 0]; s = sp + innov[:, None]*Ks[k][None, :]
    P = Ps[-1]; L = np.linalg.cholesky(P + 1e-12*np.eye(2))
    Dc = s[:, 0:1] + 0.0; samp = s[:, None, :] + (np.random.default_rng(rng.integers(1<<31)).standard_normal((N, S, 2)) @ L.T)
    Dd = samp[:, :, 0].copy(); Vv = samp[:, :, 1].copy(); Ts = np.full((N, S), float(HMAX)); alive = np.ones((N, S), bool)
    for k in range(1, HMAX+1):
        Dn = Dd + Vv + sB*rng.standard_normal((N, S)); Vv = Vv + sV*rng.standard_normal((N, S))
        hit = alive & (Dn >= Dfail)
        if hit.any():
            frac = (Dfail-Dd[hit])/np.maximum(Dn[hit]-Dd[hit], 1e-9); Ts[hit] = (k-1)+np.clip(frac, 0, 1); alive &= ~hit
        Dd = Dn
        if not alive.any(): break
    cens = float(np.mean(alive))
    That = Ts.mean(axis=1); var_post = Ts.var(axis=1)
    h = np.array([vasicek(Ts[i], 20) for i in range(N)]); Nbound = np.exp(2*h)/(2*np.pi*np.e)
    mse = float(np.mean((T-That)**2))
    return dict(gamma=gamma, mon=float(mon), mse=mse, rmse=float(np.sqrt(mse)), rmse_postvar=float(np.sqrt(var_post.mean())),
                rmse_bound=float(np.sqrt(Nbound.mean())), h_mean=float(h.mean()), cens=cens, viol=float(np.mean(Nbound > var_post*1.15)))

if __name__ == "__main__":
    gammas = [0.0, 0.5, 0.9, 0.97, 0.99, 0.995]; seeds = [11, 12, 13]; N = 400; res = {}
    Iraw = info_V(1.0, s1**2)[2]
    for gm in gammas:
        g = 1-gm; R = g**2*s1**2+s2**2; Ist, Ikin, Itot = info_V(g, R); res[gm] = dict(Istate=Ist, Ikin=Ikin, Itot=Itot, eta=Itot/Iraw, runs=[])
    for sd in seeds:
        rng = np.random.default_rng(sd); D, T = simulate(N, rng)
        for gm in gammas: res[gm]['runs'].append(run(gm, D, T, rng)); print(sd, gm, res[gm]['runs'][-1], flush=True)
    json.dump(dict(params=dict(vbar=vbar, sv0=sv0, sV=sV, sB=sB, s1=s1, s2=s2, Dfail=Dfail, t=t, N=N, seeds=seeds, Iraw=Iraw), res={str(k): v for k, v in res.items()}), open('/home/user/paper/results/exp1.json', 'w'), indent=1)
