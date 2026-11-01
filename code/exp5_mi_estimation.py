"""Exp 5: reliability of the sample-based estimate of the information efficiency eta = I(T;H)/I(T;X) with the KSG estimator (Kraskov et al. 2004, algorithm 1),
in a Gaussian model with known mutual information.  X ~ N(0, I_d) (d = 12 sensor features), T = w'X + s_T*e,  H = u'X + s_H*e' (a scalar HI).
Run from code/:  python3 exp5_mi_estimation.py   (writes ../results/exp5_mi.json and ../results/tab_exp5.tex)"""
import numpy as np, json
from scipy.spatial import cKDTree
from scipy.special import digamma
def ksg(x, y, k=5):
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x; y = np.atleast_2d(y.T).T if y.ndim == 1 else y; n = len(x); z = np.hstack([x, y])
    d, _ = cKDTree(z).query(z, k=k+1, p=np.inf); eps = d[:, -1]
    tx, ty = cKDTree(x), cKDTree(y)
    nx = np.array([len(tx.query_ball_point(x[i], eps[i]-1e-12, p=np.inf))-1 for i in range(n)]); ny = np.array([len(ty.query_ball_point(y[i], eps[i]-1e-12, p=np.inf))-1 for i in range(n)])
    return digamma(k)+digamma(n)-np.mean(digamma(nx+1)+digamma(ny+1))
d = 12; rng = np.random.default_rng(0); w = np.ones(d); sT = np.linalg.norm(w)/np.sqrt(np.e**2-1)     # I(T;X) = 1 nat
I_TX = 0.5*np.log(1+w@w/sT**2); u = w/np.linalg.norm(w)
rows = []
for eta_target in (0.7,):
    # choose s_H so that I(T;H) = eta_target * I(T;X)
    from scipy.optimize import brentq
    f = lambda sH: -0.5*np.log(1-((w@u)/np.sqrt((w@w+sT**2)*(u@u+sH**2)))**2)-eta_target*I_TX
    sH = brentq(f, 1e-3, 50); I_TH = -0.5*np.log(1-((w@u)/np.sqrt((w@w+sT**2)*(u@u+sH**2)))**2)
    for n in (100, 300, 1000, 3000):
        est = []
        for rep in range(100):
            X = rng.standard_normal((n, d)); T = X@w+sT*rng.standard_normal(n); H = X@u+sH*rng.standard_normal(n)
            a = ksg(X, T[:, None]); b = ksg(H[:, None], T[:, None]); est.append((a, b))
        est = np.array(est); eta_hat = est[:, 1]/est[:, 0]
        rows.append(dict(n=n, I_TX=float(I_TX), I_TH=float(I_TH), eta=float(I_TH/I_TX), I_TX_hat=float(est[:, 0].mean()), I_TH_hat=float(est[:, 1].mean()), eta_hat_mean=float(eta_hat.mean()), eta_hat_sd=float(eta_hat.std()), frac_eta_gt1=float((eta_hat > 1).mean())))
        print(rows[-1], flush=True)
json.dump(rows, open('../results/exp5_mi.json', 'w'), indent=1)
with open('../results/tab_exp5.tex', 'w') as fo:
    fo.write("\\begin{tabular}{r c c c c}\n\\hline\n$n$ & $\\hat I(T;\\mathbf X)$ & $\\hat I(T;H)$ & $\\hat\\eta_{\\rm info}$ (mean $\\pm$ standard deviation) & fraction $\\hat\\eta_{\\rm info}>1$ \\\\\n\\hline\n".replace("& $\\hat I(T;H)$ & $\\hat\\eta","& $\\hat I(T;H)$ & $\\hat\\eta"))
    for r in rows: fo.write(f"{r['n']} & {r['I_TX_hat']:.2f} & {r['I_TH_hat']:.2f} & ${r['eta_hat_mean']:.2f}\\pm{r['eta_hat_sd']:.2f}$ & {r['frac_eta_gt1']:.2f} \\\\\n")
    fo.write(f"\\hline\n\\multicolumn{{5}}{{l}}{{true values: $I(T;\\mathbf X)={I_TX:.2f}$, $I(T;H)={rows[0]['I_TH']:.2f}$, $\\eta_{{\\rm info}}={rows[0]['eta']:.2f}$ (nats)}}\\\\\n\\end{{tabular}}\n")
