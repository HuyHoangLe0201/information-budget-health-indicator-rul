"""Exp 2: physics-constrained RUL bound (constant-drift Wiener) + mixed-IG density check.
Setup: D_s = mu*s + sB*W_s, mu ~ N(mu0, smu^2). Threshold D_fail. Observe D_{1:t} (state D_t known => a=Dfail-D_t).
Information-lossy HI: H' = mu_hat_mle + N(0, tau^2)  => posterior of mu is Gaussian => all quantities exact."""
import numpy as np, json
from scipy import integrate, stats
rng = np.random.default_rng(2026)
mu0, smu, sB, Dfail, t = 1.0, 0.22, 0.85, 100.0, 50
# --- (i) FHT density with Gaussian drift: closed form vs numerical mixing ---
def ig_pdf(tau, a, mu, s): return a/(s*np.sqrt(2*np.pi*tau**3))*np.exp(-(a-mu*tau)**2/(2*s**2*tau))
def mixed_pdf(tau, a, m, sm, s): return a/np.sqrt(2*np.pi*tau**3*(s**2+sm**2*tau))*np.exp(-(a-m*tau)**2/(2*tau*(s**2+sm**2*tau)))
a = 50.0; maxerr = 0
for tau in [10, 25, 50, 80, 150]:
    num = integrate.quad(lambda mu: ig_pdf(tau, a, mu, sB)*stats.norm.pdf(mu, mu0, smu), -3, 6, limit=200)[0]
    maxerr = max(maxerr, abs(num-mixed_pdf(tau, a, mu0, smu, sB))/mixed_pdf(tau, a, mu0, smu, sB))
tt = np.linspace(0.5, 4000, 400000); mass = np.trapezoid(mixed_pdf(tt, a, mu0, smu, sB), tt)
# --- (ii) bound vs Bayes-optimal MSE as the HI loses information ---
N = 100000; mu = rng.normal(mu0, smu, 4*N); mu = mu[mu > 0.3][:N]  # prior truncated at 0.3 (3.2 sd, mass 7e-4)
# full-data posterior of mu (Gaussian conjugate; sufficient stat D_t/t)
prec_x = 1/smu**2 + t/sB**2; var_x = 1/prec_x
I_mu_X = 0.5*np.log(smu**2/var_x)
rows = []
for eta in [1.0, 0.9, 0.75, 0.5, 0.3, 0.15]:
    # HI keeps fraction eta of I(mu;X): I(mu;H)=0.5 ln(smu^2/var_h) => var_h = smu^2 exp(-2 eta I)
    var_h = smu**2*np.exp(-2*eta*I_mu_X)
    # construct H: Gaussian channel with posterior var var_h  (mean m_h = shrunk MLE)
    # observation y = mu + n, n~N(0,r) with 1/var_h = 1/smu^2 + 1/r
    r = 1/(1/var_h - 1/smu**2) if var_h < smu**2 else 1e18
    y = mu + np.sqrt(r)*rng.standard_normal(N) if r < 1e17 else np.full(N, mu0)
    m_h = var_h*(mu0/smu**2 + y/r) if r < 1e17 else np.full(N, mu0)
    # NOTE: this HI is derived from mu directly (the lossless limit eta=1 matches the data posterior only
    # up to the Gaussian channel); known threshold gap a is taken as given (conditioning on D_t=d, a=50)
    a_ = 50.0
    # Posterior draws of mu | H, for each unit: Bayes-optimal T_hat = E[a/mu | H] via Gauss-Hermite
    gh_x = np.linspace(-5, 5, 101); gh_w0 = stats.norm.pdf(gh_x); gh_w0 /= gh_w0.sum()
    mus = m_h[:, None] + np.sqrt(var_h)*gh_x[None, :]
    W = np.where(mus > 0.3, gh_w0[None, :], 0.0); W /= W.sum(1, keepdims=True); mus = np.maximum(mus, 0.3)
    Tm = (W*(a_/mus)).sum(1)                     # E[T|H]
    var_epi = (W*((a_/mus) - Tm[:, None])**2).sum(1)  # Var(a/mu | H)
    ale = (W*(a_*sB**2/mus**3)).sum(1)           # E[a sB^2/mu^3 | H]
    # exact entropy power of m=a/mu given H: h(m|H)=h(mu|H)+E ln(a/mu^2)
    hmu = 0.5*np.log(2*np.pi*np.e*var_h); Elog = (W*np.log(a_/mus**2)).sum(1)
    N_epi = np.exp(2*(hmu+Elog))/(2*np.pi*np.e)
    # empirical MSE of Bayes estimator vs realised T (sample T ~ IG(a/mu, a^2/sB^2))
    Ts = stats.invgauss.rvs(mu=(sB**2/(mu*a_)), scale=a_**2/sB**2, size=N, random_state=7)  # IG(mean=a/mu, shape=a^2/sB^2)
    mse = np.mean((Ts-Tm)**2)
    # delta-method form of Thm 3 (closed form in terms of eta)
    delta_epi = np.mean(a_**2/mu**4)*var_x*np.exp(2*(1-eta)*I_mu_X)  # (a^2/mu^4)*J^{-1}*exp(2(1-eta)I)
    rows.append(dict(eta=eta, ale=float(ale.mean()), epi_exact=float(var_epi.mean()), epi_entropy_bound=float(N_epi.mean()),
                     epi_delta=float(delta_epi), bound_total=float(ale.mean()+N_epi.mean()), mse_bayes=float(mse),
                     rmse_bound=float(np.sqrt(ale.mean()+N_epi.mean())), rmse_bayes=float(np.sqrt(mse))))
    print(rows[-1])
json.dump(dict(ig_mix_max_rel_err=float(maxerr), ig_mix_mass=float(mass), I_mu_X=float(I_mu_X), rows=rows), open('/home/user/paper/results/exp2.json', 'w'), indent=1)
print('mix rel err', maxerr, 'mass', mass, 'I(mu;X)', I_mu_X)
