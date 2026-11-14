"""Physics-guided information bottleneck (PGIB) encoders: learned health indicators with a Wiener first-hitting-time likelihood.

Requires PyTorch (pip install torch).  The encoder maps the sensor history X_{1:k} to the increments of an HI path,
H_k = H_{k-1} + u_k + s_k eps_k  (PGIB, positive velocity u_k),  or, with an observation correction (PGIB-KF),
H_k = (1 - w_k)(H_{k-1} + u_k + s_k eps_k) + w_k l_k  with a learned gain w_k and level estimate l_k (Section 4.4 of the paper).
The loss is the negative log-likelihood of the observed failure time under the mixed inverse-Gaussian first-hitting distribution
plus beta times the path-space KL divergence to a physics prior (Appendix B).

Use pgib_constructor(...) to obtain a function that plugs into infobudget.evaluate(units, hi=pgib_constructor(...)).
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("infobudget.models requires PyTorch: pip install torch") from e

from .indicators import _affine

DT = torch.float64


def window_stats(H, sm: int = 5):
    """Smoothed level, drift estimate and increment variance over a trailing window of max(10, 0.25 t) steps."""
    m = H.shape[0]
    Hp = torch.cat([H[:1].expand(sm - 1), H]); Hs = F.avg_pool1d(Hp[None, None], sm, 1)[0, 0]
    dH = torch.cat([H.new_zeros(1), H[1:] - H[:-1]]); c1 = torch.cumsum(dH, 0); c2 = torch.cumsum(dH ** 2, 0)
    t = torch.arange(m); w = torch.clamp((0.25 * t).long(), min=10); t0 = torch.clamp(t - w, min=0); wd = (t - t0).clamp(min=1).to(H.dtype)
    mu = (Hs - Hs[t0]) / wd; s1 = c1 - c1[t0]; s2 = c2 - c2[t0]; var = ((s2 - s1 ** 2 / wd) / (wd - 1).clamp(min=1)).clamp(min=1e-12)
    return Hs, mu, var, wd


def first_hitting_logpdf(tau, a, mu, s2, sm2):
    """Log density at remaining life tau of the first passage of a Wiener process with drift mu ~ N(mu, sm2), diffusion s2 and distance a."""
    v = s2 + sm2 * tau
    return torch.log(a) - 0.5 * torch.log(2 * np.pi * tau ** 3 * v) - (a - mu * tau) ** 2 / (2 * tau * v)


def first_hitting_nll(H, sB2, epistemic: bool = True):
    """Mean negative log-likelihood of the true remaining life along an HI path that ends at failure (threshold 1)."""
    Hs, mu, var, wd = window_stats(H); m = len(H); idx = np.arange(int(0.2 * m), int(0.92 * m) + 1)
    s2 = sB2 * torch.ones_like(var); sm2 = s2 / wd if epistemic else torch.zeros_like(s2)
    tau = torch.tensor((m - 1 - idx).astype(float), dtype=H.dtype).clamp(min=1.0); a = (1 - Hs[idx]).clamp(min=0.02)
    return -first_hitting_logpdf(tau, a, mu[idx], s2[idx], sm2[idx]).mean()


class PGIB(nn.Module):
    def __init__(self, d, physics=True, hid=16, arch="win"):
        super().__init__()
        self.arch, self.physics = arch, physics
        self.fu, self.fs = nn.Linear(hid, 1), nn.Linear(hid, 1)
        if arch == "gru":
            self.gru = nn.GRU(d, hid, batch_first=True)
        else:
            self.mlp = nn.Sequential(nn.Linear(2 * d, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        self.lt1 = nn.Parameter(torch.tensor(np.log(0.01))); self.t2 = nn.Parameter(torch.tensor(0.3)); self.lsB = nn.Parameter(torch.tensor(np.log(0.01)))

    def features(self, X):
        if self.arch == "gru":
            o, _ = self.gru(X[None]); return o[0]
        Xp = torch.cat([X[:1].expand(2, -1), X]); xm = (Xp[2:] + Xp[1:-1] + Xp[:-2]) / 3; Xq = torch.cat([xm[:1].expand(8, -1), xm])
        return self.mlp(torch.cat([xm, xm - Xq[:-8]], 1))

    def heads(self, X):
        o = self.features(X)
        return 0.02 * F.softplus(self.fu(o))[:, 0] + 1e-4, 0.01 * F.softplus(self.fs(o))[:, 0] + 1e-4

    def path(self, X, noise=True):
        u, sd = self.heads(X); eps = torch.randn_like(u) if noise else torch.zeros_like(u)
        return torch.cumsum(u + sd * eps, 0), u, sd

    def kl(self, H, u, sd):
        """Path-space KL divergence of the encoder to the physics prior (Proposition 6 of the paper), averaged over steps."""
        Hp = torch.cat([H.new_zeros(1), H[:-1]]); sB2 = torch.exp(2 * self.lsB)
        eta = torch.exp(self.lt1) * torch.exp(torch.clamp(self.t2 * Hp, -3, 3)) if self.physics else torch.zeros_like(u)
        r = sd ** 2 / sB2
        return (0.5 * (r - 1 - torch.log(r) + (u - eta) ** 2 / sB2)).mean()


class PGIBKF(PGIB):
    """PGIB with an observation-corrected level: a learned gain w_k blends the integrated path with a level estimate l_k from the sensors."""

    def __init__(self, d, physics=True, hid=16):
        super().__init__(d, physics, hid, arch="win")
        self.fl, self.fk = nn.Linear(hid, 1), nn.Linear(hid, 1)
        with torch.no_grad():
            self.fk.bias.fill_(-2.0); self.fl.weight.mul_(0.1); self.fl.bias.fill_(0.0)

    def path(self, X, noise=True):
        o = self.features(X)
        u = 0.02 * F.softplus(self.fu(o))[:, 0] + 1e-4; sd = 0.01 * F.softplus(self.fs(o))[:, 0] + 1e-4
        l = self.fl(o)[:, 0]; kap = torch.sigmoid(self.fk(o))[:, 0]; eps = torch.randn_like(u) if noise else torch.zeros_like(u)
        Hp = u.new_zeros(()); Hs, ue, se = [], [], []
        for k in range(len(u)):
            Hn = (1 - kap[k]) * (Hp + u[k] + sd[k] * eps[k]) + kap[k] * l[k]
            ue.append((1 - kap[k]) * u[k] + kap[k] * (l[k] - Hp)); se.append((1 - kap[k]) * sd[k] + 1e-4); Hs.append(Hn); Hp = Hn
        return torch.stack(Hs), torch.stack(ue), torch.stack(se)


def train_pgib(Xl, beta: float = 0.3, arch: str = "kf", steps: int = 900, seed: int = 0, physics: bool = True, lr: float = 1e-2):
    """Train an encoder on standardised training units Xl (list of arrays (T_u, d)) that run to failure.

    arch: 'kf' (PGIB-KF), 'win' (PGIB with a causal feed-forward encoder) or 'gru' (PGIB with a recurrent encoder).
    beta weights the compression term; beta = 0 removes the bottleneck.  Returns the trained module.
    """
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    d = Xl[0].shape[1]
    mod = (PGIBKF(d, physics) if arch == "kf" else PGIB(d, physics, arch=arch)).to(DT)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-3)
    Xt = [torch.tensor(np.asarray(x, float), dtype=DT) for x in Xl]
    for _ in range(steps):
        X = Xt[rng.integers(len(Xt))]
        opt.zero_grad(); H, u, sd = mod.path(X, True)
        loss = first_hitting_nll(H, torch.exp(2 * mod.lsB)) + beta * mod.kl(H, u, sd) + 10 * (H[-1] - 1) ** 2
        loss.backward(); nn.utils.clip_grad_norm_(mod.parameters(), 5.0); opt.step()
    return mod


def encode(mod, X):
    """HI trajectory (noise-free path) of one unit."""
    with torch.no_grad():
        return mod.path(torch.tensor(np.asarray(X, float), dtype=DT), False)[0].numpy()


def pgib_constructor(beta: float = 0.3, arch: str = "kf", steps: int = 900, seed: int = 0, physics: bool = True):
    """Return fit(list_of_training_arrays) -> transform(X), for infobudget.evaluate(units, hi=pgib_constructor(...)).

    The HI of each unit is mapped affinely to 0 (start) and 1 (end of life) with the training units, as for the other HIs.
    """
    def fit(Xl):
        mod = train_pgib(Xl, beta, arch, steps, seed, physics)
        aff = _affine([encode(mod, x) for x in Xl])
        return lambda X: aff(encode(mod, X))
    return fit
