"""Logistic regression with (a) an L2 prior, (b) L2 on leaf + ancestor
indicators ("rollup"), and (c) the hierarchical prior from docs/idea.md.

All objectives are  sum-NLL + penalty  minimised with L-BFGS; the intercept
is never penalised.

Hierarchical prior. Every tree node v has a parameter theta_v; a leaf's theta
is the code's LR weight, an internal node's theta is the latent mean of its
children (the root is fixed at 0). The prior is

    theta_v ~ N(theta_parent(v), sigma2_parent(v) * (1 + n_v)**beta / alpha)

so a code's related codes (its siblings, through their shared parent mean)
act as a Gaussian prior on its weight, with precision
alpha / (sigma2_g * (1 + n_v)**beta): inversely proportional to the variance
of that Gaussian and (beta = 1, as specified) to the code's own support n_v.

sigma2_g is estimated by empirical Bayes (EM with a diagonal Laplace
posterior variance, shrunk towards the pooled estimate for that depth) at
alpha = 1; alpha is then a global strength multiplier tuned on dev.
A leaf with no training support has no likelihood term, so its fitted weight
is exactly its parent's latent mean -- the prediction the baseline cannot
make.

The fit is done in the non-centred parametrisation delta_v = theta_v -
theta_parent(v) (leaf weight = sum of deltas on its root path, i.e. the
rollup design X @ A with a per-node diagonal penalty), with a Jacobi
rescaling of each delta; the centred form is badly conditioned.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, log_expit

LBFGS = {"maxiter": 5000, "gtol": 1e-6, "ftol": 1e-12}


def _nll_grad(X, y, w, b):
    z = X @ w + b
    loss = -(y * log_expit(z) + (1 - y) * log_expit(-z)).sum()
    r = expit(z) - y
    return loss, X.T @ r, r.sum()


class L2LR:
    """Plain LR on leaf indicators, penalty lam/2 ||w||^2."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y, x0=None):
        if x0 is None:
            x0 = np.zeros(X.shape[1] + 1)
            x0[-1] = np.log(y.mean() / (1 - y.mean()))

        def f(x):
            w, b = x[:-1], x[-1]
            l, gw, gb = _nll_grad(X, y, w, b)
            return l + 0.5 * self.lam * w @ w, np.append(gw + self.lam * w, gb)

        self.x = minimize(f, x0, jac=True, method="L-BFGS-B", options=LBFGS).x
        self.w, self.b = self.x[:-1], self.x[-1]
        return self

    def decision(self, X):
        return X @ self.w + self.b


class RollupLR(L2LR):
    """L2 LR on [leaf indicators, 'any code under node v' indicators]."""

    def __init__(self, hier, lam=1.0):
        super().__init__(lam)
        self.A = hier.ancestors_matrix()[:, 1:].tocsr()

    def expand(self, X):
        Z = (X @ self.A).tocsr()
        Z.data[:] = 1.0
        return Z

    def fit(self, X, y, x0=None):
        return super().fit(self.expand(X), y, x0)

    def decision(self, X):
        return super().decision(self.expand(X))


def _fit_deltas(M, k, wts, b_free, b0, prec, x0=None):
    """min_{d,b}  sum_i wts_i softplus(z_i) - k_i z_i  + 1/2 sum prec d^2,
    z = M d + b.  Returns delta, b, diag data curvature of delta, raw x."""
    p0 = k.sum() / wts.sum()
    M2 = M.multiply(M).tocsr()
    Hd = np.asarray(M2.T @ (wts * p0 * (1 - p0))).ravel()
    s = 1.0 / np.sqrt(prec + Hd)
    Ms = (M @ sp.diags(s)).tocsr()
    MsT = Ms.T.tocsr()
    pr = prec * s * s
    if x0 is None:
        x0 = np.zeros(M.shape[1] + 1)
        x0[-1] = b0
    else:
        x0 = x0.copy()
        x0[:-1] /= s

    cache = {}

    def f(x):
        u, b = x[:-1], (x[-1] if b_free else b0)
        z = Ms @ u + b
        l = (-wts * log_expit(-z) - k * z).sum()   # softplus(z) = -log_expit(-z)
        p = expit(z)
        cache["W"] = wts * p * (1 - p)
        r = wts * p - k
        return (l + 0.5 * (pr * u * u).sum(),
                np.append(MsT @ r + pr * u, r.sum() if b_free else 0.0))

    def hessp(x, v):
        # Newton-CG always evaluates f(x) before hessp(x, .)
        W = cache["W"]
        u, vb = v[:-1], (v[-1] if b_free else 0.0)
        Wz = W * (Ms @ u + vb)
        return np.append(MsT @ Wz + pr * u, Wz.sum() if b_free else 0.0)

    x = minimize(f, x0, jac=True, hessp=hessp, method="trust-ncg",
                 options={"gtol": 1e-5, "maxiter": 500}).x
    d, b = x[:-1] * s, (x[-1] if b_free else b0)
    p = expit(M @ d + b)
    Hd = np.asarray(M2.T @ (wts * p * (1 - p))).ravel()
    return d, b, Hd, np.append(d, b)


def _em_sigma2(hier, delta, vpost, n, beta, nu0):
    """EM update of each parent's child variance (alpha = 1 units), shrunk
    towards the pooled value for the parent's depth with nu0 pseudo-children."""
    par = hier.parent[1:]
    q = (delta ** 2 + vpost) / (1.0 + n) ** beta
    num = np.bincount(par, q, hier.n_nodes)
    cnt = np.bincount(par, None, hier.n_nodes).astype(float)
    dep = hier.depth
    lvl = np.bincount(dep, num) / np.maximum(np.bincount(dep, cnt), 1)
    return (num + nu0 * lvl[dep]) / (cnt + nu0)


def _node_support(Mcounts):
    Z = Mcounts.copy()
    Z.data[:] = 1.0
    return np.asarray(Z.sum(0)).ravel()


class HierLR:
    """Usage: m = HierLR(hier).fit_em(X, y); m.fit(alpha); m.decision(X)."""

    def __init__(self, hier, beta=1.0, n_em=6, nu0=3.0):
        self.h, self.beta, self.n_em, self.nu0 = hier, beta, n_em, nu0
        self.A = hier.ancestors_matrix()[:, 1:].tocsr()  # leaf -> own node + ancestors
        self.par = hier.parent[1:]

    def _prec(self, alpha):
        return alpha / (self.sigma2[self.par] * (1.0 + self.n) ** self.beta)

    def fit_em(self, X, y):
        self.M = (X @ self.A).tocsr()   # rows x non-root nodes: # leaves present under node
        self.n = _node_support(self.M)
        self.y = y
        self.b0 = np.log(y.mean() / (1 - y.mean()))
        self.sigma2 = np.ones(self.h.n_nodes)
        x0 = None
        for _ in range(self.n_em):
            prec = self._prec(1.0)
            d, _, Hd, x0 = _fit_deltas(self.M, y, np.ones_like(y), True, self.b0, prec, x0)
            self.sigma2 = _em_sigma2(self.h, d, 1.0 / (prec + Hd), self.n, self.beta, self.nu0)
        self._x_em = x0
        return self

    def fit(self, alpha):
        d, self.b, _, self.x = _fit_deltas(self.M, self.y, np.ones_like(self.y), True,
                                           self.b0, self._prec(alpha), self._x_em)
        self.alpha, self.delta = alpha, d
        self.w = self.A @ d
        return self

    def node_theta(self):
        """theta of every node (root = 0): sum of deltas on its path."""
        th = np.zeros(self.h.n_nodes)
        for v in range(1, self.h.n_nodes):  # parents precede children
            th[v] = th[self.h.parent[v]] + self.delta[v - 1]
        return th

    def decision(self, X):
        return X @ self.w + self.b


class HierUni:
    """One-input-feature regime: code c's weight is its own univariate
    log-odds shift  logit P(y | c present) = b0 + w_c  (b0 = logit of train
    prevalence), with the same hierarchical prior tying the codes together.
    pool=False: unpooled per-code estimates (weak ridge); unseen codes -> 0,
    i.e. the prevalence prior."""

    def __init__(self, hier, beta=1.0, n_em=6, nu0=3.0, pool=True):
        self.h, self.beta, self.n_em, self.nu0, self.pool = hier, beta, n_em, nu0, pool
        self.A = hier.ancestors_matrix()[:, 1:].tocsr()
        self.par = hier.parent[1:]

    def fit(self, X, y, alpha=1.0):
        n_leaf = np.asarray(X.sum(0)).ravel()
        k_leaf = np.asarray(X.T @ y).ravel()
        b0 = self.b0 = np.log(y.mean() / (1 - y.mean()))
        if not self.pool:  # leaf-only, unit-precision ridge on each log-odds shift
            I = sp.identity(len(n_leaf), format="csr")
            w, *_ = _fit_deltas(I, k_leaf, n_leaf, False, b0, np.ones(len(n_leaf)))
            self.w = w
            return self
        n = self.n = _node_support((X @ self.A).tocsr())
        self.k_leaf, self.n_leaf = k_leaf, n_leaf
        sigma2 = np.ones(self.h.n_nodes)
        x0 = None
        for _ in range(self.n_em):
            prec = 1.0 / (sigma2[self.par] * (1.0 + n) ** self.beta)
            d, _, Hd, x0 = _fit_deltas(self.A, k_leaf, n_leaf, False, b0, prec, x0)
            sigma2 = _em_sigma2(self.h, d, 1.0 / (prec + Hd), n, self.beta, self.nu0)
        self.sigma2, self._x_em = sigma2, x0
        return self.with_alpha(alpha)

    def with_alpha(self, alpha):
        """Refit at another strength, reusing the EM variances."""
        d, *_ = _fit_deltas(self.A, self.k_leaf, self.n_leaf, False, self.b0,
                            alpha / (self.sigma2[self.par] * (1.0 + self.n) ** self.beta), self._x_em)
        self.w = self.A @ d
        return self
