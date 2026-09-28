"""Study 2 (docs/idea2.md): learned Gaussian priors on the ancestor-indicator LR.

Inputs. For every non-root tree node v (a code, or a group of codes such as
"E11" or "E116"), z_v = 1 if the admission has any code under v. A code
unseen in training still switches on its groups. logit P = b + z . w.

Models (sum log-loss + penalty; the intercept is never penalised):
  "rollup"  ancestor-indicator LR with one L2 penalty lam/2 ||w||^2 (the
            baseline; study 1's best model).
  "hier"    w_v ~ N(w_parent(v), sigma2_parent(v) / alpha), w_root = 0: each
            group's weight is centred on its parent group's weight, with one
            variance per parent group learned by empirical Bayes and a global
            strength alpha tuned on dev. No support term. An unseen node has no
            likelihood term, so its weight equals its parent's.
  "hier0"   [exploratory] w_v ~ N(0, sigma2_parent(v) / alpha): the same
            learned per-group variances, centred on 0 as in "rollup".
  "l2"      [reference] study 1's plain LR on the code indicators only.

"hier" is fit in the non-centred parametrisation d_v = w_v - w_parent(v),
w = T d (T sums d down each root path), so the penalty is diagonal. T is
applied implicitly, level by level, never materialised as Z @ T.

Speed (vs study 1's solver): trust-region Newton-CG in Jacobi-scaled
coordinates; the gradient tolerance scales with sqrt(n) (the stopping rule
bounds the objective gap per admission instead of in total); the
hyperparameter grid is walked strong -> weak with warm starts; empirical
Bayes runs once per training set with warm-started, early-stopped EM.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, log_expit

GTOL = 1e-3      # gradient-norm tolerance per sqrt(training admission)
CHUNK = 20000    # rows per block when forming Z @ T for the Jacobi diagonal


def indicators(X, A):
    """Binary 'any code under node v' matrix from leaf counts X and A = leaf -> nodes."""
    Z = (X @ A).tocsr()
    Z.data[:] = 1.0
    return Z


class Identity:
    T = None

    def fwd(self, d):
        return d

    adj = fwd


class PathSum:
    """(T d)_v = sum of d_u over u on the root path of v (non-root nodes, index v - 1)."""

    def __init__(self, hier):
        self.par = hier.parent[1:] - 1          # -1 = the root
        dep = hier.depth[1:]
        self.levels = [np.flatnonzero(dep == k) for k in range(2, dep.max() + 1)]
        V = len(self.par)
        rows, cols = [np.arange(V)], [np.arange(V)]
        u = self.par.copy()
        while (u >= 0).any():
            m = u >= 0
            rows.append(np.flatnonzero(m))
            cols.append(u[m])
            u = np.where(m, self.par[np.maximum(u, 0)], -1)
        r, c = np.concatenate(rows), np.concatenate(cols)
        self.T = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(V, V))
        self.V = V

    def fwd(self, d):
        w = d.copy()
        for L in self.levels:                  # parents are final before children
            w[L] += w[self.par[L]]
        return w

    def adj(self, g):
        s = g.copy()
        for L in reversed(self.levels):        # push subtree sums up one level
            s += np.bincount(self.par[L], s[L], minlength=self.V)
        return s


def colsq(Z, op, wts):
    """sum_i wts_i (Z T)_iu^2: the diagonal of the data curvature."""
    if op.T is None:
        return np.asarray(Z.multiply(Z).T @ wts).ravel()
    out = np.zeros(Z.shape[1])
    for a in range(0, Z.shape[0], CHUNK):
        M = (Z[a:a + CHUNK] @ op.T).tocsr()
        M.data **= 2
        out += M.T @ wts[a:a + CHUNK]
    return out


def fit(Z, y, op, prec, D0, x0, gtol=GTOL):
    """min_{d,b} sum_i softplus(z_i) - y_i z_i + 1/2 sum prec d^2, z = Z T d + b.
    D0: data-curvature diagonal at the prevalence (Jacobi scaling).
    Returns (d, b), iterations."""
    s = 1.0 / np.sqrt(prec + D0)
    pr = prec * s * s
    x0 = x0.copy()
    x0[:-1] /= s
    cache = {}

    def f(x):
        u, b = x[:-1], x[-1]
        z = Z @ op.fwd(u * s) + b
        p = expit(z)
        cache["W"] = p * (1 - p)
        r = p - y
        return ((-log_expit(-z) - y * z).sum() + 0.5 * (pr * u * u).sum(),
                np.append(s * op.adj(Z.T @ r) + pr * u, r.sum()))

    def hessp(x, v):
        # trust-ncg always evaluates f(x) before hessp(x, .)
        Wz = cache["W"] * (Z @ op.fwd(v[:-1] * s) + v[-1])
        return np.append(s * op.adj(Z.T @ Wz) + pr * v[:-1], Wz.sum())

    res = minimize(f, x0, jac=True, hessp=hessp, method="trust-ncg",
                   options={"gtol": gtol * np.sqrt(len(y)), "maxiter": 500})
    x = res.x.copy()
    x[:-1] *= s
    return x, res.nit


def em_sigma2(hier, q, seen, nu0):
    """Per-parent-group variance of its children's d (alpha = 1 units) from
    E[d_v^2] = q_v, over children with training support, shrunk towards the
    pooled value for the parent's depth with nu0 pseudo-children."""
    par = hier.parent[1:]
    num = np.bincount(par, q * seen, hier.n_nodes)
    cnt = np.bincount(par, seen.astype(float), hier.n_nodes)
    dep = hier.depth
    lvl = np.bincount(dep, num) / np.maximum(np.bincount(dep, cnt), 1)
    lvl[lvl == 0] = lvl[lvl > 0].mean()
    return (num + nu0 * lvl[dep]) / (cnt + nu0)


class GroupedLR:
    """m = GroupedLR(hier, kind).setup(X, y); m.fit(hp, x0); m.decision(X)."""

    def __init__(self, hier, kind, n_em=10, nu0=3.0, sigma2_init=0.01, em_tol=0.01):
        self.h, self.kind = hier, kind
        self.n_em, self.nu0, self.sigma2_init, self.em_tol = n_em, nu0, sigma2_init, em_tol
        self.A = hier.ancestors_matrix()[:, 1:].tocsr()
        self.op = PathSum(hier) if kind == "hier" else Identity()

    def features(self, X):
        return X if self.kind == "l2" else indicators(X, self.A)

    def setup(self, X, y):
        self.Z = self.features(X)
        self.y = y
        p0 = y.mean()
        self.b0 = np.log(p0 / (1 - p0))
        self.D0 = colsq(self.Z, self.op, np.full(len(y), p0 * (1 - p0)))
        self.support = np.asarray(self.Z.sum(0)).ravel()
        self.x = np.append(np.zeros(self.Z.shape[1]), self.b0)
        self.nit = 0
        if self.kind in ("hier", "hier0"):
            self.empirical_bayes()
        return self

    def prec(self, hp):
        if self.kind in ("l2", "rollup"):
            return np.full(self.Z.shape[1], float(hp))
        return hp / self.sigma2[self.h.parent[1:]]

    def empirical_bayes(self):
        """EM for the per-group variances at alpha = 1 (diagonal Laplace posterior),
        from sigma2 = sigma2_init (~ the dev-optimal rollup penalty), stopped when
        the mean |log change| over groups drops below em_tol or after n_em steps.
        EM is slow for weakly identified groups, which stay near the init /
        their depth's pooled value; the evidence fixed point (MacKay) instead
        collapses the deep levels to zero variance and did worse on dev."""
        seen = self.support > 0
        self.sigma2 = np.full(self.h.n_nodes, self.sigma2_init)
        self.em_trace = []
        for it in range(self.n_em):
            prec = self.prec(1.0)
            self.x, nit = fit(self.Z, self.y, self.op, prec, self.D0, self.x)
            self.nit += nit
            d = self.x[:-1]
            p = expit(self.Z @ self.op.fwd(d) + self.x[-1])
            H = colsq(self.Z, self.op, p * (1 - p))
            new = em_sigma2(self.h, d * d + 1.0 / (prec + H), seen, self.nu0)
            groups = np.bincount(self.h.parent[1:][seen], minlength=self.h.n_nodes) > 0
            change = np.abs(np.log(new / self.sigma2))[groups].mean()
            self.sigma2 = new
            self.em_trace.append(float(change))
            if change < self.em_tol:
                break
        self.x_em = self.x.copy()

    def fit(self, hp, x0=None):
        self.hp = hp
        self.x, nit = fit(self.Z, self.y, self.op, self.prec(hp), self.D0,
                          self.x if x0 is None else x0)
        self.nit += nit
        self.w = self.op.fwd(self.x[:-1])    # per-feature weight
        self.b = self.x[-1]
        return self

    def decision(self, X, zero_unseen=False):
        w = self.w
        if zero_unseen:
            w = np.where(self.support > 0, w, 0.0)
        return self.features(X) @ w + self.b

    def leaf_weight(self):
        """Effective log-odds shift of each code alone (its indicator plus its groups')."""
        return self.w if self.kind == "l2" else self.A @ self.w
