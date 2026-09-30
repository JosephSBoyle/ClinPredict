"""Study 3: one-code mode. Is the weight estimated from a code's relatives a
good prior, compared with learning the code's own weight from its own data?

One-code model. An observation is one (admission, code) pair: we see a single
code c and update the prevalence by that code's weight,

    logit P(readmit | c) = b0 + w_c,        b0 = logit(training prevalence).

Two modes:
  any      every code occurrence is an observation (an admission with 13 codes
           gives 13 pairs, each scored on its own)
  primary  only the primary diagnosis (seq_num = 1): exactly one input per
           admission, so the pairs are the admissions

Estimators of w_c, all from the training subset T:
  null     0 (predict the prevalence)
  own_mle  the code's own log-odds shift, log((k+.5)/(n-k+.5)) - b0
  own      the code's own data with a zero-centred prior N(0, s2) (one-feature
           ridge; s2 fitted by marginal likelihood over all codes): the fair
           version of "learn the true code"
  sib      inverse-variance-weighted mean of the siblings' own_mle (the other
           leaves under parent(c) with training data; none -> the grandparent's
           other leaves, and so on up). Used as is, unshrunk.
  tree     leave-c-out predictive mean of a Gaussian hierarchy over the prefix
           tree: w_s ~ N(mu_parent, s2_leaf[depth]),
           mu_a ~ N(mu_parent(a), s2_node[depth]), mu_root ~ N(0, 1).
           Each code's binomial likelihood enters as a Gaussian site (Laplace
           at the code's posterior mode, iterated to a fixed point); the 11
           variances maximise the marginal likelihood. One upward and one
           downward pass of Gaussian message passing give, for every code at
           once, N(m_c, P_c): the distribution of w_c given every other code's
           data but not its own. Siblings dominate when they are well
           supported; otherwise the estimate falls back smoothly towards the
           cousins and the chapter.
  post     tree prior N(m_c, P_c) updated by the code's exact binomial
           likelihood (MAP)

Chapter Z (factors influencing health status, encounters for aftercare,
weeks of gestation, ...) is excluded everywhere: Z codes are neither scored nor
used as anyone's relatives, and in primary mode admissions with a Z primary
diagnosis drop out. b0 is the prevalence over the kept training pairs.

Score on an evaluation set E (train T itself, or dev; test is not touched):
  ll       log-loss per pair (nats); brier: Brier score per pair
  auc      AUROC over the pairs, each scored by its code's w (in primary mode
           the pairs are the admissions, so this is the admission-level AUROC)
  gain     log-likelihood gain over null, per 1000 pairs of E (millinats)
  ceiling  gain of the best constant per code fitted on E itself
           (= sum_c n_c KL(Bern(k_c/n_c) || Bern(p0)))
Groups: all codes, codes seen / unseen in training, and training-support bins.
Headline ratio rho = gain(tree) / gain(own); the hypothesis is rho >= 1/3.
On train, own is fitted in-sample; the priors never use c's own data except
through the 11 shared variances and the siblings' Laplace expansion points.

  .venv/Scripts/python -m hierprior.onecode
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, log_expit, xlogy
from sklearn.metrics import roc_auc_score

from hierprior.data import ROOT, build_cohort, load
from hierprior.experiment2 import FRACS, SEEDS, splits

OUT = Path("output/hier3")
ESTIMATORS = ["null", "sib", "tree", "own_mle", "own", "post"]
BINS = [(1, 10), (10, 100), (100, 1000), (1000, 10**9)]
ROOT_VAR = 1.0


def primary_design(hier):
    """Admission x leaf CSR with only the primary (seq_num = 1) diagnosis."""
    coh, _ = build_cohort()
    d = (pl.read_csv(ROOT / "diagnoses_icd.csv.gz",
                     schema_overrides={"icd_code": pl.Utf8, "icd_version": pl.Int8})
         .filter((pl.col("icd_version") == 10) & (pl.col("seq_num") == 1))
         .select("hadm_id", pl.col("icd_code").str.strip_chars()))
    row = {h: i for i, h in enumerate(coh["hadm_id"].to_list())}
    d = d.filter(pl.col("hadm_id").is_in(list(row)))
    r = np.array([row[h] for h in d["hadm_id"].to_list()])
    c = np.array([hier.col[x] for x in d["icd_code"].to_list()])
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(coh), len(hier.leaves)))


def counts(X, y, rows):
    Xr = X[rows]
    return np.asarray(Xr.sum(0)).ravel(), np.asarray(Xr.T @ y[rows]).ravel()


def own(n, k, b0):
    w = np.where(n > 0, np.log((k + .5) / (n - k + .5)) - b0, 0.0)
    v = np.where(n > 0, 1 / (k + .5) + 1 / (n - k + .5), np.inf)
    return w, v


def sibling_ivw(hier, A, w, v, sup):
    """Leave-c-out inverse-variance-weighted mean of the supported leaves under
    c's nearest ancestor that has any besides c."""
    u = np.where(sup, 1 / v, 0.0)
    K, S0, S1 = A.T @ sup.astype(float), A.T @ u, A.T @ (u * w)
    out = np.zeros(len(w))
    for j, node in enumerate(hier.leaf_node):
        a = hier.parent[node]
        while a > 0 and K[a] - sup[j] < 1:
            a = hier.parent[a]
        if a > 0:
            out[j] = (S1[a] - u[j] * w[j]) / (S0[a] - u[j])
    return out


class GaussTree:
    """Gaussian hierarchy over a code tree, fitted to per-code estimates (w, v).
    theta = log variances, one per variance group. A tree may set var_group
    (node -> index into theta, -1 for none); the prefix tree's default is
    [leaf depths 3..7, node depths 1..6]."""
    LEAF_D, NODE_D = range(3, 8), range(1, 7)

    def __init__(self, hier, w, v, sup):
        self.h, self.w, self.v, self.sup = hier, w, v, sup
        self.leaf_of_node = np.full(hier.n_nodes, -1)
        self.leaf_of_node[hier.leaf_node] = np.arange(len(hier.leaf_node))
        self.levels = [np.flatnonzero(hier.depth == d) for d in range(hier.depth.max() + 1)]
        self.group = getattr(hier, "var_group", None)
        if self.group is None:
            g = np.full(hier.n_nodes, -1)
            for i, d in enumerate(self.LEAF_D):
                g[hier.is_leaf & (hier.depth == d)] = i
            for i, d in enumerate(self.NODE_D, len(self.LEAF_D)):
                g[~hier.is_leaf & (hier.depth == d)] = i
            self.group = g
        self.n_theta = int(self.group.max()) + 1

    def node_var(self, theta):
        return np.where(self.group >= 0, np.exp(np.asarray(theta)[np.maximum(self.group, 0)]), 0.0)

    def upward(self, theta):
        """Messages from each node's subtree. acc[:, a] sums over a's children
        that carry data: 1/s, x/s, x^2/s, log s, count, where child i says
        "x_i ~ N(mu_a, s_i)". Returns acc, x, s, s2 and the log evidence of
        every inner-node combination below the root."""
        h, s2 = self.h, self.node_var(theta)
        N = h.n_nodes
        acc = np.zeros((5, N))
        x, s = np.zeros(N), np.full(N, np.inf)
        logZ = 0.0
        for d in range(len(self.levels) - 1, 0, -1):
            nodes = self.levels[d]
            lf = nodes[h.is_leaf[nodes]]
            j = self.leaf_of_node[lf]
            ok = self.sup[j]
            x[lf[ok]], s[lf[ok]] = self.w[j[ok]], self.v[j[ok]] + s2[lf[ok]]
            inn = nodes[~h.is_leaf[nodes]]
            ii = inn[acc[0, inn] > 0]
            logZ += self.combine(acc[:, ii])
            x[ii] = acc[1, ii] / acc[0, ii]
            s[ii] = 1 / acc[0, ii] + s2[ii]
            snd = nodes[np.isfinite(s[nodes])]
            p = h.parent[snd]
            for r, val in enumerate((1 / s[snd], x[snd] / s[snd], x[snd] ** 2 / s[snd],
                                     np.log(s[snd]), np.ones(len(snd)))):
                acc[r] += np.bincount(p, val, minlength=N)
        return acc, x, s, s2, logZ

    @staticmethod
    def combine(q):
        """log Z of prod_i N(x_i; mu, s_i) = Z N(M; mu, 1/Lam), summed over columns."""
        return np.sum(-0.5 * (q[2] - q[1] ** 2 / q[0]) - 0.5 * q[3]
                      - 0.5 * (q[4] - 1) * np.log(2 * np.pi) - 0.5 * np.log(q[0]))

    def evidence(self, theta):
        acc, _, _, _, logZ = self.upward(theta)
        q = acc[:, 0]
        M, V = q[1] / q[0], 1 / q[0]
        return (logZ + self.combine(q[:, None])
                - 0.5 * (np.log(2 * np.pi * (V + ROOT_VAR)) + M ** 2 / (V + ROOT_VAR)))

    def fit(self, t0=None):
        if t0 is None:
            t0 = np.full(self.n_theta, np.log(0.05))
        r = minimize(lambda t: -self.evidence(t), t0, method="L-BFGS-B",
                     bounds=[(-12, 2)] * len(t0))
        self.theta = r.x
        return self

    def predict(self, nodes=False):
        """Leave-own-data-out N(m, P) for every leaf column. With nodes=True
        also the posterior N(mean, var) of every node's mu given all data."""
        h = self.h
        acc, x, s, s2, _ = self.upward(self.theta)
        Lam, Hs = acc[0], acc[1]
        pm, pv = np.zeros(h.n_nodes), np.zeros(h.n_nodes)    # outside-subtree marginal of each node
        pv[0] = ROOT_VAR
        for d in range(1, len(self.levels)):
            c = self.levels[d]
            a = h.parent[c]
            send = np.isfinite(s[c])
            prec = 1 / pv[a] + Lam[a] - np.where(send, 1 / np.where(send, s[c], 1), 0)
            eta = pm[a] / pv[a] + Hs[a] - np.where(send, x[c] / np.where(send, s[c], 1), 0)
            pm[c] = eta / prec
            pv[c] = 1 / prec + s2[c]
        if nodes:
            prec = 1 / pv + Lam
            return pm[h.leaf_node], pv[h.leaf_node], (pm / pv + Hs) / prec, 1 / prec
        return pm[h.leaf_node], pv[h.leaf_node]


def map_update(m, P, n, k, b0, iters=30):
    """argmax_w  log N(w; m, P) + k log s(b0+w) + (n-k) log s(-b0-w)."""
    w = m.copy()
    for _ in range(iters):
        p = expit(b0 + w)
        g = -(w - m) / P + k - n * p
        H = 1 / P + n * p * (1 - p)
        w = w + np.clip(g / H, -2, 2)
    return w


def site(w, n, k, b0):
    """Gaussian site of the binomial likelihood, Laplace-expanded at w."""
    p = expit(b0 + w)
    v = 1 / np.maximum(n * p * (1 - p), 1e-12)
    return w + (k - n * p) * v, v


def flat_ridge(n, k, b0, iters=8):
    """own: w_c ~ N(0, s2), s2 by marginal likelihood, Laplace sites."""
    sup = n > 0
    x, v = own(n, k, b0)
    for _ in range(iters):
        xs, vs = x[sup], v[sup]
        nll = lambda t: 0.5 * np.sum(np.log(vs + np.exp(t[0])) + xs ** 2 / (vs + np.exp(t[0])))
        s2 = np.exp(minimize(nll, [np.log(0.1)], bounds=[(-12, 2)]).x[0])
        w = map_update(np.zeros(len(n)), np.full(len(n), s2), n, k, b0)
        x, v = site(w, n, k, b0)
    return w, s2


def estimate(hier, A, n, k, b0, iters=6):
    w_mle, v_mle = own(n, k, b0)
    sup = n > 0
    w_own, s2 = flat_ridge(n, k, b0)
    w, v = site(w_own, n, k, b0)             # start the sites from the flat fit
    theta = None
    for _ in range(iters):
        # EB can stall at a zero-variance corner: refit from the fixed start as
        # well as from the previous solution and keep the better
        fits = [GaussTree(hier, w, v, sup).fit(t) for t in ([None] if theta is None else [None, theta])]
        gt = max(fits, key=lambda g: g.evidence(g.theta))
        theta = gt.theta
        m, P, node_m, node_v = gt.predict(nodes=True)
        wp = map_update(m, P, n, k, b0)
        w, v = site(wp, n, k, b0)
    return {"null": np.zeros(len(n)), "own_mle": w_mle, "own": w_own,
            "sib": sibling_ivw(hier, A, w_mle, v_mle, sup),
            "tree": m, "tree_P": P, "post": wp,
            "node_m": node_m, "node_v": node_v}, np.r_[theta, np.log(s2)]


def gain(nE, kE, w, b0):
    """Per-code log-likelihood gain of logit b0 + w over b0 on the eval pairs."""
    z = b0 + w
    return kE * (log_expit(z) - log_expit(b0)) + (nE - kE) * (log_expit(-z) - log_expit(-b0))


def ceiling(nE, kE, b0):
    p0 = expit(b0)
    q = np.where(nE > 0, kE / np.maximum(nE, 1), 0)
    return nE * (xlogy(q, q / p0) + xlogy(1 - q, (1 - q) / (1 - p0)))


def pair_auc(nE, kE, w, m):
    m = m & (nE > 0)
    return float(roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(m.sum())], np.r_[w[m], w[m]],
                               sample_weight=np.r_[kE[m], nE[m] - kE[m]]))


def logloss(nE, kE, w, b0):
    z = b0 + w
    return -(kE * log_expit(z) + (nE - kE) * log_expit(-z))


def brier(nE, kE, w, b0):
    p = expit(b0 + w)
    return kE * (1 - p) ** 2 + (nE - kE) * p ** 2


def score(nT, nE, kE, est, b0):
    """Per group of codes: log-loss, Brier and gain per pair, pair AUROC."""
    out = {}
    groups = {"all": nT >= 0, "seen": nT > 0, "unseen": nT == 0}
    groups.update({f"n{lo}-{hi}": (nT >= lo) & (nT < hi) for lo, hi in BINS})
    for gname, gm in groups.items():
        m = gm & (nE > 0)
        occ = nE[m].sum()
        if occ == 0:
            continue
        rec = {"codes": int(m.sum()), "pairs": int(occ), "pos": int(kE[m].sum()),
               "ceiling": float(ceiling(nE, kE, b0)[m].sum() / occ * 1e3)}
        for e in ESTIMATORS:
            w = est[e]
            rec[e] = float(gain(nE, kE, w, b0)[m].sum() / occ * 1e3)
            rec[f"ll_{e}"] = float(logloss(nE, kE, w, b0)[m].sum() / occ)
            rec[f"brier_{e}"] = float(brier(nE, kE, w, b0)[m].sum() / occ)
            if 0 < kE[m].sum() < occ:
                rec[f"auc_{e}"] = pair_auc(nE, kE, w, m) if e != "null" else 0.5
        out[gname] = rec
    return out


def not_z(hier):
    return np.array([not c.startswith("Z") for c in hier.leaves])


def main():
    X, y, subj, split, hier = load()
    yg = np.zeros(len(y), np.int8)              # random setting only
    A = hier.ancestors_matrix().tocsr()
    OUT.mkdir(parents=True, exist_ok=True)
    designs = {"any": X, "primary": primary_design(hier)}
    keep = not_z(hier)
    rows = []
    for mode, Xm in designs.items():
        for frac in FRACS:
            for seed in (SEEDS if frac < 1 else [0]):
                tr, dv, _ = splits("R", frac, seed, subj, split, yg)
                nT, kT = (a * keep for a in counts(Xm, y, tr))
                b0 = float(np.log(kT.sum() / (nT.sum() - kT.sum())))
                nE, kE = (a * keep for a in counts(Xm, y, dv))
                est, theta = estimate(hier, A, nT, kT, b0)
                rec = {"mode": mode, "frac": frac, "seed": seed, "n_train": int(tr.sum()),
                       "b0": b0, "sd": np.exp(theta / 2).round(4).tolist(),
                       "train": score(nT, nT, kT, est, b0),
                       "dev": score(nT, nE, kE, est, b0)}
                if seed == 0:
                    np.savez_compressed(OUT / f"{mode}_{frac}_est.npz", nT=nT, kT=kT, b0=b0,
                                        theta=theta, **est)
                rows.append(rec)
                s = rec["dev"]["all"]
                print(mode, frac, seed, rec["n_train"], "dev all: auc",
                      *(f"{e}={s.get('auc_' + e, np.nan):.3f}" for e in ESTIMATORS),
                      "ll", *(f"{e}={s['ll_' + e]:.4f}" for e in ESTIMATORS), flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
