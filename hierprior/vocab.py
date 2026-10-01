"""The vocabulary prior: synthesis analyses for docs/vocabulary_prior.md.

Studies 1-3 share one model. A code's log-odds shift is a sum over its
prefixes, w_c = sum_{p prefix of c} theta_p with theta_p ~ N(0, tau2[len p]):
a Gaussian process over code strings whose covariance is the variance that two
codes accumulate along their shared prefix. On study 3's one-code fits
(output/hier3/*_est.npz, chapter Z excluded) this script checks what the model
implies, and it collects the multivariate evidence of study 2:

  correlogram   correlation of two codes' log-odds shifts against the number of
                leading characters they share, model-free, next to the value
                implied by the fitted hierarchy. Primary mode only: there each
                admission has one code, so different codes' estimates have
                independent noise.
  calibration   standardised errors of the leave-code-out prior against each
                code's own training estimate, and coverage of its intervals
  worth         pseudo-admissions of a prior, 1 / (variance * p0 (1 - p0)), and
                the break-even they predict, against the measured one
  test          study 3's headline numbers on the untouched 20% test split,
                run once, with patient-bootstrap intervals; the learning curve
                on dev and test with a bootstrap over codes for the break-even
  foresight     codes a 5k-admission subset has barely seen, predicted by the
                prior and compared with their full-data estimates
  multivariate  study 2: training admissions the codes-only LR needs to match
                the ancestor-indicator LR
  explorer      every code's path through the tree, for the published page
  study4        study 4's outcomes and ICD-level trees (hierprior.levels), for the page

Log-loss is binary cross-entropy with natural logs. It is reported as the
percentage of the base-rate model's log-loss that a model removes, which is
McFadden's R^2 computed on held-out data.

  .venv/Scripts/python -m hierprior.vocab           # ~6 min
  .venv/Scripts/python -m hierprior.vocab --page    # figures and artifact_data.json from saved results
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, log_expit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from hierprior.analyze import GRID_C, MUTED, TEXT, style
from hierprior.analyze3 import clean, descriptions, m_equivalent
from hierprior.data import load
from hierprior.experiment2 import FRACS, splits
from hierprior.onecode import OUT as OUT3, counts, logloss, map_update, not_z, primary_design, score
from hierprior.onecode_curve import M_GRID, M_MIN, R, auc

OUT = Path("output/vocab")
NMIN = 30                   # training admissions a code needs to enter the model-free checks
LEVELS = [0.5, 0.8, 0.95]   # interval coverage levels
BOOT = 500                  # patient-bootstrap replicates for the dev and test intervals
ORANGE, BLUE, GREEN = "#eb6834", "#2a78d6", "#1baf7a"


def est(mode, frac):
    z = np.load(OUT3 / f"{mode}_{frac}_est.npz")
    return {k: z[k] for k in z.files}


def own(n, k, b0):
    return np.log((k + .5) / (n - k + .5)) - b0, 1 / (k + .5) + 1 / (n - k + .5)


def variances(theta):
    """Fitted prior variances by depth: leaf (code-level) for depths 3-7, node for 1-6."""
    return dict(zip(range(3, 8), np.exp(theta[:5]))), dict(zip(range(1, 7), np.exp(theta[5:11])))


def ancestors_by_depth(hier):
    """(n_leaves, max depth + 1): each leaf's ancestor node at every depth, -1 where none."""
    anc = np.full((len(hier.leaves), hier.depth.max() + 1), -1)
    for j, v in enumerate(hier.leaf_node):
        u = hier.parent[v]
        while u > 0:
            anc[j, hier.depth[u]] = u
            u = hier.parent[u]
    return anc


def pair_cov(x, u, anc):
    """Weighted mean of x_c x_c' over pairs of distinct codes that share exactly
    k leading characters (their deepest common ancestor is at depth k)."""
    D = anc.shape[1]
    S, W = np.zeros(D + 1), np.zeros(D + 1)
    S[0], W[0] = np.sum(u * x) ** 2 - np.sum((u * x) ** 2), np.sum(u) ** 2 - np.sum(u ** 2)
    for k in range(1, D):
        ok = anc[:, k] >= 0
        if not ok.any():
            continue
        g = np.unique(anc[ok, k], return_inverse=True)[1]
        ux, uu = u[ok] * x[ok], u[ok]
        S[k] = np.sum(np.bincount(g, ux) ** 2 - np.bincount(g, ux ** 2))
        W[k] = np.sum(np.bincount(g, uu) ** 2 - np.bincount(g, uu ** 2))
    S_eq, W_eq = S[:-1] - S[1:], W[:-1] - W[1:]
    return np.where(W_eq > 0, S_eq / np.maximum(W_eq, 1e-300), np.nan), W_eq


def share_curve(x, v, u, anc):
    """Share of the between-code variance of the true log-odds shifts carried by
    k shared characters: covariance at k over the noise-corrected variance."""
    xc = x - np.sum(u * x) / np.sum(u)
    var = np.sum(u * (xc ** 2 - v)) / np.sum(u)
    cov, W = pair_cov(xc, u, anc)
    return cov / var, W, var


def correlogram(hier, keep):
    z = est("primary", 1.0)
    nT, kT, b0 = z["nT"], z["kT"], float(z["b0"])
    sel = keep & (nT >= NMIN)
    x, v = own(nT[sel], kT[sel], b0)
    anc = ancestors_by_depth(hier)[sel]
    s2_leaf, s2_node = variances(z["theta"])
    u = 1 / (0.4 + v)                                      # ~1 / (true variance + noise)
    rho, W, var = share_curve(x, v, u, anc)
    letters = np.array([c[0] for c in np.array(hier.leaves)[sel]])
    pv = np.array([share_curve(x[letters != L], v[letters != L], u[letters != L], anc[letters != L])[0]
                   for L in np.unique(letters)])
    g, ok = len(pv), np.isfinite(pv).any(0)           # depths with no pairs stay NaN
    se = np.full(pv.shape[1], np.nan)
    se[ok] = np.sqrt((g - 1) / g * np.nansum((pv[:, ok] - np.nanmean(pv[:, ok], 0)) ** 2, 0))
    depth = hier.depth[hier.leaf_node][sel]
    mvar = np.array([sum(s2_node.get(j, 0) for j in range(1, d)) + s2_leaf.get(d, 0) for d in depth])
    mbar = np.sum(u * mvar) / np.sum(u)
    model = [sum(s2_node.get(j, 0) for j in range(1, k + 1)) / mbar for k in range(len(rho))]
    npairs = W / np.mean(u) ** 2
    return {"codes": int(sel.sum()), "admissions": int(nT[sel].sum()), "true_var": float(var),
            "model_var": float(mbar), "shared": list(range(len(rho))), "empirical": rho.tolist(),
            "se": se.tolist(), "model": model, "pairs": npairs.round().tolist(),
            "sd_leaf": {d: float(np.sqrt(s)) for d, s in s2_leaf.items()},
            "sd_node": {d: float(np.sqrt(s)) for d, s in s2_node.items()}}


def decomposition(hier, keep):
    """Model variance of a code's log-odds shift split by level, averaged over
    codes weighted by training admissions."""
    out = {}
    depth = hier.depth[hier.leaf_node]
    for mode in ("primary", "any"):
        z = est(mode, 1.0)
        s2_leaf, s2_node = variances(z["theta"])
        sel = keep & (z["nT"] > 0)
        share = np.zeros(7)            # levels 1..6 shared ancestors, then the code itself
        for d in np.unique(depth[sel]):
            m = sel & (depth == d)
            comp = np.array([s2_node.get(j, 0) if j < d else 0 for j in range(1, 7)] + [s2_leaf.get(d, 0)])
            share += z["nT"][m].sum() * comp / comp.sum()
        share /= z["nT"][sel].sum()
        out[mode] = {"by_level": share.tolist(), "code_specific": float(share[-1]),
                     "sd_node": {d: float(np.sqrt(s)) for d, s in s2_node.items()},
                     "sd_leaf": {d: float(np.sqrt(s)) for d, s in s2_leaf.items()},
                     "sd_naive": float(np.exp(z["theta"][-1] / 2))}
    return out


def calibration_and_worth(designs, y, dv, keep, C):
    """Leave-code-out prior N(m, P) against each code's own training estimate
    (x, v): z = (x - m) / sqrt(P + v) should be N(0, 1). Pseudo-admissions and
    the predicted break-even use the curve's codes (>= M_MIN training adm)."""
    out = {}
    for mode, Xm in designs.items():
        z = est(mode, 1.0)
        nT, kT, b0 = z["nT"], z["kT"], float(z["b0"])
        nE, kE = (a * keep for a in counts(Xm, y, dv))
        p0 = expit(b0)
        I0 = p0 * (1 - p0)
        s2 = float(np.exp(z["theta"][-1]))
        rec = {}
        for tag, lo in (("calib", NMIN), ("curve", M_MIN)):
            sel = keep & (nT >= lo) & (nE > 0)
            x, v = own(nT[sel], kT[sel], b0)
            m, P = z["tree"][sel], z["tree_P"][sel]
            zz = (x - m) / np.sqrt(P + v)
            info = expit(b0 + x) * (1 - expit(b0 + x))              # Fisher information per admission
            wt = nE[sel] * info                                     # a code's weight in the dev log-loss
            A = np.sum(wt * ((x - m) ** 2 - v)) / wt.sum()          # actual error variance of the prior
            w2 = x ** 2 - v                                         # unbiased for the true shift squared
            S = np.sum(wt * w2) / wt.sum()
            # squared bias + variance of the posterior mean under N(0, s2) after g admissions
            grid = np.linspace(0, 400, 4001)
            mse = np.array([np.sum(wt * (w2 / s2 ** 2 + g * info) / (1 / s2 + g * info) ** 2) / wt.sum()
                            for g in grid])
            rec[tag] = {
                "codes": int(sel.sum()), "z_mean": float(zz.mean()), "z_sq": float(np.mean(zz ** 2)),
                "coverage": {str(c): float(np.mean(np.abs(zz) < norm.ppf(0.5 + c / 2))) for c in LEVELS},
                "z_hist": np.histogram(np.clip(zz, -4, 4), bins=np.linspace(-4, 4, 33))[0].tolist(),
                "slope": float(np.sum(nT[sel] * x * m) / np.sum(nT[sel] * m ** 2)),
                "corr": float(np.corrcoef(m, x)[0, 1]),
                "stated_var": float(np.mean(P)), "actual_var": float(A), "spread": float(S), "naive_var": s2,
                "pseudo_prior_stated": float(np.median(1 / (P * I0))), "pseudo_prior_actual": float(1 / (A * I0)),
                "pseudo_naive": float(1 / (s2 * I0)),
                "breakeven_theory": float(grid[np.argmax(mse <= A)]) if (mse <= A).any() else None}
        rec["curve"]["breakeven_measured"] = {k: m_equivalent(C, mode, 1.0, k) for k in ("ll", "auc")}
        out[mode] = rec
    return out


def admission_arrays(Xm, y, subj, rows, keep):
    """Primary mode: one (code, outcome, patient) per admission in rows, chapter Z dropped."""
    Xr = Xm[rows].tocsr()
    has = np.diff(Xr.indptr) == 1
    code = np.full(Xr.shape[0], -1)
    code[has] = Xr.indices[Xr.indptr[:-1][has]]
    ok = has & keep[np.maximum(code, 0)]
    return code[ok], y[rows][ok], subj[rows][ok]


def boot_scores(code, yy, pt, ests, b0, nT, seed=0):
    """Point estimates and 95% patient-bootstrap intervals of AUROC and log-loss
    reduction, on all admissions and on those whose code is unseen in training."""
    rng = np.random.default_rng(seed)
    upt, inv = np.unique(pt, return_inverse=True)
    subsets = {"all": np.ones(len(code), bool), "unseen": nT[code] == 0}

    def metrics(wts):
        res = {}
        for g, m in subsets.items():
            if wts[m].sum() == 0:
                continue
            base = np.sum(wts[m] * -(yy[m] * log_expit(b0) + (1 - yy[m]) * log_expit(-b0)))
            for e, w in ests.items():
                zc = b0 + w[code[m]]
                ll = np.sum(wts[m] * -(yy[m] * log_expit(zc) + (1 - yy[m]) * log_expit(-zc)))
                res[f"{g}|llred_{e}"] = 100 * (base - ll) / base
                res[f"{g}|auc_{e}"] = roc_auc_score(yy[m], zc, sample_weight=wts[m]) if np.ptp(zc) > 0 else 0.5
        return res

    point = metrics(np.ones(len(code)))
    reps = []
    for _ in range(BOOT):
        cnt = np.bincount(rng.integers(0, len(upt), len(upt)), minlength=len(upt))
        reps.append(metrics(cnt[inv].astype(float)))
    return {k: [float(point[k]), float(np.percentile([r[k] for r in reps], 2.5)),
                float(np.percentile([r[k] for r in reps], 97.5))] for k in point}


def curve_codes(Xm, y, tr, ev, keep, mode, rng, z=None):
    """hierprior.onecode_curve's experiment with relatives from all training
    data, kept per code so that the break-even can be bootstrapped over codes.
    Given a fresh default_rng(0), dev as ev, and the modes in the order "any",
    "primary", it makes the same draws as onecode_curve. z: the saved fit that
    gives the prior (default: study 3's full-data fit for mode)."""
    nF, kF = (a * keep for a in counts(Xm, y, tr))
    nE, kE = (a * keep for a in counts(Xm, y, ev))
    el = np.flatnonzero((nF >= M_MIN) & (nE > 0))
    nF, kF, nE, kE = nF[el], kF[el], nE[el], kE[el]
    draws = {m: [rng.hypergeometric(kF.astype(int), (nF - kF).astype(int), m).astype(float) if m
                 else np.zeros(len(el)) for _ in range(R)] for m in M_GRID}
    z = est(mode, 1.0) if z is None else z
    b0, s2 = float(z["b0"]), float(np.exp(z["theta"][-1]))
    mt, Pt = z["tree"][el], z["tree_P"][el]
    L = {"hier": [], "naive": []}      # per m: each code's eval log-loss sum, mean over draws
    A = {"hier": [], "naive": []}      # per m: pooled AUROC, mean over draws
    n = np.full(len(el), 0.0)
    for m in M_GRID:
        acc = {e: ([], []) for e in L}
        n[:] = m
        for k in draws[m][:R if m else 1]:
            for e, w in (("hier", map_update(mt, Pt, n, k, b0)),
                         ("naive", map_update(np.zeros(len(el)), np.full(len(el), s2), n, k, b0))):
                acc[e][0].append(logloss(nE, kE, w, b0))
                acc[e][1].append(auc(nE, kE, w) if np.ptp(w) > 0 else 0.5)
        for e in L:
            L[e].append(np.mean(acc[e][0], 0))
            A[e].append(float(np.mean(acc[e][1])))
    return {"nE": nE, "L": {e: np.array(v) for e, v in L.items()}, "auc": A, "ref": logloss(nE, kE, 0.0, b0)}


def breakeven(c, idx=slice(None)):
    """Own admissions the generic-prior learner needs for its pooled log-loss to
    reach the vocabulary prior's with none (interpolated on log(1 + m))."""
    tot = c["nE"][idx].sum()
    target = c["L"]["hier"][0][idx].sum() / tot
    ys = c["L"]["naive"][:, idx].sum(1) / tot
    xs = np.log1p(M_GRID)
    ok = ys <= target
    if not ok.any():
        return np.nan
    i = int(np.argmax(ok))
    if i == 0:
        return 0.0
    f = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
    return float(np.expm1(xs[i - 1] + f * (xs[i] - xs[i - 1])))


def curve_summary(c, seed=0, B=2000):
    rng = np.random.default_rng(seed)
    n = len(c["nE"])
    boot = np.array([breakeven(c, rng.integers(0, n, n)) for _ in range(B)])
    tot = c["nE"].sum()
    ref = c["ref"].sum() / tot
    rows = [{"m": m, "ll_hier": float(c["L"]["hier"][i].sum() / tot), "ll_naive": float(c["L"]["naive"][i].sum() / tot),
             "auc_hier": c["auc"]["hier"][i], "auc_naive": c["auc"]["naive"][i],
             "llred_hier": float(100 * (1 - c["L"]["hier"][i].sum() / tot / ref)),
             "llred_naive": float(100 * (1 - c["L"]["naive"][i].sum() / tot / ref))} for i, m in enumerate(M_GRID)]
    return {"codes": n, "pairs": int(tot), "ref_ll": float(ref), "rows": rows, "breakeven_ll": breakeven(c),
            "breakeven_ll_ci": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]}


def curves(designs, y, subj, split, keep):
    """The learning curve on dev (onecode_curve's draws) and on test (new draws)."""
    tr, dv, te = splits("R", 1.0, 0, subj, split, np.zeros(len(y), np.int8))
    out = {"dev": {}, "test": {}}
    for ev, rng in (("dev", np.random.default_rng(0)), ("test", np.random.default_rng(1))):
        for mode in ("any", "primary"):
            c = curve_codes(designs[mode], y, tr, dv if ev == "dev" else te, keep, mode, rng)
            out[ev][mode] = curve_summary(c)
    return out


def chapter(letter):
    return {"A": "Infectious and parasitic diseases", "B": "Infectious and parasitic diseases",
            "C": "Malignant neoplasms", "D": "Other neoplasms (D00-D49); blood and immune (D50-D89)",
            "E": "Endocrine, nutritional and metabolic diseases", "F": "Mental and behavioural disorders",
            "G": "Diseases of the nervous system", "H": "Eye (H00-H59); ear (H60-H95)",
            "I": "Diseases of the circulatory system", "J": "Diseases of the respiratory system",
            "K": "Diseases of the digestive system", "L": "Diseases of the skin",
            "M": "Musculoskeletal and connective tissue", "N": "Diseases of the genitourinary system",
            "O": "Pregnancy, childbirth and the puerperium", "P": "Perinatal conditions",
            "Q": "Congenital malformations", "R": "Symptoms, signs and abnormal findings",
            "S": "Injury (by body region)", "T": "Injury, poisoning, complications of care",
            "U": "Codes for special purposes", "V": "External causes: transport", "W": "External causes",
            "X": "External causes", "Y": "External causes"}.get(letter, "")


def anatomy(hier, desc, keep, designs, y, dv):
    """Primary mode, full data: each code's path through the prefix tree, for
    the page's code explorer. Every ancestor on a code's path gets its
    posterior mean given all training data except that code's (Gaussian sites
    at the saved posterior modes), so the path ends at the code's leave-code-out
    prior N(m, P). Codes with at least 5 training + dev admissions, or unseen
    in training with at least 2 dev admissions."""
    from hierprior.onecode import ROOT_VAR, GaussTree, site
    z = est("primary", 1.0)
    nT, kT, b0 = z["nT"], z["kT"], float(z["b0"])
    w, v = site(z["post"], nT, kT, b0)
    gt = GaussTree(hier, w, v, nT > 0)
    acc, x, s, s2, _ = gt.upward(z["theta"][:11])
    pm, pv = np.zeros(hier.n_nodes), np.zeros(hier.n_nodes)
    pv[0] = ROOT_VAR
    for d in range(1, len(gt.levels)):             # outside-subtree marginals, as in GaussTree.predict
        c = gt.levels[d]
        a = hier.parent[c]
        send = np.isfinite(s[c])
        prec = 1 / pv[a] + acc[0, a] - np.where(send, 1 / np.where(send, s[c], 1), 0)
        eta = pm[a] / pv[a] + acc[1, a] - np.where(send, x[c] / np.where(send, s[c], 1), 0)
        pm[c], pv[c] = eta / prec, 1 / prec + s2[c]

    def path_without(j):
        """Ancestors of leaf j, root first, with their posterior means given all
        data except leaf j's: its message is taken out and the change is carried
        up the path."""
        u = hier.leaf_node[j]
        drop = np.array([1 / s[u], x[u] / s[u]]) if np.isfinite(s[u]) else np.zeros(2)
        out = []
        a = hier.parent[u]
        while a > 0:
            q = acc[:2, a] - drop
            q[0] = max(q[0], 0.0)
            out.append((a, (pm[a] / pv[a] + q[1]) / (1 / pv[a] + q[0])))
            old = np.array([1 / s[a], x[a] / s[a]]) if np.isfinite(s[a]) else np.zeros(2)
            new = np.array([1 / (1 / q[0] + s2[a]), q[1] / q[0] / (1 / q[0] + s2[a])]) if q[0] > 1e-12 else np.zeros(2)
            drop = old - new
            a = hier.parent[a]
        return out[::-1]

    n_node = hier.ancestors_matrix().T @ nT
    nE, kE = (a * keep for a in counts(designs["primary"], y, dv))
    show = np.flatnonzero(keep & ((nT + nE >= 5) | ((nT == 0) & (nE >= 2))))
    kids = hier.children()

    def label(a):
        name = hier.names[a]
        if len(name) == 1:
            return chapter(name)
        if len(name) == 2:          # not an ICD-10-CM level: name the categories it spans
            cats = sorted(hier.names[c].rstrip("$") for c in kids[a])
            return f"categories {cats[0]}-{cats[-1]}" if len(cats) > 1 else f"category {cats[0]}"
        return desc.get(name, "")

    nodes, leaves, check = {}, [], 0.0
    for j in show:
        path = path_without(j)
        check = max(check, abs(path[-1][1] - z["tree"][j]))
        for a, _ in path:
            if int(a) not in nodes:
                nodes[int(a)] = [hier.names[a], int(n_node[a]), label(a)]
        leaves.append([hier.leaves[j], desc.get(hier.leaves[j], ""), int(nT[j]), int(kT[j]), int(nE[j]), int(kE[j]),
                       round(float(z["tree"][j]), 3), round(float(np.sqrt(z["tree_P"][j])), 3),
                       round(float(z["post"][j]), 3), [int(a) for a, _ in path], [round(float(m), 2) for _, m in path]])
    return {"b0": b0, "check_max_abs": float(check), "nodes": nodes, "leaves": leaves}


def test_confirmation(designs, y, subj, split, keep):
    yg = np.zeros(len(y), np.int8)
    out = {"scaling": [], "boot": {}}
    for mode, Xm in designs.items():
        for frac in FRACS:
            tr, _, te = splits("R", frac, 0, subj, split, yg)
            z = est(mode, frac)
            nE, kE = (a * keep for a in counts(Xm, y, te))
            ests = {e: z[e] for e in ("null", "sib", "tree", "own_mle", "own", "post")}
            s = score(z["nT"], nE, kE, ests, float(z["b0"]))
            for g in s.values():
                for e in ests:
                    g[f"llred_{e}"] = 100 * (g["ll_null"] - g[f"ll_{e}"]) / g["ll_null"]
            out["scaling"].append({"mode": mode, "frac": frac, "n_train": int(tr.sum()), "test": s})
    _, dv, te = splits("R", 1.0, 0, subj, split, yg)
    z = est("primary", 1.0)
    for name, rows in (("dev", dv), ("test", te)):
        code, yy, pt = admission_arrays(designs["primary"], y, subj, rows, keep)
        out["boot"][name] = boot_scores(code, yy, pt, {e: z[e] for e in ("tree", "own", "post")},
                                        float(z["b0"]), z["nT"])
        out["boot"][name]["admissions"] = int(len(code))
        out["boot"][name]["unseen_admissions"] = int(np.sum(z["nT"][code] == 0))
        out["boot"][name]["unseen_codes"] = int(len(np.unique(code[z["nT"][code] == 0])))
    return out


def foresight(designs, y, subj, split, keep, hier, desc, frac=0.03):
    """Codes seen at most twice in a small training subset (seed 0) but at
    least 100 times in the full training split: the prior from the small
    subset against the full-data estimate."""
    out = {}
    for mode in designs:
        zs, zf = est(mode, frac), est(mode, 1.0)
        b0s, b0f = float(zs["b0"]), float(zf["b0"])
        sel = keep & (zs["nT"] <= 2) & (zf["nT"] >= 100)
        xf, vf = own(zf["nT"][sel], zf["kT"][sel], b0f)
        truth = b0f + xf                                   # full-data log-odds
        prior = b0s + zs["tree"][sel]
        base = np.full(sel.sum(), b0s)
        mle = b0s + zs["own_mle"][sel]
        unseen = zs["nT"][sel] == 0
        tr = splits("R", frac, 0, subj, split, np.zeros(len(y), np.int8))[0]
        rec = {"codes": int(sel.sum()), "unseen": int(unseen.sum()), "n_train_small": int(tr.sum())}
        for name, pred in (("prior", prior), ("base_rate", base), ("own_mle", mle)):
            for g, m in (("", np.ones(len(pred), bool)), ("unseen_", unseen)):
                err = (pred[m] - truth[m]) ** 2 - vf[m]
                rec[g + name] = {"rmse": float(np.sqrt(max(np.mean(err), 0))),
                                 "corr": float(np.corrcoef(pred[m], truth[m])[0, 1]) if np.ptp(pred[m]) > 0 else None}
        idx = np.flatnonzero(sel)
        rows = []
        for i, j in enumerate(idx):
            rows.append({"code": hier.leaves[j], "desc": desc.get(hier.leaves[j], ""),
                         "n_small": int(zs["nT"][j]), "k_small": int(zs["kT"][j]),
                         "n_full": int(zf["nT"][j]), "k_full": int(zf["kT"][j]),
                         "p_prior": float(expit(prior[i])), "p_prior_lo": float(expit(prior[i] - 1.96 * np.sqrt(zs["tree_P"][j]))),
                         "p_prior_hi": float(expit(prior[i] + 1.96 * np.sqrt(zs["tree_P"][j]))),
                         "p_base": float(expit(b0s)), "p_full": float(expit(truth[i])),
                         "parent": hier.names[hier.parent[hier.leaf_node[j]]]})
        rec["rows"] = rows
        out[mode] = rec
    return out


def multivariate():
    """Study 2, random setting: test AUROC and log-loss reduction by training
    size, and the training admissions the codes-only LR needs to match the
    ancestor-indicator LR (log-linear interpolation of AUROC in log N)."""
    runs = [json.loads(p.read_text()) for p in Path("output/hier2/runs").glob("R_*.json")]
    tab = {}
    for r in runs:
        if r["model"] not in ("l2", "rollup", "hier", "hier0"):
            continue
        p_tr, p_te = r["prev_train"], r["prev_test"]
        base = -(p_te * np.log(p_tr) + (1 - p_te) * np.log(1 - p_tr))
        t = r["best"]["test"]
        tab.setdefault((r["model"], r["frac"]), []).append(
            {"n": r["n_train"], "auc": t["auc"], "llred": 100 * (base - t["ll"]) / base, "auc_aff": t.get("auc_aff")})
    rows = {}
    for (model, frac), rs in sorted(tab.items()):
        rows.setdefault(model, []).append({"frac": frac, "n": float(np.mean([r["n"] for r in rs])),
                                           **{k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))]
                                              for k in ("auc", "llred", "auc_aff")}})
    ln = np.log([r["n"] for r in rows["l2"]])
    a_l2 = np.array([r["auc"][0] for r in rows["l2"]])
    for r in rows["rollup"]:
        target = r["auc"][0]
        r["codes_only_needs"] = float(np.exp(np.interp(target, a_l2, ln))) if a_l2[0] <= target <= a_l2[-1] else None
        r["multiplier"] = r["codes_only_needs"] / r["n"] if r["codes_only_needs"] else None
    return rows


def figures(res):
    OUT.mkdir(parents=True, exist_ok=True)
    c = res["correlogram"]
    k = np.array(c["shared"][:6])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.errorbar(k, c["empirical"][:6], yerr=1.96 * np.array(c["se"][:6]), fmt="o", color=ORANGE, ms=7,
                capsize=3, lw=1.5, label="model-free estimate ±95% (jackknife over chapters)")
    ax.plot(k, c["model"][:6], color=MUTED, lw=1.5, ls="--", marker="s", ms=4, label="implied by the fitted hierarchy")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(k, ["0\ndifferent\nchapter", "1\nE", "2\nE1", "3\nE11", "4\nE116", "5\nE1165"])
    ax.set_ylabel("correlation of two codes'\nreadmission log-odds", color=TEXT)
    ax.set_xlabel("leading characters the two codes share", color=TEXT)
    ax.set_title(f"Primary diagnoses, {c['codes']:,} codes with >= {NMIN} training admissions", color=TEXT,
                 loc="left", fontsize=10)
    ax.grid(True, color=GRID_C, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "correlogram.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, mode in zip(axes, ("primary", "any")):
        cal = res["calibration"][mode]["calib"]
        edges = np.linspace(-4, 4, 33)
        h = np.array(cal["z_hist"], float)
        ax.bar(edges[:-1], h / h.sum() / np.diff(edges), width=np.diff(edges), align="edge", color=ORANGE,
               alpha=0.55, lw=0)
        xs = np.linspace(-4, 4, 200)
        ax.plot(xs, norm.pdf(xs), color=TEXT, lw=1.5)
        cov = cal["coverage"]
        ax.set_title(f"{'Primary diagnoses' if mode == 'primary' else 'Every code occurrence'}: {cal['codes']:,} codes\n"
                     f"inside the prior's 50 / 80 / 95% interval: {100 * cov['0.5']:.0f} / {100 * cov['0.8']:.0f} / "
                     f"{100 * cov['0.95']:.0f}%", color=TEXT, loc="left", fontsize=9.5)
        ax.set_xlabel("(own estimate - prior mean) / combined sd", color=TEXT)
        ax.grid(True, color=GRID_C, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("density", color=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ticks = [0, 1, 2, 5, 10, 20, 50, 100]
    for ax, (metric, ylab) in zip(axes, (("auc", "AUROC"), ("llred", "log-loss reduction vs base rate (%)"))):
        for ev, alpha in (("dev", 1.0), ("test", 0.5)):
            c = res["curves"][ev]["primary"]
            x = np.log1p([r["m"] for r in c["rows"]])
            for e, col, lab in (("hier", GREEN, "own data + vocabulary prior"), ("naive", BLUE, "own data + generic prior")):
                ax.plot(x, [r[f"{metric}_{e}"] for r in c["rows"]], color=col, lw=2, alpha=alpha, marker="o", ms=4,
                        mec="white", label=f"{lab}, {ev}")
        be = res["curves"]["dev"]["primary"]["breakeven_ll"]
        ax.axvline(np.log1p(be), color=MUTED, lw=1)
        ax.set_xticks(np.log1p(ticks), [str(t) for t in ticks])
        ax.set_xlabel("admissions of the code's own data given to the learner", color=TEXT)
        ax.set_ylabel(ylab, color=TEXT)
        ax.grid(True, color=GRID_C, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    c = res["curves"]
    axes[0].set_title(f"Primary diagnoses, 328 codes. Break-even (log-loss): dev {c['dev']['primary']['breakeven_ll']:.0f} "
                      f"[{c['dev']['primary']['breakeven_ll_ci'][0]:.0f}-{c['dev']['primary']['breakeven_ll_ci'][1]:.0f}], "
                      f"test {c['test']['primary']['breakeven_ll']:.0f} [{c['test']['primary']['breakeven_ll_ci'][0]:.0f}-"
                      f"{c['test']['primary']['breakeven_ll_ci'][1]:.0f}]", color=TEXT, loc="left", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exchange.png", dpi=140)
    plt.close(fig)

    mv = res["multivariate"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for model, col, lab in (("rollup", GREEN, "codes + prefix groups (ancestor-indicator LR)"),
                            ("l2", BLUE, "codes only (L2 LR)")):
        n = [r["n"] for r in mv[model]]
        mu = np.array([r["auc"][0] for r in mv[model]])
        sd = np.array([r["auc"][1] for r in mv[model]])
        ax.fill_between(n, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.plot(n, mu, color=col, lw=2, marker="o", ms=5, mec="white", label=lab)
    for r in mv["rollup"]:
        if r["codes_only_needs"]:
            ax.annotate("", xy=(r["codes_only_needs"], r["auc"][0]), xytext=(r["n"], r["auc"][0]),
                        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
            ax.text(np.sqrt(r["n"] * r["codes_only_needs"]), r["auc"][0] + 0.002, f"x{r['multiplier']:.1f}",
                    ha="center", va="bottom", fontsize=8, color=MUTED)
    style(ax)
    ax.set_xlabel("training admissions (log scale)", color=TEXT)
    ax.set_ylabel("test AUROC (all diagnosis codes)", color=TEXT)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "multivariate.png", dpi=140)
    plt.close(fig)


def r3(x):
    """Round floats in nested lists and dicts for the page: 4 significant
    figures, but counts such as training sizes keep their units digit."""
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    if isinstance(x, float):
        return round(x, 1) if abs(x) >= 1000 else float(f"{x:.4g}")
    return x


def study4_data():
    """Study 4 (hierprior.levels, output/hier4) for the page: both outcomes on
    the prefix tree and on the official ICD-10-CM levels."""
    H4 = Path("output/hier4")
    R = json.loads((H4 / "results.json").read_text())
    C = json.loads((H4 / "curves.json").read_text())
    X = json.loads((H4 / "extra.json").read_text())
    by = {}
    for r in R:
        by.setdefault((r["outcome"], r["mode"], r["tree"], r["frac"]), []).append(r)
    scaling = {}
    for (o, m, t, f), rs in sorted(by.items(), key=lambda kv: kv[0][3]):
        row = {"n": float(np.mean([r["n_train"] for r in rs]))}
        for e in ("tree", "own", "post"):
            for k in ("auc", "llred"):
                v = [r["dev"]["all"][f"{k}_{e}"] for r in rs]
                row[f"{k}_{e}"] = [float(np.mean(v)), float(np.std(v))]
        scaling.setdefault(m, {}).setdefault(o, {}).setdefault(t, []).append(row)
    full = {(r["outcome"], r["mode"], r["tree"]): r for r in R if r["frac"] == 1.0}
    breakeven = {}
    for c in C:
        if c["rel_frac"] == 1.0:
            breakeven.setdefault(c["outcome"], {}).setdefault(c["mode"], {})[c["tree"]] =                 [c["breakeven_ll"]] + list(c["breakeven_ll_ci"])
    calib = {}
    for c in X["calibration"]:
        if c["tree"] == "prefix":
            calib.setdefault(c["outcome"], {})[c["mode"]] = {"codes": c["codes"], "cov95": c["coverage"]["0.95"],
                                                             "z_sq": c["z_sq"], "corr": c["corr"]}
    chapters = []
    for code, r in X["coefs"]["primary"].items():
        if r["level"] == "chapter" and all(r.get(o, {}).get("n", 0) >= 100 for o in ("readmit", "mort1y")):
            chapters.append({"code": code, "name": r["desc"], "n": r["readmit"]["n"],
                             **{o: {"theta": r[o]["theta"], "rate": r[o]["rate"]} for o in ("readmit", "mort1y")}})
    chapters.sort(key=lambda c: -c["mort1y"]["theta"])
    return {"base_rate": {o: float(expit(full[(o, "primary", "prefix")]["b0"])) for o in ("readmit", "mort1y")},
            "scaling": scaling, "breakeven": breakeven, "calibration": calib, "chapters": chapters,
            "sd_official_category": {o: full[(o, "primary", "official_category")]["sd"][:4] for o in ("readmit", "mort1y")},
            "sd_naive": {o: full[(o, "primary", "prefix")]["sd"][-1] for o in ("readmit", "mort1y")},
            "correlogram": {o: {k: g[k] for k in ("codes", "empirical", "se", "model", "true_var", "model_var")}
                            for o, g in X["correlogram"].items()}}


def page_data():
    """output/vocab/artifact_data.json: everything the published page draws,
    assembled from vocab.json, explorer.json and study 3's outputs."""
    V = json.loads((OUT / "vocab.json").read_text())
    ex = json.loads((OUT / "explorer.json").read_text())
    agg = {(a["mode"], a["frac"]): a for a in json.loads((OUT3 / "artifact_data.json").read_text())["agg"]}
    drill = json.loads((OUT3 / "artifact_data.json").read_text())["drill"]["primary"]
    ests = ("sib", "tree", "own_mle", "own", "post")
    bins = ("unseen", "n1-10", "n10-100", "n100-1000", "n1000-1000000000")
    scaling = {}
    for mode in ("primary", "any"):
        dev = [{"n": a["n_train"], **{f"{m}_{e}": a["dev"]["all"][f"{m}_{e}"] for e in ests for m in ("auc", "llred")},
                "support": [a["dev"].get(b, {"pairs": [0, 0]})["pairs"][0] / a["dev"]["all"]["pairs"][0] for b in bins]}
               for (md, f), a in sorted(agg.items()) if md == mode]
        test = [{"n": t["n_train"], **{f"{m}_{e}": [t["test"]["all"][f"{m}_{e}"], 0.0] for e in ests for m in ("auc", "llred")}}
                for t in V["test"]["scaling"] if t["mode"] == mode]
        scaling[mode] = {"dev": dev, "test": test}
    C = json.loads((OUT3 / "curve.json").read_text())
    worth = {mode: [{"n_rel": c["n_rel"], "ll": m_equivalent(C, mode, c["rel_frac"], "ll"),
                     "auc": m_equivalent(C, mode, c["rel_frac"], "auc")}
                    for c in C if c["mode"] == mode and c["m"] == 0] for mode in ("primary", "any")}
    groups = {}
    for g in ("C78", "I61", "I502", "K859"):
        d = drill[f"{g}|1.0"]
        groups[g] = {"name": d["name"], "llred": d["llred"], "dev_pairs": d["dev_pairs"],
                     "rows": [{k: r[k] for k in ("code", "desc", "n_train", "n_dev", "own_mle", "own_mle_ci", "tree",
                                                  "tree_ci", "dev_w", "dev_ci", "rate_train", "rate_dev")} for r in d["rows"]]}
    out = {"b0": ex["b0"], "correlogram": V["correlogram"], "decomposition": V["decomposition"],
           "calibration": V["calibration"], "foresight": {m: {k: v for k, v in d.items() if k != "rows"}
                                                          for m, d in V["foresight"].items()},
           "multivariate": V["multivariate"], "boot": V["test"]["boot"], "curves": V["curves"],
           "scaling": scaling, "worth": worth, "groups": groups, "explorer": ex, "study4": study4_data()}
    (OUT / "artifact_data.json").write_text(json.dumps(clean(r3(out)), separators=(",", ":")))


def main():
    import sys
    if "--page" in sys.argv:
        figures(json.loads((OUT / "vocab.json").read_text()))
        page_data()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, subj, split, hier = load()
    keep = not_z(hier)
    designs = {"primary": primary_design(hier), "any": X}
    _, dv, _ = splits("R", 1.0, 0, subj, split, np.zeros(len(y), np.int8))
    C = json.loads((OUT3 / "curve.json").read_text())
    desc = descriptions()
    res = {"correlogram": correlogram(hier, keep), "decomposition": decomposition(hier, keep),
           "calibration": calibration_and_worth(designs, y, dv, keep, C),
           "foresight": foresight(designs, y, subj, split, keep, hier, desc),
           "multivariate": multivariate()}
    res["test"] = test_confirmation(designs, y, subj, split, keep)
    res["curves"] = curves(designs, y, subj, split, keep)
    (OUT / "vocab.json").write_text(json.dumps(clean(res), indent=1))
    ex = anatomy(hier, desc, keep, designs, y, dv)
    (OUT / "explorer.json").write_text(json.dumps(clean(ex), separators=(",", ":")))
    print("explorer:", len(ex["leaves"]), "codes,", len(ex["nodes"]), "prefix nodes; max |prior - saved prior|",
          ex["check_max_abs"])
    figures(res)
    page_data()
    c = res["correlogram"]
    print("correlogram (shared chars: empirical ± se / model):",
          [f"{k}: {e:+.2f}±{s:.2f} / {m:.2f}" for k, e, s, m in zip(c["shared"], c["empirical"], c["se"], c["model"])][:6])
    for mode, d in res["decomposition"].items():
        print(mode, "variance share by level", np.round(d["by_level"], 3), "code-specific", round(d["code_specific"], 3))
    for mode, d in res["calibration"].items():
        print(mode, {t: {k: v for k, v in r.items() if k != "z_hist"} for t, r in d.items()})
    for mode, d in res["foresight"].items():
        print(mode, "foresight", {k: v for k, v in d.items() if k != "rows"})
    for r in res["multivariate"]["rollup"]:
        print("multivariate", round(r["n"]), r["auc"], r.get("multiplier"))
    t = res["test"]
    for name, b in t["boot"].items():
        print(name, "primary, full data:", {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in b.items()})
    for ev, d in res["curves"].items():
        for mode, c in d.items():
            print("curve", ev, mode, c["codes"], "codes; break-even (log-loss)", round(c["breakeven_ll"], 1),
                  "95% CI over codes", np.round(c["breakeven_ll_ci"], 1))
    for r in t["scaling"]:
        s = r["test"]["all"]
        print("test", r["mode"], r["frac"], r["n_train"], *(f"{e}: {s['auc_' + e]:.3f}/{s['llred_' + e]:+.1f}%"
                                                            for e in ("sib", "tree", "own_mle", "own", "post")))


if __name__ == "__main__":
    main()
