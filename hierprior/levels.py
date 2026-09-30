"""Study 4: a second outcome and the official ICD-10-CM levels.

Study 3's one-code model and its vocabulary prior, unchanged,

    logit P(y | c) = b0 + w_c,    w_c = sum of theta_a over c's ancestors a and c itself,
    theta_a ~ N(0, tau2[group of a]),

run for two outcomes on the same cohort, patient split and training subsets:

  readmit  readmission within 30 days of discharge (studies 1-3)
  mort1y   death within 365 days of discharge (data.mortality_1y)

and over four trees for the sum (data.Hierarchy, data.ICDTree):

  prefix   study 3: first letter -> 2, 3, ... characters -> code; 11 variances
  icd3     the official levels chapter -> block -> code: every chapter, block
           and code has its own learnt coefficient theta, one variance per
           level, and the code's weight is the sum of its three
  icd4     chapter -> block -> 3-character category -> code; 4 variances
  icd_sub  chapter -> block -> category -> 4, 5, 6 characters -> code; 10 variances

Each (outcome, mode, training fraction, seed) job fits all four trees on the
same counts (onecode.estimate: variances by marginal likelihood, leave-code-out
prior N(m_c, P_c), posterior with the code's own data) and scores them on dev.
The test split is not used. Chapter Z is excluded, as in study 3.

Then, from the full-data fits:
  curves       study 3's learning curve and break-even (vocab.curve_codes) for
               every outcome, mode and tree, relatives from ~5k and all training
               admissions, the same draws for every tree
  calibration  coverage of the leave-code-out prior's intervals
  correlogram  model-free correlation of two primary codes' log-odds shifts by
               the deepest ICD level they share, with the icd4 tree's implied values
  coefs        posterior of every chapter's and block's coefficient (icd3 and icd4)

  .venv/Scripts/python -m hierprior.levels [--jobs 8]
"""
import argparse
import json
import os
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit
from scipy.stats import norm

from hierprior.analyze3 import clean, descriptions
from hierprior.data import ICDTree, build_cohort, load, mortality_1y
from hierprior.experiment2 import FRACS, SEEDS, splits
from hierprior.onecode import counts, estimate, not_z, primary_design, score
from hierprior.vocab import LEVELS, NMIN, ancestors_by_depth, curve_codes, curve_summary, own, share_curve

OUT = Path("output/hier4")
OUTCOMES = ("readmit", "mort1y")
TREES = ("prefix", "icd3", "icd4", "icd_sub")
MODES = ("primary", "any")
REL_FRACS = (0.03, 1.0)


def setup():
    X, y, subj, split, hier = load()
    coh, _ = build_cohort()
    trees = {"prefix": hier, "icd3": ICDTree(hier.leaves, category=False), "icd4": ICDTree(hier.leaves),
             "icd_sub": ICDTree(hier.leaves, subcategories=True)}
    assert all(t.leaves == hier.leaves for t in trees.values())
    return {"readmit": y, "mort1y": mortality_1y(coh)}, {"primary": primary_design(hier), "any": X}, subj, split, trees


def est_path(outcome, mode, tree, frac):
    return OUT / "est" / f"{outcome}_{mode}_{tree}_{frac}.npz"


def job(outcome, mode, frac, seed, Xm, y, subj, split, trees, keep):
    tr, dv, _ = splits("R", frac, seed, subj, split, np.zeros(len(y), np.int8))
    nT, kT = (a * keep for a in counts(Xm, y, tr))
    nE, kE = (a * keep for a in counts(Xm, y, dv))
    b0 = float(np.log(kT.sum() / (nT.sum() - kT.sum())))
    recs = []
    for name, t in trees.items():
        est, theta = estimate(t, t.ancestors_matrix().tocsr(), nT, kT, b0)
        rec = {"outcome": outcome, "mode": mode, "tree": name, "frac": frac, "seed": seed,
               "n_train": int(tr.sum()), "b0": b0, "sd": np.exp(theta / 2).round(4).tolist(),
               "dev": score(nT, nE, kE, est, b0)}
        for g in rec["dev"].values():
            for e in ("sib", "tree", "own_mle", "own", "post"):
                g[f"llred_{e}"] = 100 * (g["ll_null"] - g[f"ll_{e}"]) / g["ll_null"]
        if seed == 0:
            np.savez_compressed(est_path(outcome, mode, name, frac), nT=nT, kT=kT, b0=b0, theta=theta, **est)
        recs.append(rec)
    s = {r["tree"]: r["dev"]["all"] for r in recs}
    print(outcome, mode, frac, seed, int(tr.sum()), "dev AUROC own+prior",
          {k: round(v["auc_post"], 4) for k, v in s.items()}, "prior only",
          {k: round(v["auc_tree"], 4) for k, v in s.items()}, flush=True)
    return recs


def load_est(outcome, mode, tree, frac=1.0):
    z = np.load(est_path(outcome, mode, tree, frac))
    return {k: z[k] for k in z.files}


def curves(ys, designs, subj, split, keep):
    tr, dv, _ = splits("R", 1.0, 0, subj, split, np.zeros(len(subj), np.int8))
    out = []
    for outcome in OUTCOMES:
        for mode in MODES:
            for tree in TREES:
                for rf in REL_FRACS:
                    c = curve_codes(designs[mode], ys[outcome], tr, dv, keep, mode, np.random.default_rng(0),
                                    z=load_est(outcome, mode, tree, rf))
                    out.append({"outcome": outcome, "mode": mode, "tree": tree, "rel_frac": rf, **curve_summary(c)})
                    print("curve", outcome, mode, tree, rf, out[-1]["codes"], "codes, break-even",
                          round(out[-1]["breakeven_ll"], 1), np.round(out[-1]["breakeven_ll_ci"], 1), flush=True)
    return out


def calibration(keep):
    out = []
    for outcome in OUTCOMES:
        for mode in MODES:
            for tree in TREES:
                z = load_est(outcome, mode, tree)
                nT, kT, b0 = z["nT"], z["kT"], float(z["b0"])
                sel = keep & (nT >= NMIN)
                x, v = own(nT[sel], kT[sel], b0)
                m, P = z["tree"][sel], z["tree_P"][sel]
                zz = (x - m) / np.sqrt(P + v)
                out.append({"outcome": outcome, "mode": mode, "tree": tree, "codes": int(sel.sum()),
                            "coverage": {str(c): float(np.mean(np.abs(zz) < norm.ppf(0.5 + c / 2))) for c in LEVELS},
                            "z_sq": float(np.mean(zz ** 2)), "corr": float(np.corrcoef(m, x)[0, 1])})
    return out


def correlogram(trees, keep):
    """Primary mode: correlation of two codes' log-odds shifts by the deepest
    ICD level they share (0 = different chapter, 1 chapter, 2 block,
    3 category), jackknife over chapters, and the icd4 tree's implied value."""
    t = trees["icd4"]
    anc_all = ancestors_by_depth(t)
    out = {}
    for outcome in OUTCOMES:
        z = load_est(outcome, "primary", "icd4")
        nT, kT, b0 = z["nT"], z["kT"], float(z["b0"])
        sel = keep & (nT >= NMIN)
        x, v = own(nT[sel], kT[sel], b0)
        anc = anc_all[sel]
        u = 1 / (0.4 + v)
        rho, W, var = share_curve(x, v, u, anc)
        chap = anc[:, 1]
        pv = np.array([share_curve(x[chap != g], v[chap != g], u[chap != g], anc[chap != g])[0]
                       for g in np.unique(chap)])
        n = len(pv)
        se = np.sqrt((n - 1) / n * np.nansum((pv - np.nanmean(pv, 0)) ** 2, 0))
        tau2 = np.exp(z["theta"][:4])               # chapter, block, category, code
        out[outcome] = {"codes": int(sel.sum()), "empirical": rho[:4].tolist(), "se": se[:4].tolist(),
                        "model": (np.cumsum(np.r_[0, tau2[:3]]) / tau2.sum()).tolist(),
                        "pairs": (W / np.mean(u) ** 2).round()[:4].tolist(), "true_var": float(var),
                        "model_var": float(tau2.sum()), "tau": np.sqrt(tau2).tolist()}
    return out


def coefs(trees, desc, tree="icd3"):
    """Full data: posterior mean of every chapter's and block's coefficient
    theta_a = mu_a - mu_parent(a), and of mu_a (the sum of the coefficients
    down to a) with its sd, with the training admissions and outcome rate
    under a, for both outcomes and modes."""
    t = trees[tree]
    A = t.ancestors_matrix().tocsc()
    out = {}
    for mode in MODES:
        rows = {}
        for outcome in OUTCOMES:
            z = load_est(outcome, mode, tree)
            nm, nv = z["node_m"], z["node_v"]
            n_node = A.T @ z["nT"]
            k_node = A.T @ z["kT"]
            for a in np.flatnonzero(~t.is_leaf & (t.depth >= 1) & (t.depth <= 2)):
                r = rows.setdefault(t.names[a], {"level": ["chapter", "block"][t.depth[a] - 1],
                                                 "parent": t.names[t.parent[a]], "desc": desc.get(t.names[a], "")})
                r[outcome] = {"n": int(n_node[a]), "rate": float(k_node[a] / n_node[a]) if n_node[a] else None,
                              "mu": float(nm[a]), "theta": float(nm[a] - nm[t.parent[a]]), "sd_mu": float(np.sqrt(nv[a]))}
        out[mode] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--skip-fits", action="store_true")
    args = ap.parse_args()
    (OUT / "est").mkdir(parents=True, exist_ok=True)
    ys, designs, subj, split, trees = setup()
    keep = not_z(trees["prefix"])
    for o, y in ys.items():
        print(o, "base rate", round(float(y.mean()), 4))
    if not args.skip_fits:
        grid = [(o, m, f, s) for o in OUTCOMES for m in MODES for f in FRACS[::-1] for s in (SEEDS if f < 1 else [0])]
        res = Parallel(n_jobs=args.jobs, verbose=0)(
            delayed(job)(o, m, f, s, designs[m], ys[o], subj, split, trees, keep) for o, m, f, s in grid)
        (OUT / "results.json").write_text(json.dumps(clean([r for rs in res for r in rs]), indent=1))
    extra = {"base_rate": {o: float(y.mean()) for o, y in ys.items()},
             "var_kinds": {k: [list(map(int, g)) for g in t.var_kinds] for k, t in trees.items() if k != "prefix"},
             "calibration": calibration(keep), "correlogram": correlogram(trees, keep),
             "coefs": coefs(trees, {**descriptions(), **icd_names()})}
    (OUT / "extra.json").write_text(json.dumps(clean(extra), indent=1))
    (OUT / "curves.json").write_text(json.dumps(clean(curves(ys, designs, subj, split, keep)), indent=1))


def icd_names():
    """Chapter and block titles from PyHealth's ICD10CM.csv (keys like "I30-I5A")."""
    import polars as pl
    from hierprior.data import ICD10CM
    d = pl.read_csv(ICD10CM, columns=["code", "name"]).filter(pl.col("code").str.contains("-"))
    return dict(zip(d["code"].to_list(), d["name"].to_list()))


if __name__ == "__main__":
    main()
