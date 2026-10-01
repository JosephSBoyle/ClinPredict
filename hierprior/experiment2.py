"""Study 2 scaling experiments (docs/idea2.md): ancestor-indicator LR with one
L2 penalty ("rollup", (AKA truncate) the baseline) vs learned per-group Gaussian priors
("hier"; "hier0" and study 1's leaf-only "l2" as exploratory references).

  .venv/Scripts/python -m hierprior.experiment2 [--jobs 4]

Settings (no patients are dropped; unseen codes arise naturally):
  R  random: training patients (70% split) subsampled at log-spaced fractions,
     3 seeds; dev / test = the 10% / 20% patient splits.
  T  temporal: train on patients whose anchor_year_group is before 2020-2022
     (the 70% + 20% splits, subsampled likewise), dev = their 10% split,
     test = every patient in the latest bin, 2020-2022.
"Unseen" codes have no training admission; "affected" = dev/test admissions
with at least one unseen code. One job = (setting, fraction, seed, model), so
the grid spreads over cores; each model's hyperparameter grid is walked from
strong to weak regularisation with warm starts.
"""
import argparse
import json
import os
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
from joblib import Parallel, delayed
from scipy.special import log_expit
from sklearn.metrics import roc_auc_score

from hierprior.data import build_cohort, load, year_group
from hierprior.grouped import GroupedLR

OUT = Path("output/hier2")
GRID = {"l2": [10000, 3000, 1000, 300, 100, 30, 10, 3, 1],
        "rollup": [30000, 10000, 3000, 1000, 300, 100, 30, 10, 3],
        "hier": [1000, 300, 100, 30, 10, 3, 1, 0.3, 0.1],
        "hier0": [1000, 300, 100, 30, 10, 3, 1, 0.3, 0.1]}
# exploratory: "<model>_em50" runs empirical Bayes for 50 EM steps (no early stop),
# so that the per-group variances move further from their shared start
EM_LONG = {"n_em": 50, "em_tol": 0.0}
FRACS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2]


def splits(setting, frac, seed, subj, split, yg):
    """Boolean train / dev / test masks over admissions."""
    if setting == "R":
        pool, dev, test = split == 0, split == 1, split == 2
    else:
        early = yg < 4
        pool, dev, test = early & (split != 1), early & (split == 1), yg == 4
    pts = np.unique(subj[pool])
    if frac < 1:
        pts = np.random.default_rng(seed).choice(pts, max(1, round(frac * len(pts))), replace=False)
    return pool & np.isin(subj, pts), dev, test


def scores(y, z, aff):
    ll = -(y * log_expit(z) + (1 - y) * log_expit(-z))
    out = {"auc": roc_auc_score(y, z), "ll": float(ll.mean())}
    if aff.any() and 0 < y[aff].mean() < 1:
        out["auc_aff"] = roc_auc_score(y[aff], z[aff])
        out["ll_aff"] = float(ll[aff].mean())
    return out


def run(setting, frac, seed, model, sub="runs"):
    t0 = time.time()
    X, y, subj, split, hier = load()
    yg = year_group(build_cohort()[0])
    tr, dv, te = splits(setting, frac, seed, subj, split, yg)
    unseen = np.flatnonzero(np.asarray(X[tr].sum(0)).ravel() == 0)
    aff = np.asarray(X[:, unseen].sum(1)).ravel() > 0
    kind = model.split("_")[0]
    m = GroupedLR(hier, kind, **(EM_LONG if model.endswith("_em50") else {})).setup(X[tr], y[tr])
    t_setup = time.time() - t0
    ev = {name: (m.features(X[mk]), y[mk], aff[mk]) for name, mk in (("dev", dv), ("test", te))}
    rec = {"setting": setting, "frac": frac, "seed": seed, "model": model,
           "n_train": int(tr.sum()), "n_train_patients": int(len(np.unique(subj[tr]))),
           "prev_train": float(y[tr].mean()), "n_unseen_codes": int(len(unseen)),
           "n_eval": {k: int(v[1].size) for k, v in ev.items()},
           "n_aff": {k: int(v[2].sum()) for k, v in ev.items()},
           "prev_test": float(y[te].mean()), "grid": []}
    best = None
    for hp in GRID[kind]:
        m.fit(hp)
        res = {"hp": hp, **{k: scores(v[1], v[0] @ m.w + m.b, v[2]) for k, v in ev.items()}}
        rec["grid"].append(res)
        if best is None or res["dev"]["auc"] > best["dev"]["auc"]:
            best = res
            keep = {"w": m.w.copy(), "b": m.b}
            if kind == "hier":
                # imputation ablation: unseen codes and groups get weight 0, as in "rollup"
                Fz, yz, az = ev["test"]
                keep["zeroed"] = scores(yz, Fz @ np.where(m.support > 0, m.w, 0.0) + m.b, az)
    rec["best"] = {k: v for k, v in best.items()}
    if kind == "hier":
        rec["best"]["test_zeroed"] = keep.pop("zeroed")
        rec["em_trace"] = m.em_trace
    rec["newton_iters"] = int(m.nit)
    rec["seconds"] = {"setup": t_setup, "total": time.time() - t0}
    tag = f"{setting}_{frac}_{seed}_{model}"
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    (OUT / sub / f"{tag}.json").write_text(json.dumps(rec))
    extra = {"sigma2": m.sigma2} if kind in ("hier", "hier0") else {}
    np.savez_compressed(OUT / sub / f"{tag}.npz", support=m.support, unseen=unseen, **keep, **extra)
    print(f"done {tag} n={tr.sum()} {rec['seconds']['total']:.0f}s", flush=True)
    return tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--settings", default="R,T")
    p.add_argument("--models", default="rollup,hier,hier0,l2")
    p.add_argument("--fracs", default=None, help="override, e.g. 0.03,0.1")
    p.add_argument("--sub", default="runs", help="output subdirectory")
    a = p.parse_args()
    fracs = [float(f) for f in a.fracs.split(",")] if a.fracs else FRACS
    jobs = [(st, f, s, m) for st in a.settings.split(",") for f in fracs
            for s in (SEEDS if f < 1 else [0]) for m in a.models.split(",")]
    jobs = [j for j in jobs if not (OUT / a.sub / "{}_{}_{}_{}.json".format(*j)).exists()]
    jobs.sort(key=lambda j: -j[1])   # biggest first so the long fits don't straggle
    print(f"{len(jobs)} jobs", flush=True)
    # import by module name so workers unpickle run() by reference, not from __main__
    from hierprior.experiment2 import run as job
    Parallel(n_jobs=a.jobs, verbose=0)(delayed(job)(*j, a.sub) for j in jobs)


if __name__ == "__main__":
    main()
