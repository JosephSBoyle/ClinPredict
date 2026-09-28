"""Scaling experiments: L2 baseline vs hierarchical-prior LR (vs rollup).

  .venv/Scripts/python -m hierprior.experiment [--jobs 12]

Protocols (held-out codes chosen once with seed 0; training patients with
any held-out code are dropped, dev/test untouched):
  A  literal docs/idea.md: one random sibling from every final-level group
  B  one random sibling from a random 10% of final-level groups (A drops 82%
     of training patients; B drops 27%)
  N  exploratory: nothing held out; codes unseen in a small training subset
     are the natural out-of-distribution codes

For each (protocol, training fraction, seed) the training patients left
after the drop are subsampled, every model is fit over its hyperparameter
grid, and dev/test AUROC is logged for every grid point. "affected" = the
dev/test admissions with at least one held-out (or, for N, unseen) code.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from hierprior.data import load, pick_held_out
from hierprior.models import HierLR, L2LR, RollupLR

OUT = Path("output/hier")
_G = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
GRID = {"l2": _G, "rollup": _G, "hier_b0": _G,
        "hier_b1": [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
        # exploratory beta ablation: precision ~ alpha / (1 + n)**beta needs larger alpha
        "hier_b05": [3, 10, 30, 100, 300, 1000, 3000, 10000, 30000],
        "hier_b2": [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]}
FRACS = {"A": [0.03, 0.1, 0.3, 1.0], "B": [0.01, 0.03, 0.1, 0.3, 1.0],
         "N": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}
SEEDS = [0, 1, 2]


def held_out(protocol, hier):
    H = pick_held_out(hier, seed=0)
    if protocol == "B":
        H = H[np.random.default_rng(0).random(len(H)) < 0.1]
    if protocol == "N":
        H = H[:0]
    return H


def train_rows(protocol, frac, seed, X, subj, split, H):
    tr = split == 0
    hasH = np.asarray(X[:, H].sum(1)).ravel() > 0
    pts = np.setdiff1d(np.unique(subj[tr]), np.unique(subj[tr & hasH]))
    if frac < 1:
        pts = np.random.default_rng(seed).choice(pts, max(1, round(frac * len(pts))), replace=False)
    return tr & np.isin(subj, pts)


def scores(y, s, mask):
    out = {"auc": roc_auc_score(y, s), "ap": average_precision_score(y, s)}
    if mask.sum() > 0 and 0 < y[mask].mean() < 1:
        out["auc_aff"] = roc_auc_score(y[mask], s[mask])
        out["ap_aff"] = average_precision_score(y[mask], s[mask])
    return out


def run(protocol, frac, seed, models, sub="runs"):
    t0 = time.time()
    X, y, subj, split, hier = load()
    H = held_out(protocol, hier)
    rows = train_rows(protocol, frac, seed, X, subj, split, H)
    Xt, yt = X[rows], y[rows]
    support = np.asarray(Xt.sum(0)).ravel()
    ood = H if protocol != "N" else np.flatnonzero(support == 0)
    aff = np.asarray(X[:, ood].sum(1)).ravel() > 0
    ev = {s: (X[split == s], y[split == s], aff[split == s]) for s in (1, 2)}

    rec = {"protocol": protocol, "frac": frac, "seed": seed, "n_train": int(rows.sum()),
           "n_train_patients": int(len(np.unique(subj[rows]))), "n_heldout": int(len(H)),
           "n_ood_codes": int(len(ood)), "prev_train": float(yt.mean()),
           "n_aff": {"dev": int(ev[1][2].sum()), "test": int(ev[2][2].sum())},
           "models": {}}
    weights = {}
    for name in models:
        grid = []
        if name.startswith("hier"):
            beta = {"hier_b1": 1.0, "hier_b0": 0.0, "hier_b05": 0.5, "hier_b2": 2.0}[name]
            m = HierLR(hier, beta=beta, n_em=4).fit_em(Xt, yt)
            fit = m.fit
        for hp in GRID[name]:
            if name == "l2":
                m = L2LR(hp).fit(Xt, yt)
            elif name == "rollup":
                m = RollupLR(hier, hp).fit(Xt, yt)
            else:
                fit(hp)
            res = {"hp": hp, "dev": scores(ev[1][1], m.decision(ev[1][0]), ev[1][2]),
                   "test": scores(ev[2][1], m.decision(ev[2][0]), ev[2][2])}
            grid.append(res)
            if res["dev"]["auc"] >= max(g["dev"]["auc"] for g in grid):
                # rollup: a leaf's effective weight is its own + its ancestors' indicators
                weights[name] = (m.A @ m.w if name == "rollup" else m.w).copy()
                if name.startswith("hier"):
                    weights[name + "_sigma2"] = m.sigma2.copy()
        rec["models"][name] = grid
    rec["seconds"] = time.time() - t0
    tag = f"{protocol}_{frac}_{seed}"
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    (OUT / sub / f"{tag}.json").write_text(json.dumps(rec))
    np.savez_compressed(OUT / sub / f"{tag}_w.npz", support=support, ood=ood, **weights)
    print(f"done {tag} n={rows.sum()} {rec['seconds']:.0f}s", flush=True)
    return tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--protocols", default="A,B,N")
    p.add_argument("--models", default="l2,rollup,hier_b1,hier_b0")
    p.add_argument("--fracs", default=None, help="override, e.g. 0.03,0.1")
    p.add_argument("--sub", default="runs", help="output subdirectory")
    a = p.parse_args()
    if a.fracs:
        for k in FRACS:
            FRACS[k] = [float(f) for f in a.fracs.split(",")]
    models = a.models.split(",")
    jobs = [(pr, f, s) for pr in a.protocols.split(",") for f in FRACS[pr]
            for s in (SEEDS if f < 1 else [0])]
    jobs = [j for j in jobs if not (OUT / a.sub / f"{j[0]}_{j[1]}_{j[2]}.json").exists()]
    # biggest first so the long fits don't straggle at the end
    jobs.sort(key=lambda j: -j[1])
    print(f"{len(jobs)} runs", flush=True)
    Parallel(n_jobs=a.jobs, verbose=0)(delayed(run)(*j, models, a.sub) for j in jobs)


if __name__ == "__main__":
    main()
