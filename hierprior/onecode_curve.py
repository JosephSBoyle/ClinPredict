"""Study 3, learning curves: how much is the relatives' prior worth, in units
of the code's own data?

For every code c with >= M_MIN training admissions (full training split,
chapter Z excluded) we give the learner exactly m of c's training admissions,
drawn at random (hypergeometric: k_m positives out of m), and fit w_c three ways:

  hier   posterior under the tree prior N(m_c, P_c) built from c's relatives
         (leave-c-out, from a training subset of size N)
  naive  posterior under the zero-centred prior N(0, s2): prevalence plus the
         code's own data, no hierarchy (s2 from the same fit)
  mle    data only: log((k+.5)/(m-k+.5)) - b0, no prior (0 when m = 0)

and score them on c's dev pairs: log-loss and Brier per pair, and AUROC over
the pooled pairs of all eligible codes (each scored by its code's w). The
prevalence (w = 0) is the reference. m = 0 is the relatives-only prior for
hier and the prevalence for naive and mle. Averaged over R random draws.

The relatives' data N (the training subset behind the tree prior) is varied
separately from m, over the saved seed-0 fits of hierprior.onecode.

  .venv/Scripts/python -m hierprior.onecode_curve
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from hierprior.data import load
from hierprior.experiment2 import splits
from hierprior.onecode import (OUT, brier, counts, logloss, map_update, not_z,
                               primary_design)

M_GRID = [0, 1, 2, 3, 5, 10, 20, 50, 100]
M_MIN = 100
REL_FRACS = [0.03, 0.1, 0.3, 1.0]
R = 30


def auc(nE, kE, w):
    return float(roc_auc_score(np.r_[np.ones(len(w)), np.zeros(len(w))], np.r_[w, w],
                               sample_weight=np.r_[kE, nE - kE]))


def main():
    X, y, subj, split, hier = load()
    yg = np.zeros(len(y), np.int8)
    tr, dv, _ = splits("R", 1.0, 0, subj, split, yg)
    keep = not_z(hier)
    rng = np.random.default_rng(0)
    out = []
    for mode, Xm in {"any": X, "primary": primary_design(hier)}.items():
        nF, kF = (a * keep for a in counts(Xm, y, tr))
        nE, kE = (a * keep for a in counts(Xm, y, dv))
        el = np.flatnonzero((nF >= M_MIN) & (nE > 0))
        nF, kF, nE, kE = nF[el], kF[el], nE[el], kE[el]
        draws = {m: [rng.hypergeometric(kF.astype(int), (nF - kF).astype(int), m).astype(float) if m else np.zeros(len(el)) for _ in range(R)]
                 for m in M_GRID}
        for frac in REL_FRACS:
            z = np.load(OUT / f"{mode}_{frac}_est.npz")
            b0, s2 = float(z["b0"]), float(np.exp(z["theta"][-1]))
            mt, Pt = z["tree"][el], z["tree_P"][el]
            ref = {"ll": logloss(nE, kE, 0.0, b0).sum() / nE.sum(),
                   "brier": brier(nE, kE, 0.0, b0).sum() / nE.sum()}
            for m in M_GRID:
                acc = {e: {"ll": [], "brier": [], "auc": []} for e in ("hier", "naive", "mle")}
                for k in draws[m]:
                    n = np.full(len(el), float(m))
                    ws = {"hier": map_update(mt, Pt, n, k, b0),
                          "naive": map_update(np.zeros(len(el)), np.full(len(el), s2), n, k, b0),
                          "mle": np.log((k + .5) / (n - k + .5)) - b0 if m else np.zeros(len(el))}
                    for e, w in ws.items():
                        acc[e]["ll"].append(logloss(nE, kE, w, b0).sum() / nE.sum())
                        acc[e]["brier"].append(brier(nE, kE, w, b0).sum() / nE.sum())
                        acc[e]["auc"].append(auc(nE, kE, w) if np.ptp(w) > 0 else 0.5)
                    if m == 0:
                        break                   # no randomness at m = 0
                rec = {"mode": mode, "rel_frac": frac, "n_rel": int(round(frac * tr.sum())), "m": m,
                       "codes": int(len(el)), "pairs": int(nE.sum()), "ref": ref}
                for e, d in acc.items():
                    rec[e] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()}
                out.append(rec)
                print(mode, frac, m, *(f"{e}: ll={rec[e]['ll'][0]:.4f} auc={rec[e]['auc'][0]:.3f}"
                                       for e in acc), flush=True)
    (OUT / "curve.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
