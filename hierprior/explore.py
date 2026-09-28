"""EXPLORATORY: are the imputed weights of held-out codes any good?

Protocol B, full retained training set. For each held-out code, compare the
weight the hierarchical model imputes (it never saw the code) with the
"oracle" weight an L2 LR learns when the code's patients are kept (every
training patient; lambda = 30, the dev-optimal value at >= 10k admissions in
every protocol). (The protocol-N full-data run that would have supplied a
hierarchical-model oracle was killed by memory pressure and not rerun.) Same for the
one-feature regime, where the oracle is the code's own unpooled univariate
estimate on the full training set.

Also: test AUROC of hier vs L2 split by the number of held-out codes an
admission carries.

Uses the weights saved by hierprior.experiment / hierprior.univariate
(AUROC ignores the intercept, so the weights suffice).

  .venv/Scripts/python -m hierprior.explore
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from hierprior.analyze import COLOR, GRID_C, MUTED, TEXT, style
from hierprior.data import load
from hierprior.experiment import OUT, held_out, train_rows
from hierprior.models import HierUni, L2LR


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def compare(imp, orc, wts):
    return {"n_codes": int(len(imp)),
            "pearson": float(pearsonr(imp, orc)[0]), "spearman": float(spearmanr(imp, orc)[0]),
            "support_weighted_r": wcorr(imp, orc, wts),
            "sign_agreement": float((np.sign(imp) == np.sign(orc)).mean()),
            "rmse_imputed": float(np.sqrt(np.average((imp - orc) ** 2, weights=wts))),
            "rmse_zero_baseline": float(np.sqrt(np.average(orc ** 2, weights=wts)))}


def main():
    X, y, subj, split, hier = load()
    H = held_out("B", hier)
    zB = np.load(OUT / "runs" / "B_1.0_0_w.npz")
    rowsN = train_rows("N", 1.0, 0, X, subj, split, H[:0])
    supN = np.asarray(X[rowsN][:, H].sum(0)).ravel()
    keep = supN >= 20
    res = {"min_full_train_support": 20}

    pairs = {}
    orc = L2LR(30).fit(X[rowsN], y[rowsN]).w[H][keep]
    for m in ("hier_b1", "hier_b0", "rollup"):
        imp = zB[m][H][keep]
        res[f"multivariate/{m}"] = compare(imp, orc, supN[keep])
        pairs[m] = (imp, orc)

    uB = np.load(OUT / "univariate" / "B_1.0_0_w.npz")
    uo = HierUni(hier, pool=False).fit(X[rowsN], y[rowsN]).w[H][keep]
    for m in ("hier_b1", "hier_b0", "sibling_mean"):
        res[f"univariate/{m}"] = compare(uB[m][H][keep], uo, supN[keep])
    pairs["uni"] = (uB["hier_b1"][H][keep], uo)

    te = split == 2
    Xte, yte = X[te], y[te]
    nh = np.asarray(Xte[:, H].sum(1)).ravel()
    by = {}
    for lo, hi, name in ((0, 0, "0"), (1, 1, "1"), (2, 99, "2+")):
        mk = (nh >= lo) & (nh <= hi)
        by[name] = {"n": int(mk.sum()), "prev": float(yte[mk].mean()),
                    **{m: float(roc_auc_score(yte[mk], Xte[mk] @ zB[m]))
                       for m in ("l2", "hier_b1", "hier_b0")}}  # rollup: saved weights are per-leaf, not its binarised design
    res["test_auc_by_n_heldout_codes"] = by

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (key, t) in zip(axes, (("hier_b1", "Multivariate hierarchical LR (beta=1)"),
                                   ("uni", "One-feature regime (beta=1)"))):
        a, b = pairs[key]
        s = 10 + 60 * np.sqrt(supN[keep] / supN[keep].max())
        ax.scatter(b, a, s=s, color=COLOR["hier_b1"], alpha=0.6, edgecolor="white", lw=0.6)
        lo, hi = min(a.min(), b.min()) - 0.1, max(a.max(), b.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.axhline(0, color=GRID_C, lw=1, zorder=0)
        ax.axvline(0, color=GRID_C, lw=1, zorder=0)
        style(ax)
        ax.set_xscale("linear")
        ax.set_xlabel("oracle weight (code's patients kept; L2 LR / own univariate fit)", fontsize=8, color=MUTED)
        ax.set_ylabel("imputed weight (code never seen in training)", fontsize=8, color=MUTED)
        ax.set_title(f"{t}\nPearson r = {pearsonr(a, b)[0]:.2f}, {len(a)} codes", fontsize=9,
                     color=TEXT, loc="left")
    fig.suptitle("EXPLORATORY: protocol B held-out codes with >= 20 occurrences in the full training set "
                 "(marker area ~ support; the baseline imputes 0)", fontsize=9, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "explore_imputed_vs_oracle.png", dpi=150)
    (OUT / "explore.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
