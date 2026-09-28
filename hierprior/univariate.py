"""One-input-feature regime (docs/idea.md, "due to collinearity...").

For every held-out code c the model is  logit P(readmit) = b0 + w_c * x_c,
b0 = logit(prevalence). The baseline has no data on c, so w_c = 0 and it
predicts the prevalence for everyone. The hierarchical models impute w_c from
the (univariate) weights of c's relatives. Because each code's model has one
feature there is no collinearity: every code's weight is its marginal effect.

Over the held-out codes with test support we score
  dll      test log-likelihood gain per code occurrence versus the prevalence
           prior (summed over codes, divided by occurrences)
  occ_auc  AUROC over all (test admission, held-out code present) pairs,
           each scored by w_c: do the imputed weights rank the codes' risks?
           The baseline's constant w = 0 scores 0.5.
  sign_acc occurrence-weighted fraction of codes with >= 20 test occurrences
           whose sign(w_c) matches the sign of the code's empirical test
           log-odds shift (w_c = 0 counts as wrong)
alpha is tuned on dev by summed dll.

  .venv/Scripts/python -m hierprior.univariate
"""
import json

import numpy as np
from scipy.special import log_expit
from sklearn.metrics import roc_auc_score

from hierprior.data import load
from hierprior.experiment import FRACS, OUT, SEEDS, held_out, train_rows
from hierprior.models import HierUni

ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100]


def per_code(Xe, ye, w, b0, codes):
    """dll per code, test support n, positives k, w of each code."""
    n = np.asarray(Xe[:, codes].sum(0)).ravel()
    k = np.asarray(Xe[:, codes].T @ ye).ravel()
    wc = w[codes]
    z1 = b0 + wc
    dll = (k * (log_expit(z1) - log_expit(b0)) + (n - k) * (log_expit(-z1) - log_expit(-b0)))
    return dll, n, k, wc


def sibling_mean(hier, w_unpooled, support, H):
    """Unweighted mean of the supported siblings' unpooled weights."""
    out = np.zeros(len(w_unpooled))
    node2col = {v: j for j, v in enumerate(hier.leaf_node)}
    ch = hier.children()
    for j in H:
        p = hier.parent[hier.leaf_node[j]]
        sib = [node2col[v] for v in ch[p] if hier.is_leaf[v] and node2col[v] != j]
        sib = [s for s in sib if support[s] > 0]
        out[j] = np.mean(w_unpooled[sib]) if sib else 0.0
    return out


def summarize(dll, n, k, wc, b0):
    m = n >= 1
    occ = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
    occ_auc = roc_auc_score(occ, np.r_[wc[m], wc[m]], sample_weight=np.r_[k[m], n[m] - k[m]])
    s = n >= 20
    emp = np.log((k[s] + 0.5) / (n[s] - k[s] + 0.5)) - b0
    right = np.sign(wc[s]) == np.sign(emp)
    return {"n_codes": int(m.sum()), "occurrences": int(n[m].sum()),
            "dll_per_occ": float(dll[m].sum() / max(n[m].sum(), 1)),
            "occ_auc": float(occ_auc),
            "n_codes_20": int(s.sum()),
            "sign_acc": float((right * n[s]).sum() / n[s].sum()),
            "frac_codes_improved": float((dll[m] > 0).mean())}


def main():
    X, y, subj, split, hier = load()
    dv, te = split == 1, split == 2
    rows_out = []
    for protocol in ("A", "B"):
        H = held_out(protocol, hier)
        for frac in FRACS[protocol]:
            for seed in (SEEDS if frac < 1 else [0]):
                rows = train_rows(protocol, frac, seed, X, subj, split, H)
                Xt, yt = X[rows], y[rows]
                support = np.asarray(Xt.sum(0)).ravel()
                # weights are log-odds shifts learned relative to the train
                # prevalence; score them on top of the population prevalence
                # prior (dev), since the drop biases the training prevalence
                b0 = np.log(y[dv].mean() / (1 - y[dv].mean()))
                w_unp = HierUni(hier, pool=False).fit(Xt, yt).w
                ws = {"baseline": np.zeros(len(hier.leaves)),
                      "sibling_mean": sibling_mean(hier, w_unp, support, H)}
                for beta in (1.0, 0.0):
                    best = None
                    hu = HierUni(hier, beta=beta).fit(Xt, yt)
                    for a in ALPHAS:
                        w = hu.with_alpha(a).w.copy()
                        d = per_code(X[dv], y[dv], w, b0, H)[0].sum()
                        if best is None or d > best[0]:
                            best = (d, a, w)
                    ws[f"hier_b{int(beta)}"] = best[2]
                    ws[f"hier_b{int(beta)}_alpha"] = best[1]
                rec = {"protocol": protocol, "frac": frac, "seed": seed, "n_train": int(rows.sum())}
                for name, w in ws.items():
                    if name.endswith("_alpha"):
                        rec[name] = w
                        continue
                    for sname, s in (("dev", dv), ("test", te)):
                        rec[f"{name}/{sname}"] = summarize(*per_code(X[s], y[s], w, b0, H), b0)
                (OUT / "univariate").mkdir(exist_ok=True)
                np.savez_compressed(OUT / "univariate" / f"{protocol}_{frac}_{seed}_w.npz", H=H, **{
                    k: v for k, v in ws.items() if not k.endswith("_alpha")})
                rows_out.append(rec)
                print(protocol, frac, seed, {k: round(rec[f"{k}/test"]["dll_per_occ"], 4)
                                             for k in ("baseline", "sibling_mean", "hier_b1", "hier_b0")},
                      flush=True)
    (OUT / "univariate.json").write_text(json.dumps(rows_out, indent=1))


if __name__ == "__main__":
    main()
