"""EXPLORATORY analyses for study 2 (chosen after seeing the main results).

E1  Drill-down: how a code's weight is estimated in the enhanced model, for a
    heart-failure group (I50.4x, combined systolic + diastolic HF) and a
    diabetes group (E11.4x, type 2 diabetes with neurological complications),
    at ~5k and at all ~174k training admissions (random setting, seed 0).
    For each code c in the group, with every other weight held at its fitted
    value, the log-likelihood in w_c is ~ Gaussian (Laplace) with mean
    w_hat = w_c - g_c / H_c and variance 1 / H_c (g, H: gradient and curvature
    of the log-loss at the fit). The prior is N(w_group, sigma2_group / alpha).
    The fitted w_c is their precision-weighted combination; an unseen code
    (H_c = 0) sits exactly at the prior mean.
E2  Learned prior spread sigma2 / alpha by hierarchy depth and training size.
E3  Temporal setting: the most frequent codes new in 2020-2022 and the
    log-odds shift each model assigns them (code indicator + its groups').
E4  How temporal the temporal split is: estimated calendar year of the
    training admissions.

  .venv/Scripts/python -m hierprior.explore2
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import expit

from hierprior.analyze import GRID_C, MUTED, TEXT, style
from hierprior.analyze2 import COLOR, LABEL
from hierprior.data import ROOT, build_cohort, load, year_group
from hierprior.experiment2 import OUT, splits
from hierprior.grouped import indicators

GROUPS = {"I504": "I50.4x combined systolic and diastolic heart failure",
          "E114": "E11.4x type 2 diabetes with neurological complications"}
RUNS = [("R_0.03_0", "~5k training admissions"), ("R_1.0_0", "all training admissions")]


def load_run(tag, model):
    z = np.load(OUT / "runs" / f"{tag}_{model}.npz")
    rec = json.loads((OUT / "runs" / f"{tag}_{model}.json").read_text())
    return z, rec


def drilldown(X, y, subj, split, yg, hier, A):
    out = {}
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.8), gridspec_kw={"height_ratios": [1, 2.2]},
                             sharex="col")
    ch = hier.children()
    for j, ((tag, desc), g) in enumerate([(r, g) for g in GROUPS for r in RUNS]):
        st, f, seed = tag.split("_")
        tr, _, _ = splits(st, float(f), int(seed), subj, split, yg)
        zh, rh = load_run(tag, "hier")
        zr, rr = load_run(tag, "rollup")
        alpha = rh["best"]["hp"]
        Z = indicators(X[tr], A)
        p = expit(Z @ zh["w"] + zh["b"])
        H = np.asarray(Z.T @ (p * (1 - p))).ravel()
        grad = np.asarray(Z.T @ (p - y[tr])).ravel()
        gv = hier.idx[g]
        kids = [k for k in ch[gv] if hier.is_leaf[k]]
        mu, var = zh["w"][gv - 1], zh["sigma2"][gv] / alpha
        rows = []
        for k in kids:
            c = k - 1
            rows.append({"code": hier.names[k].rstrip("$"), "n": int(zh["support"][c]),
                         "w": float(zh["w"][c]), "w_rollup": float(zr["w"][c]),
                         "w_data": float(zh["w"][c] - grad[c] / H[c]) if H[c] > 0 else None,
                         "se_data": float(1 / np.sqrt(H[c])) if H[c] > 0 else None})
        out[f"{g}/{tag}"] = {"alpha": alpha, "prior_mean": float(mu), "prior_sd": float(np.sqrt(var)),
                             "rollup_lambda": rr["best"]["hp"], "codes": rows}
        # top: prior densities
        ax = axes[0, j]
        lo = min([mu - 3.5 * np.sqrt(var)] + [r["w_data"] - 2 * r["se_data"] for r in rows if r["w_data"] is not None])
        hi = max([mu + 3.5 * np.sqrt(var)] + [r["w_data"] + 2 * r["se_data"] for r in rows if r["w_data"] is not None])
        lo, hi = max(lo, -1.2), min(hi, 1.2)   # data bars of rare codes run off the edge
        xs = np.linspace(lo, hi, 400)
        dens = np.exp(-0.5 * (xs - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
        ax.fill_between(xs, dens, color=COLOR["hier"], alpha=0.18, lw=0)
        ax.plot(xs, dens, color=COLOR["hier"], lw=2, label="enhanced: N(group weight, sigma2 / alpha)")
        vr = 1 / rr["best"]["hp"]
        ax.plot(xs, np.exp(-0.5 * xs ** 2 / vr) / np.sqrt(2 * np.pi * vr), color=COLOR["rollup"], lw=2,
                ls="--", label="baseline: N(0, 1 / lambda)")
        style(ax)
        ax.set_xscale("linear")
        ax.set_yticks([])
        ax.set_title(f"{g}: {desc}\nprior on a code's weight (sd {np.sqrt(var):.3f})", fontsize=8.5,
                     color=TEXT, loc="left")
        # bottom: per-code forest
        ax = axes[1, j]
        for i, r in enumerate(rows):
            yv = len(rows) - 1 - i
            if r["w_data"] is not None:
                ax.errorbar(r["w_data"], yv + 0.18, xerr=2 * r["se_data"], fmt="o", color=MUTED, mfc="white",
                            ms=6, lw=1.2, capsize=0)
            ax.plot(r["w_rollup"], yv - 0.18, "D", color=COLOR["rollup"], ms=7, mec="white", mew=1.2)
            ax.plot(r["w"], yv, "s", color=COLOR["hier"], ms=8, mec="white", mew=1.2)
        ax.axvline(mu, color=COLOR["hier"], lw=1, ls=":")
        ax.axvline(0, color=MUTED, lw=0.8)
        style(ax)
        ax.set_xscale("linear")
        ax.set_xlim(lo, hi)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['code']} (n={r['n']:,})" for r in rows][::-1], fontsize=8)
        ax.set_xlabel(f"weight of the code's own indicator ({desc})", fontsize=8, color=MUTED)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=MUTED, marker="o", mfc="white", ls="-"),
                Line2D([], [], color=COLOR["hier"], marker="s", ls="", mec="white"),
                Line2D([], [], color=COLOR["rollup"], marker="D", ls="", mec="white")],
               ["data alone: conditional likelihood mean ± 2 sd (others held at the fit)",
                "enhanced fit (prior x data)", "baseline fit"],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("EXPLORATORY: how a code's weight is estimated. Dotted line = prior mean = the group's "
                 "weight; unseen codes (no data bar) sit on it", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(OUT / "explore_drilldown.png", dpi=150)
    plt.close(fig)
    return out


def variance_by_depth(hier):
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    fr = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    for ax, model in zip(axes, ("hier", "hier0")):
        res = {}
        for f in fr:
            z, rec = load_run(f"R_{f}_0", model)
            eff = z["sigma2"] / rec["best"]["hp"]
            has = np.bincount(hier.parent[1:][z["support"] > 0], minlength=hier.n_nodes) > 0
            res[f] = {"n_train": rec["n_train"], "alpha": rec["best"]["hp"],
                      "median_sd_by_parent_depth": {int(d): float(np.sqrt(np.median(eff[(hier.depth == d) & has])))
                                                    for d in range(7) if ((hier.depth == d) & has).any()}}
        out[model] = res
        n = [res[f]["n_train"] for f in fr]
        shades = ["#cde2f7", "#9ec5f0", "#6aa6e6", "#2a78d6", "#1d58a3", "#123a6e", "#0b2446"]
        for d in range(7):
            v = [res[f]["median_sd_by_parent_depth"].get(d, np.nan) for f in fr]
            ax.plot(n, v, color=shades[d], lw=2, marker="o", ms=5, mec="white",
                    label=["root -> letter", "letter -> 2 chars", "2 -> 3 chars (category)", "3 -> 4 chars",
                           "4 -> 5 chars", "5 -> 6 chars", "6 -> 7 chars"][d])
        style(ax)
        ax.set_yscale("log")
        ax.set_title(f"{LABEL[model]}", fontsize=8.5, color=TEXT, loc="left")
        ax.set_xlabel("training admissions (log scale)", fontsize=8, color=MUTED)
    axes[0].set_ylabel("median prior sd of a child's weight\n(sqrt(sigma2 / alpha), groups with data)",
                       fontsize=8, color=MUTED)
    axes[1].legend(fontsize=7, frameon=False, title="parent -> child level", title_fontsize=7)
    fig.suptitle("EXPLORATORY: learned prior spread by hierarchy level (random setting, seed 0)",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "explore_variance_by_depth.png", dpi=150)
    plt.close(fig)
    return out


def new_codes(X, y, split, yg, hier, A):
    zh, _ = load_run("T_1.0_0", "hier")
    zr, _ = load_run("T_1.0_0", "rollup")
    unseen = zh["unseen"]
    te = yg == 4
    n = np.asarray(X[te][:, unseen].sum(0)).ravel()
    k = np.asarray(X[te][:, unseen].T @ y[te]).ravel()
    order = np.argsort(-n)[:12]
    eff_h, eff_r = A @ zh["w"], A @ zr["w"]   # a code alone: its indicator + every group's
    rows = []
    for i in order:
        c = unseen[i]
        rows.append({"code": hier.leaves[c], "test_n": int(n[i]), "test_readmit": float(k[i] / n[i]),
                     "shift_enhanced": float(eff_h[c]), "shift_baseline": float(eff_r[c])})
    return rows


def temporal_years(split, yg):
    coh = build_cohort()[0]
    pat = pl.read_csv(ROOT / "patients.csv.gz", columns=["subject_id", "anchor_year", "anchor_year_group"])
    c = coh.select("subject_id", "admittime").join(pat, on="subject_id", how="left")
    est = (c["admittime"].dt.year() - c["anchor_year"]
           + c["anchor_year_group"].str.slice(0, 4).cast(pl.Int32) + 1).to_numpy()
    tr, te = (yg < 4) & (split != 1), yg == 4
    return {"train_est_year_quantiles_10_50_90": np.quantile(est[tr], [0.1, 0.5, 0.9]).tolist(),
            "train_frac_est_2020_or_later": float((est[tr] >= 2020).mean()),
            "test_est_year_quantiles_10_50_90": np.quantile(est[te], [0.1, 0.5, 0.9]).tolist(),
            "note": "estimated year = admit year - anchor_year + middle of anchor_year_group (+-1 year)"}


def main():
    X, y, subj, split, hier = load()
    yg = year_group(build_cohort()[0])
    A = hier.ancestors_matrix()[:, 1:].tocsr()
    res = {"E1_drilldown": drilldown(X, y, subj, split, yg, hier, A),
           "E2_variance_by_depth": variance_by_depth(hier),
           "E3_new_codes_2020_2022": new_codes(X, y, split, yg, hier, A),
           "E4_temporal_years": temporal_years(split, yg)}
    (OUT / "explore.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "E1_drilldown"}, indent=1))
    for key, v in res["E1_drilldown"].items():
        print(key, {k: v[k] for k in ("alpha", "prior_mean", "prior_sd", "rollup_lambda")})
        for r in v["codes"]:
            print("   ", r)


if __name__ == "__main__":
    main()
