"""Tables and figures for study 4 (hierprior.levels): 1-year mortality next to
30-day readmission, and the official ICD-10-CM levels next to the prefix tree.

Metrics: dev AUROC, and log-loss reduction, the percentage of the base-rate
model's log-loss that a model removes (McFadden's R^2 on held-out data; the
base-rate model predicts the training rate of the outcome for everyone).

  .venv/Scripts/python -m hierprior.analyze4

Writes output/hier4/summary.md, scaling.png, trees.png, curve.png,
correlogram.png and chapters.png.
"""
import json
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from hierprior.analyze import GRID_C, MUTED, TEXT, style
from hierprior.levels import MODES, OUT, OUTCOMES, TREES

OUTCOME = {"readmit": "30-day readmission", "mort1y": "1-year mortality"}
LS = {"mort1y": "-", "readmit": "--"}
MODE = {"primary": "Primary diagnosis only (one code per admission)", "any": "Every code occurrence"}
TREE = {"prefix": "prefix tree (study 3)", "official_block": "chapter + block + code",
        "official_category": "chapter + block + category + code", "official_prefix": "chapter + block + category + prefixes + code"}
TREE_COLOR = {"official_block": "#2a78d6", "official_category": "#eb6834", "official_prefix": "#1baf7a"}
EST = {"own": ("Own data + naive prior N(0, s2)", "#2a78d6"),
       "tree": ("Hierarchical prior only (relatives)", "#eb6834"),
       "post": ("Own data + hierarchical prior", "#1baf7a")}


def aggregate(R):
    """(outcome, mode, tree, frac) -> {n_train, key -> [per-seed values]} for dev 'all'."""
    by = defaultdict(lambda: defaultdict(list))
    for r in R:
        d = by[(r["outcome"], r["mode"], r["tree"], r["frac"])]
        d["n_train"].append(r["n_train"])
        d["seed"].append(r["seed"])
        for k, v in r["dev"]["all"].items():
            d[k].append(v)
    return by


def series(A, outcome, mode, tree, key):
    fr = sorted(f for (o, m, t, f) in A if (o, m, t) == (outcome, mode, tree))
    n = np.array([np.mean(A[(outcome, mode, tree, f)]["n_train"]) for f in fr])
    v = [np.array(A[(outcome, mode, tree, f)][key], float) for f in fr]
    return fr, n, v


def legend_outcomes(ax, extra=()):
    h = list(extra) + [Line2D([], [], color=MUTED, ls=LS[o], lw=2, label=OUTCOME[o]) for o in ("mort1y", "readmit")]
    ax.legend(handles=h, frameon=False, fontsize=8.5)


def scaling(A):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    for i, mode in enumerate(("primary", "any")):
        for j, (metric, ylab) in enumerate((("auc", "dev AUROC"),
                                            ("llred", "dev log-loss reduction vs\nbase-rate model (%)"))):
            ax = axes[i, j]
            for o in ("mort1y", "readmit"):
                for e, (lab, col) in EST.items():
                    _, n, v = series(A, o, mode, "prefix", f"{metric}_{e}")
                    mu, sd = np.array([x.mean() for x in v]), np.array([x.std() for x in v])
                    ax.fill_between(n, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
                    ax.plot(n, mu, color=col, ls=LS[o], lw=2, marker="o", ms=5, mec="white")
                _, n, v = series(A, o, mode, "prefix", f"{metric}_post")
                ax.annotate(OUTCOME[o], (n[-1], v[-1].mean()), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=TEXT)
            if metric == "llred":
                ax.axhline(0, color=MUTED, lw=1)
            style(ax)
            ax.set_xlim(300, 9e5)
            ax.set_ylabel(ylab, color=TEXT)
            ax.set_title(MODE[mode], color=TEXT, loc="left", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("training admissions (log scale)", color=TEXT)
    h = [Line2D([], [], color=c, lw=2, label=l) for l, c in EST.values()]
    h += [Line2D([], [], color=MUTED, ls=LS[o], lw=2, label=OUTCOME[o]) for o in ("mort1y", "readmit")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("One-code model on the prefix tree, both outcomes (mean ± 1 sd over 3 training subsets)",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "scaling.png", dpi=130)
    plt.close(fig)


def trees(A):
    """AUROC of each ICD-level tree minus the prefix tree's, paired by training subset."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharey="row")
    for i, mode in enumerate(("primary", "any")):
        for j, e in enumerate(("tree", "post")):
            ax = axes[i, j]
            for o in ("mort1y", "readmit"):
                _, n, base = series(A, o, mode, "prefix", f"auc_{e}")
                for t in TREES[1:]:
                    _, _, v = series(A, o, mode, t, f"auc_{e}")
                    d = [a - b for a, b in zip(v, base)]
                    mu, sd = np.array([x.mean() for x in d]), np.array([x.std() for x in d])
                    ax.fill_between(n, mu - sd, mu + sd, color=TREE_COLOR[t], alpha=0.1, lw=0)
                    ax.plot(n, mu, color=TREE_COLOR[t], ls=LS[o], lw=2, marker="o", ms=5, mec="white")
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_title(f"{MODE[mode]}\n{EST[e][0]}", color=TEXT, loc="left", fontsize=10.5)
            if j == 0:
                ax.set_ylabel("dev AUROC minus the prefix tree's", color=TEXT)
            style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("training admissions (log scale)", color=TEXT)
    legend_outcomes(axes[0, 1], [Line2D([], [], color=TREE_COLOR[t], lw=2, label=TREE[t]) for t in TREES[1:]])
    fig.suptitle("Official ICD-10-CM levels against the prefix tree (paired by training subset; ± 1 sd)",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "trees.png", dpi=130)
    plt.close(fig)


def curve_rows(C, outcome, mode, tree, rf=1.0):
    return next(c for c in C if (c["outcome"], c["mode"], c["tree"], c["rel_frac"]) == (outcome, mode, tree, rf))


def curve_plot(C):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lines = {"naive": ("Own data + naive prior N(0, s2)", "#2a78d6", "prefix"),
             "prefix": ("Own data + prefix-tree prior", "#1baf7a", "prefix"),
             "official_block": ("Own data + chapter/block/code prior", "#eb6834", "official_block")}
    for ax, mode in zip(axes, ("primary", "any")):
        for o in ("mort1y", "readmit"):
            for k, (lab, col, t) in lines.items():
                c = curve_rows(C, o, mode, t)
                x = np.log1p([r["m"] for r in c["rows"]])
                y = [r["llred_naive" if k == "naive" else "llred_hier"] for r in c["rows"]]
                ax.plot(x, y, color=col, ls=LS[o], lw=2, marker="o", ms=4.5, mec="white")
            ax.annotate(OUTCOME[o], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=TEXT)
        ax.axhline(0, color=MUTED, lw=1)
        ticks = [0, 1, 2, 5, 10, 20, 50, 100]
        ax.set_xticks(np.log1p(ticks), [str(t) for t in ticks])
        ax.set_xlim(-0.1, np.log1p(100) + 1.1)
        codes = {o: curve_rows(C, o, mode, "prefix")["codes"] for o in OUTCOMES}
        ax.set_title(f"{MODE[mode]}\ncodes with >= 100 training adm: {codes['readmit']:,}", color=TEXT,
                     loc="left", fontsize=10.5)
        ax.set_xlabel("admissions of the code's own data given to the learner (m)", color=TEXT)
        ax.grid(True, color=GRID_C, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("dev log-loss reduction vs\nbase-rate model (%)", color=TEXT)
    legend_outcomes(axes[0], [Line2D([], [], color=c, lw=2, label=l) for l, c, _ in lines.values()])
    fig.tight_layout()
    fig.savefig(OUT / "curve.png", dpi=130)
    plt.close(fig)


def correlogram_plot(X):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    lab = ["different\nchapter", "same chapter,\ndifferent block", "same block,\ndifferent category",
           "same\ncategory"]
    x = np.arange(4)
    for o, dx in (("mort1y", -0.08), ("readmit", 0.08)):
        g = X["correlogram"][o]
        e, se = np.array(g["empirical"], float), np.array(g["se"], float)
        ax.errorbar(x + dx, e, yerr=1.96 * se, color=TEXT, ls=LS[o], lw=1.5, marker="o", ms=6,
                    mfc=TEXT, mec="white", capsize=3, label=f"{OUTCOME[o]}: model-free ({g['codes']} codes)")
        ax.plot(x + dx, g["model"], color="#eb6834", ls=LS[o], lw=1.5, marker="s", ms=6, mec="white",
                label=f"{OUTCOME[o]}: fitted chapter/block/category/code model")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, lab, fontsize=8.5)
    ax.set_ylabel("correlation of two codes'\nlog-odds shifts (95% jackknife)", color=TEXT)
    ax.set_title("Primary diagnoses: similarity by the deepest ICD level two codes share", color=TEXT,
                 loc="left", fontsize=10.5)
    ax.grid(True, axis="y", color=GRID_C, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "correlogram.png", dpi=130)
    plt.close(fig)


def chapters_plot(X, mode="primary"):
    rows = [(k, r) for k, r in X["coefs"][mode].items()
            if r["level"] == "chapter" and all(r.get(o, {}).get("n", 0) >= 100 for o in OUTCOMES)]
    rows.sort(key=lambda kr: kr[1]["mort1y"]["theta"])
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(rows) + 1.4))
    y = np.arange(len(rows))
    for o, col, mk in (("mort1y", "#2a78d6", "o"), ("readmit", "#eb6834", "s")):
        ax.scatter([r[o]["theta"] for _, r in rows], y, color=col, marker=mk, s=40, edgecolor="white",
                   zorder=3, label=OUTCOME[o])
    for i, (_, r) in enumerate(rows):
        ax.plot([r["readmit"]["theta"], r["mort1y"]["theta"]], [i, i], color=GRID_C, lw=2, zorder=1)
    ax.set_yticks(y, [re.sub(r"\s*\([A-Z0-9-]+\)$", "", r["desc"])[:55] for _, r in rows], fontsize=8)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("chapter coefficient (posterior mean, log-odds shift from the base rate)", color=TEXT)
    ax.set_title("Chapter + block + code model: chapter coefficients\n(primary diagnoses, all training data)",
                 color=TEXT, loc="left", fontsize=10)
    ax.grid(True, axis="x", color=GRID_C, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "chapters.png", dpi=130)
    plt.close(fig)


def cell(A, o, mode, t, f, e):
    d = A[(o, mode, t, f)]
    return f"{np.mean(d['auc_' + e]):.3f} / {np.mean(d['llred_' + e]):+.1f}%"


def tables(A, C, X):
    L = ["# Study 4: full tables", "",
         "Each cell: dev AUROC / log-loss reduction (% of the base-rate model's log-loss removed; the",
         "base-rate model predicts the training rate of the outcome for everyone). Mean over 3 training",
         "subsets (1 at full data). Chapter Z excluded.", ""]
    rates = X["base_rate"]
    L += [f"Base rates over all cohort admissions: readmission {100 * rates['readmit']:.1f}%, "
          f"1-year mortality {100 * rates['mort1y']:.1f}%.", ""]
    for o in OUTCOMES:
        for mode in ("primary", "any"):
            fr = sorted(f for (oo, m, t, f) in A if (oo, m, t) == (o, mode, "prefix"))
            L += [f"## {OUTCOME[o]}, {MODE[mode].lower()}", "",
                  "| training adm | own + naive prior | " + " | ".join(f"prior only: {t}" for t in TREES)
                  + " | " + " | ".join(f"own + prior: {t}" for t in TREES) + " |",
                  "|---" * (2 + 2 * len(TREES)) + "|"]
            for f in fr:
                n = np.mean(A[(o, mode, "prefix", f)]["n_train"])
                L.append(f"| {n:,.0f} | {cell(A, o, mode, 'prefix', f, 'own')} | "
                         + " | ".join(cell(A, o, mode, t, f, "tree") for t in TREES) + " | "
                         + " | ".join(cell(A, o, mode, t, f, "post") for t in TREES) + " |")
            L.append("")
    L += ["## Break-even: own admissions the naive-prior learner needs to match each prior with none", "",
          "Pooled dev log-loss, codes with >= 100 training admissions; 95% bootstrap over codes.", "",
          "| outcome | mode | relatives from | " + " | ".join(TREES) + " |", "|---|---|---|" + "---|" * len(TREES)]
    for o in OUTCOMES:
        for mode in ("primary", "any"):
            for rf in (0.03, 1.0):
                cs = [curve_rows(C, o, mode, t, rf) for t in TREES]
                L.append(f"| {OUTCOME[o]} | {mode} | {'~5k adm' if rf < 1 else 'all 174k'} | "
                         + " | ".join(f"{c['breakeven_ll']:.0f} ({c['breakeven_ll_ci'][0]:.0f}-{c['breakeven_ll_ci'][1]:.0f})"
                                      if c["breakeven_ll"] is not None else "-" for c in cs) + " |")
    L += ["", "## Learning curves, relatives from all training data (log-loss reduction, %)", ""]
    for o in OUTCOMES:
        for mode in ("primary", "any"):
            rows = {t: curve_rows(C, o, mode, t)["rows"] for t in TREES}
            ms = [r["m"] for r in rows["prefix"]]
            L += [f"**{OUTCOME[o]}, {mode}** ({curve_rows(C, o, mode, 'prefix')['codes']} codes)", "",
                  "| m | " + " | ".join(str(m) for m in ms) + " |", "|---" * (len(ms) + 1) + "|",
                  "| own + naive prior | " + " | ".join(f"{r['llred_naive']:+.1f}" for r in rows["prefix"]) + " |"]
            L += [f"| own + {t} prior | " + " | ".join(f"{r['llred_hier']:+.1f}" for r in rows[t]) + " |" for t in TREES]
            L.append("")
    L += ["## Calibration of the leave-code-out prior (codes with >= 30 training adm, full data)", "",
          "| outcome | mode | tree | codes | inside 50% | 80% | 95% | mean z^2 | corr(prior, own) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in X["calibration"]:
        cv = c["coverage"]
        L.append(f"| {OUTCOME[c['outcome']]} | {c['mode']} | {c['tree']} | {c['codes']} | {100 * cv['0.5']:.0f}% | "
                 f"{100 * cv['0.8']:.0f}% | {100 * cv['0.95']:.0f}% | {c['z_sq']:.2f} | {c['corr']:.2f} |")
    L += ["", "## Correlogram by shared ICD level (primary diagnoses, full data)", "",
          "| outcome | codes | level sds (chapter, block, category, code) | different chapter | same chapter | same block | same category |",
          "|---|---|---|---|---|---|---|"]
    for o in OUTCOMES:
        g = X["correlogram"][o]
        L.append(f"| {OUTCOME[o]} | {g['codes']} | {', '.join(f'{s:.2f}' for s in g['tau'])} | "
                 + " | ".join(f"{e:.2f} ± {1.96 * s:.2f} (model {m:.2f})"
                              for e, s, m in zip(g["empirical"], g["se"], g["model"])) + " |")
    L += ["", "## Fitted prior sds at full data (log-odds)", "",
          "Order: the tree's variance groups (hierprior.levels var_kinds), then the naive prior's s.", ""]
    L.append("")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    R = json.loads((OUT / "results.json").read_text())
    C = json.loads((OUT / "curves.json").read_text())
    X = json.loads((OUT / "extra.json").read_text())
    A = aggregate(R)
    scaling(A)
    trees(A)
    curve_plot(C)
    correlogram_plot(X)
    chapters_plot(X)
    tables(A, C, X)
    sds = [f"| {r['outcome']} | {r['mode']} | {r['tree']} | {', '.join(f'{s:.2f}' for s in r['sd'])} |"
           for r in R if r["frac"] == 1.0]
    with open(OUT / "summary.md", "a", encoding="utf-8") as fh:
        fh.write("| outcome | mode | tree | sds |\n|---|---|---|---|\n" + "\n".join(sds) + "\n")


if __name__ == "__main__":
    main()
