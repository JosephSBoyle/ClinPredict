"""Tables, figures and drill-downs for study 3 (one-code mode, chapter Z
excluded).

  .venv/Scripts/python -m hierprior.analyze3

Units. Log-loss is the mean negative natural-log probability that a model
gave to the outcome that happened (binary cross-entropy, in nats per
prediction; divide by ln 2 = 0.693 for bits). The prevalence-only model scores
H(p0), the entropy of the base rate; "log-loss reduction" is the percentage of
that removed. AUROC is over the scored predictions (in primary mode, one per
admission). Brier is the mean squared error of the predicted probability.

Writes output/hier3/summary.md, scaling.png, curve.png, scatter.png,
drilldown_any.png, drilldown_primary.png, the primary-only onecode_scaling.png
and onecode_curve.png, and artifact_data.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from hierprior.analyze import GRID_C, MUTED, TEXT, style
from hierprior.data import load
from hierprior.experiment2 import splits
from hierprior.onecode import ESTIMATORS, OUT, counts, logloss, not_z, primary_design

COLOR = {"tree": "#eb6834", "own": "#2a78d6", "post": "#1baf7a", "sib": "#4a3aa7",
         "own_mle": "#8f8e88", "null": "#52514e"}
LABEL = {"null": "Prevalence only", "sib": "Naive prior: raw sibling average",
         "tree": "Hierarchical prior only (relatives)", "own_mle": "Own data only, no prior (MLE)",
         "own": "Own data + naive prior N(0, s2)", "post": "Own data + hierarchical prior"}
CURVE = {"hier": ("post", "Own data + hierarchical prior"), "naive": ("own", "Own data + naive prior N(0, s2)"),
         "mle": ("own_mle", "Own data only, no prior (MLE)")}
MODES = {"any": "Every code occurrence", "primary": "Primary diagnosis only (one code per admission)"}
SHORT = {"any": "Every code occurrence", "primary": "Primary diagnosis only"}
BIN_LABEL = {"n1-10": "1-9", "n10-100": "10-99", "n100-1000": "100-999", "n1000-1000000000": ">= 1000"}
DRILL = {"any": ["E87", "C78", "D63", "N18", "K76", "I502"],
         "primary": ["F32", "I61", "C78", "K573", "I502", "K859"]}
DRILL_MIN = {"any": 5, "primary": 3}
DRILL_FRACS = [0.1, 1.0]
SCATTER_MIN = {"any": 200, "primary": 100}


def descriptions():
    f = Path.home() / ".cache/pyhealth/medcode/ICD10CM.csv"
    if not f.exists():
        return {}
    d = pl.read_csv(f, columns=["code", "name"])
    return dict(zip(d["code"].str.replace(".", "", literal=True).to_list(), d["name"].to_list()))


def derive(s):
    """Add log-loss reduction (% of the prevalence model's) and rho."""
    for e in ESTIMATORS:
        s[f"llred_{e}"] = 100 * (s["ll_null"] - s[f"ll_{e}"]) / s["ll_null"]
    d_own = s["ll_null"] - s["ll_own"]
    s["rho"] = (s["ll_null"] - s["ll_tree"]) / d_own if d_own > 0 else np.nan
    return s


def aggregate(R):
    """(mode, frac) -> eval -> group -> key -> [mean, sd] over seeds."""
    by = defaultdict(list)
    for r in R:
        by[(r["mode"], r["frac"])].append(r)
    out = {}
    for key, rs in sorted(by.items()):
        res = {"n_train": float(np.mean([r["n_train"] for r in rs])), "seeds": len(rs)}
        for ev in ("train", "dev"):
            res[ev] = {}
            for g in rs[0][ev]:
                ss = [derive(dict(r[ev][g])) for r in rs if g in r[ev]]
                res[ev][g] = {k: [float(np.nanmean([s.get(k, np.nan) for s in ss])),
                                  float(np.nanstd([s.get(k, np.nan) for s in ss]))] for k in ss[0]}
        out[key] = res
    return out


def m_equivalent(C, mode, frac, metric):
    """Own admissions the naive-prior learner needs to match the hierarchical
    prior with none (interpolated on log(1 + m))."""
    rows = sorted([c for c in C if c["mode"] == mode and c["rel_frac"] == frac], key=lambda c: c["m"])
    target = rows[0]["hier"][metric][0]
    xs = np.log1p([c["m"] for c in rows])
    ys = np.array([c["naive"][metric][0] for c in rows])
    better = (ys <= target) if metric in ("ll", "brier") else (ys >= target)
    if not better.any():
        return None
    i = int(np.argmax(better))
    if i == 0:
        return 0.0
    f = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
    return float(np.expm1(xs[i - 1] + f * (xs[i] - xs[i - 1])))


def f3(v):
    return f"{v:.3f}"


def tables(A, C):
    L = ["# Study 3 (one-code mode, chapter Z excluded): full tables", "",
         "Log-loss: mean negative natural-log probability given to the observed outcome (nats per",
         "prediction; bits = nats / 0.693). Reduction: % of the prevalence-only model's log-loss removed.",
         "AUROC over the scored predictions (primary mode: one per admission). Mean over 3 training",
         "subsamples (1 at full data); sd in brackets where shown.", ""]
    for mode in MODES:
        for ev in ("dev", "train"):
            L += [f"## {MODES[mode]}: {ev}, all predictions", "",
                  "| train adm | predictions | AUROC: " + " / ".join(ESTIMATORS[1:]) + " | log-loss prevalence | "
                  + " | ".join(f"log-loss {e} (red.)" for e in ("sib", "tree", "own_mle", "own", "post")) + " |",
                  "|---" * 8 + "|"]
            for (md, f), res in A.items():
                if md != mode:
                    continue
                s = res[ev]["all"]
                L.append(f"| {res['n_train']:,.0f} | {s['pairs'][0]:,.0f} | "
                         + " / ".join(f3(s[f"auc_{e}"][0]) for e in ESTIMATORS[1:])
                         + f" | {s['ll_null'][0]:.4f} | "
                         + " | ".join(f"{s[f'll_{e}'][0]:.4f} ({s[f'llred_{e}'][0]:+.1f}%)"
                                      for e in ("sib", "tree", "own_mle", "own", "post")) + " |")
            L.append("")
        L += [f"## {MODES[mode]}: dev at full data, by the code's number of training admissions", "",
              "| support | codes | predictions | AUROC tree / own / post | log-loss red. tree / own / post |",
              "|---|---|---|---|---|"]
        res = A[(mode, 1.0)]["dev"]
        for g, lab in [("unseen", "0 (unseen)")] + list(BIN_LABEL.items()):
            s = res.get(g)
            if s is None:
                continue
            L.append(f"| {lab} | {s['codes'][0]:,.0f} | {s['pairs'][0]:,.0f} | "
                     + " / ".join(f3(s[f"auc_{e}"][0]) for e in ("tree", "own", "post")) + " | "
                     + " / ".join(f"{s[f'llred_{e}'][0]:+.1f}%" for e in ("tree", "own", "post")) + " |")
        L.append("")
        L += [f"## {MODES[mode]}: learning curve (dev, codes with >= 100 training admissions)", "",
              "| relatives' training adm | m own adm | AUROC hier / naive (= MLE) | log-loss hier / naive / MLE |",
              "|---|---|---|---|"]
        for c in C:
            if c["mode"] == mode:
                L.append(f"| {c['n_rel']:,} | {c['m']} | {f3(c['hier']['auc'][0])} / {f3(c['naive']['auc'][0])} | "
                         f"{c['hier']['ll'][0]:.4f} / {c['naive']['ll'][0]:.4f} / {c['mle']['ll'][0]:.4f} |")
        L += ["", "Own admissions the naive-prior learner needs to match the hierarchical prior with none:", ""]
        for frac in sorted({c["rel_frac"] for c in C}):
            L.append(f"* relatives from {frac:.0%} of training: log-loss {m_equivalent(C, mode, frac, 'll')}, "
                     f"AUROC {m_equivalent(C, mode, frac, 'auc')}")
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


def scaling(A, modes=tuple(MODES), fname="scaling.png"):
    fig, axes = plt.subplots(len(modes), 2, figsize=(12, 4.3 * len(modes)), squeeze=False)
    for i, mode in enumerate(modes):
        keys = [k for k in A if k[0] == mode]
        n = np.array([A[k]["n_train"] for k in keys])
        for j, (metric, ylab) in enumerate((("auc", "dev AUROC"), ("llred", "dev log-loss reduction vs\nprevalence-only (%)"))):
            ax = axes[i, j]
            for e in ("post", "own", "tree", "own_mle", "sib"):
                mu = np.array([A[k]["dev"]["all"][f"{metric}_{e}"][0] for k in keys])
                sd = np.array([A[k]["dev"]["all"][f"{metric}_{e}"][1] for k in keys])
                ax.fill_between(n, mu - sd, mu + sd, color=COLOR[e], alpha=0.12, lw=0)
                ax.plot(n, mu, color=COLOR[e], lw=2, marker="o", ms=5, mec="white", label=LABEL[e],
                        ls="--" if e in ("own_mle", "sib") else "-")
            if metric == "llred":
                ax.axhline(0, color=MUTED, lw=1)
                ax.set_ylim(-8, 6 if mode == "any" else 6)
            else:
                ax.set_ylim(0.495, 0.67)
            ax.set_ylabel(ylab, color=TEXT)
            ax.set_title(MODES[mode], color=TEXT, loc="left", fontsize=11)
            style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("training admissions (log scale)", color=TEXT)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def curve_plot(C, modes=tuple(MODES), fname="curve.png"):
    fig, axes = plt.subplots(len(modes), 2, figsize=(12, 4.3 * len(modes)), squeeze=False)
    for i, mode in enumerate(modes):
        for j, (metric, ylab) in enumerate((("auc", "dev AUROC"), ("ll", "dev log-loss (nats per prediction)"))):
            ax = axes[i, j]
            for frac, alpha in ((1.0, 1.0), (0.03, 0.45)):
                rows = sorted([c for c in C if c["mode"] == mode and c["rel_frac"] == frac], key=lambda c: c["m"])
                x = np.log1p([c["m"] for c in rows])
                for e in ("hier", "naive", "mle"):
                    if frac != 1.0 and e != "hier":
                        continue
                    lab = CURVE[e][1] + ("" if frac == 1.0 else ", relatives from ~5k adm")
                    if metric == "auc" and e == "mle":
                        lab += " (same ranking as naive)"
                    ax.plot(x, [c[e][metric][0] for c in rows], color=COLOR[CURVE[e][0]], lw=2, alpha=alpha,
                            marker="o", ms=4.5, mec="white", ls="--" if e == "mle" else "-", label=lab)
            if metric == "ll":
                ref = next(c for c in C if c["mode"] == mode and c["rel_frac"] == 1.0)["ref"]["ll"]
                ax.axhline(ref, color=MUTED, lw=1)
                ax.text(x[-1], ref + 0.0005, "prevalence only", ha="right", va="bottom", fontsize=8, color=MUTED)
                lo = min(c["hier"]["ll"][0] for c in C if c["mode"] == mode)
                ax.set_ylim(lo - 0.002, ref + (0.004 if mode == "any" else 0.012))
            ticks = [0, 1, 2, 5, 10, 20, 50, 100]
            ax.set_xticks(np.log1p(ticks), [str(t) for t in ticks])
            ax.set_ylabel(ylab, color=TEXT)
            ax.set_title(f"{SHORT[mode]}: {rows[0]['codes']} codes with >= 100 training adm", color=TEXT,
                         loc="left", fontsize=10.5)
            ax.grid(True, color=GRID_C, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("admissions of the code's own data given to the learner (m)", color=TEXT)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def load_est(mode, frac):
    z = np.load(OUT / f"{mode}_{frac}_est.npz")
    return {k: z[k] for k in z.files}


def ci_logodds(n, k, b0):
    w = np.log((k + .5) / (n - k + .5)) - b0
    return w, 1.96 * np.sqrt(1 / (k + .5) + 1 / (n - k + .5))


def drill_data(designs, y, subj, split, hier, desc):
    out = {}
    keep = not_z(hier)
    _, dv, _ = splits("R", 1.0, 0, subj, split, np.zeros(len(y), np.int8))
    for mode, Xm in designs.items():
        nE, kE = (a * keep for a in counts(Xm, y, dv))
        out[mode] = {}
        for frac in DRILL_FRACS:
            z = load_est(mode, frac)
            b0 = float(z["b0"])
            for g in DRILL[mode]:
                node = hier.idx[g]
                rows = []
                for j in (j for j, v in enumerate(hier.leaf_node) if hier.parent[v] == node):
                    n, k = float(z["nT"][j]), float(z["kT"][j])
                    if n < DRILL_MIN[mode] and nE[j] < DRILL_MIN[mode]:
                        continue
                    we, ce = ci_logodds(nE[j], kE[j], b0) if nE[j] > 0 else (np.nan, np.nan)
                    wm, cm = ci_logodds(n, k, b0) if n > 0 else (np.nan, np.nan)
                    ll = {e: float(logloss(nE[j], kE[j], z[e][j], b0)) for e in ("null", "tree", "own", "post")}
                    rows.append({"code": hier.leaves[j], "desc": desc.get(hier.leaves[j], ""),
                                 "n_train": int(n), "rate_train": k / n if n else None,
                                 "n_dev": int(nE[j]), "rate_dev": float(kE[j] / nE[j]) if nE[j] else None,
                                 "own_mle": float(wm), "own_mle_ci": float(cm), "own": float(z["own"][j]),
                                 "tree": float(z["tree"][j]), "tree_ci": float(1.96 * np.sqrt(z["tree_P"][j])),
                                 "post": float(z["post"][j]), "dev_w": float(we), "dev_ci": float(ce),
                                 "p_tree": float(1 / (1 + np.exp(-(b0 + z["tree"][j])))),
                                 "p_own": float(1 / (1 + np.exp(-(b0 + z["own"][j])))),
                                 "ll": ll})
                tot = {e: sum(r["ll"][e] for r in rows) for e in ("null", "tree", "own", "post")}
                pairs = sum(r["n_dev"] for r in rows)
                red = {e: 100 * (tot["null"] - tot[e]) / tot["null"] for e in tot}
                d_own = tot["null"] - tot["own"]
                out[mode][f"{g}|{frac}"] = {
                    "group": g, "name": desc.get(g, g), "frac": frac, "b0": b0, "p0": float(1 / (1 + np.exp(-b0))),
                    "rows": rows, "dev_pairs": pairs, "llred": red,
                    "rho": (tot["null"] - tot["tree"]) / d_own if d_own > 0 else None}
    return out


def drill_plot(D, mode, frac=1.0):
    keys = [k for k in D[mode] if D[mode][k]["frac"] == frac]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, key in zip(axes.ravel(), keys):
        d = D[mode][key]
        rows = d["rows"]
        yy = np.arange(len(rows))[::-1]
        for r, yv in zip(rows, yy):
            ax.errorbar(r["own_mle"], yv + 0.18, xerr=r["own_mle_ci"], fmt="o", color=COLOR["own"], ms=5, lw=1.2)
            if np.isfinite(r["dev_w"]):
                ax.errorbar(r["dev_w"], yv - 0.18, xerr=r["dev_ci"], fmt="o", mfc="white", color=TEXT, ms=5, lw=1)
            ax.errorbar(r["tree"], yv, xerr=r["tree_ci"], fmt="D", color=COLOR["tree"], ms=6, lw=2.2, alpha=0.9)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{r['code']}  n={r['n_train']:,}\n{r['desc'][:28]}" for r in rows], fontsize=7)
        ax.set_xlim(-2.5, 3.5)
        red = d["llred"]
        ax.set_title(f"{d['group']}.x {d['name'][:45]}\n dev log-loss reduction: prior {red['tree']:+.1f}%, "
                     f"own {red['own']:+.1f}%", color=TEXT, loc="left", fontsize=10)
        ax.grid(True, axis="x", color=GRID_C, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("log-odds shift from the prevalence", color=TEXT)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COLOR["own"], marker="o", lw=1.2, label="own training estimate ±95%"),
                        Line2D([], [], color=COLOR["tree"], marker="D", lw=2.2, label="hierarchical prior (relatives only) ±95%"),
                        Line2D([], [], color=TEXT, marker="o", mfc="white", lw=1, label="dev estimate ±95%")],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.01, 0.04, 1, 1), w_pad=2)
    fig.savefig(OUT / f"drilldown_{mode}.png", dpi=120)
    plt.close(fig)


def scatter_data(hier, desc):
    out = {}
    for mode in MODES:
        z = load_est(mode, 1.0)
        sel = np.flatnonzero(z["nT"] >= SCATTER_MIN[mode])
        wts, t, o = z["nT"][sel], z["tree"][sel], z["own"][sel]
        pts = [{"code": hier.leaves[j], "desc": desc.get(hier.leaves[j], ""), "n": int(z["nT"][j]),
                "own": float(z["own"][j]), "tree": float(z["tree"][j]),
                "parent": hier.names[hier.parent[hier.leaf_node[j]]]} for j in sel]
        loss = wts * (o - t) ** 2
        out[mode] = {"min_n": SCATTER_MIN[mode], "points": pts,
                     "r2_uncentred": float(1 - np.sum(wts * (o - t) ** 2) / np.sum(wts * o ** 2)),
                     "slope": float(np.sum(wts * o * t) / np.sum(wts * t ** 2)),
                     "corr": float(np.corrcoef(t, o)[0, 1]),
                     "misses": [pts[i] for i in np.argsort(-loss)[:10]]}
    return out


def scatter_plot(S):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (mode, s) in zip(axes, S.items()):
        t = np.array([p["tree"] for p in s["points"]])
        o = np.array([p["own"] for p in s["points"]])
        n = np.array([p["n"] for p in s["points"]])
        ax.scatter(t, o, s=6 + 30 * np.sqrt(n / n.max()), color=COLOR["tree"], alpha=0.45, lw=0)
        lim = [-2, 2.5]
        ax.plot(lim, lim, color=MUTED, lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("hierarchical prior from relatives (log-odds shift)", color=TEXT)
        ax.set_ylabel("code's own estimate (log-odds shift)", color=TEXT)
        ax.set_title(f"{MODES[mode]}\n{len(t):,} codes with >= {s['min_n']} training adm: r = {s['corr']:.2f}, "
                     f"slope = {s['slope']:.2f}, uncentred R^2 = {s['r2_uncentred']:.2f}", color=TEXT,
                     loc="left", fontsize=9.5)
        ax.grid(True, color=GRID_C, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "scatter.png", dpi=130)
    plt.close(fig)


def clean(x):
    """NaN / inf -> None, recursively, so the JSON is strict."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main():
    R = json.loads((OUT / "results.json").read_text())
    C = json.loads((OUT / "curve.json").read_text())
    A = aggregate(R)
    tables(A, C)
    scaling(A)
    curve_plot(C)
    scaling(A, ("primary",), "onecode_scaling.png")
    curve_plot(C, ("primary",), "onecode_curve.png")
    X, y, subj, split, hier = load()
    desc = descriptions()
    D = drill_data({"any": X, "primary": primary_design(hier)}, y, subj, split, hier, desc)
    for mode in MODES:
        drill_plot(D, mode)
    S = scatter_data(hier, desc)
    scatter_plot(S)
    meq = {f"{m}|{f}|{k}": m_equivalent(C, m, f, k) for m in MODES for f in (0.03, 0.1, 0.3, 1.0)
           for k in ("ll", "auc")}
    data = {"agg": [{"mode": k[0], "frac": k[1], **v} for k, v in A.items()], "curve": C, "m_eq": meq,
            "drill": D, "scatter": S}
    (OUT / "artifact_data.json").write_text(json.dumps(clean(data)))
    print(json.dumps(meq, indent=0))
    for mode in MODES:
        for k, v in D[mode].items():
            print(mode, k, v["name"][:40], {e: round(x, 2) for e, x in v["llred"].items()}, len(v["rows"]))
        print(mode, {k: v for k, v in S[mode].items() if k not in ("points", "misses")})
        for p in S[mode]["misses"]:
            print("  ", p["code"], p["desc"][:50], p["n"], round(p["own"], 2), round(p["tree"], 2))


if __name__ == "__main__":
    main()
