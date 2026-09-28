"""Tables and figures from output/hier/runs/*.json and univariate.json.

  .venv/Scripts/python -m hierprior.analyze
"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from hierprior.experiment import OUT

# reference categorical palette, fixed slot per model (never re-ordered)
COLOR = {"l2": "#2a78d6", "hier_b1": "#eb6834", "hier_b0": "#1baf7a", "rollup": "#eda100",
         "baseline": "#2a78d6", "sibling_mean": "#4a3aa7"}
LABEL = {"l2": "Baseline LR (L2)", "hier_b1": "Hierarchical prior (spec, beta=1)",
         "hier_b0": "Hierarchical prior (beta=0) [exploratory]",
         "rollup": "Ancestor-indicator LR [reference]",
         "baseline": "Baseline (prevalence prior)", "sibling_mean": "Sibling mean [reference]"}
MARK = {"l2": "o", "hier_b1": "s", "hier_b0": "^", "rollup": "D", "baseline": "o", "sibling_mean": "v"}
PROTO = {"A": "Protocol A (literal): 1 sibling held out per final-level group",
         "B": "Protocol B: 1 sibling held out in 10% of groups",
         "N": "Exploratory N: no hold-out, codes unseen in the training subset"}
TEXT, MUTED, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs():
    runs = [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))]
    table = defaultdict(lambda: defaultdict(list))  # (proto, frac) -> model -> [rec]
    meta = {}
    for r in runs:
        key = (r["protocol"], r["frac"])
        meta.setdefault(key, []).append(r)
        for m, grid in r["models"].items():
            best = max(grid, key=lambda g: g["dev"]["auc"])
            table[key][m].append({"hp": best["hp"], **{f"{s}_{k}": v for s in ("dev", "test")
                                                        for k, v in best[s].items()}})
    return table, meta


def agg(vals):
    v = np.array(vals, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def style(ax):
    ax.set_xscale("log")
    ax.grid(True, color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def scaling_figure(table, meta, protocols, models, path):
    fig, axes = plt.subplots(len(protocols), 2, figsize=(11, 3.4 * len(protocols)), squeeze=False)
    for i, pr in enumerate(protocols):
        fracs = sorted(f for (p, f) in table if p == pr)
        n = [np.mean([r["n_train"] for r in meta[(pr, f)]]) for f in fracs]
        for j, (metric, title) in enumerate((("test_auc", "all test admissions"),
                                             ("test_auc_aff", "test admissions with a held-out/unseen code"))):
            ax = axes[i, j]
            for m in models:
                if not all(m in table[(pr, f)] for f in fracs):
                    continue
                mu, sd = zip(*[agg([r[metric] for r in table[(pr, f)][m]]) for f in fracs])
                mu, sd = np.array(mu), np.array(sd)
                ax.fill_between(n, mu - sd, mu + sd, color=COLOR[m], alpha=0.12, lw=0)
                ax.plot(n, mu, color=COLOR[m], lw=2, marker=MARK[m], ms=6,
                        mec="white", mew=1.2, label=LABEL[m])
            style(ax)
            ax.set_title(f"{pr}: test AUROC, {title}", fontsize=9, color=TEXT, loc="left")
            ax.set_xlabel("training admissions (log scale)", fontsize=8, color=MUTED)
            ax.set_ylabel("AUROC", fontsize=8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("30-day readmission (MIMIC-IV, ICD-10-CM diagnoses): AUROC vs training size "
                 "(mean ± 1 sd over 3 subsamples)", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06 if len(protocols) > 1 else 0.12, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown_table(table, meta, pr, models):
    fracs = sorted(f for (p, f) in table if p == pr)
    lines = ["| train frac | train adm | " + " | ".join(f"{LABEL[m]} all / aff" for m in models) + " |",
             "|---|---|" + "---|" * len(models)]
    for f in fracs:
        n = int(np.mean([r["n_train"] for r in meta[(pr, f)]]))
        cells = []
        for m in models:
            rs = table[(pr, f)].get(m, [])
            if not rs:
                cells.append("-")
                continue
            a, sa = agg([r["test_auc"] for r in rs])
            b, sb = agg([r.get("test_auc_aff", np.nan) for r in rs])
            cells.append(f"{a:.3f}±{sa:.3f} / {b:.3f}±{sb:.3f}")
        lines.append(f"| {f} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_deltas(table, pr, a, b, metric):
    """mean and sd over seeds of (a - b), per fraction."""
    out = {}
    for (p, f), d in sorted(table.items()):
        if p != pr or a not in d or b not in d:
            continue
        diff = [x[metric] - y[metric] for x, y in zip(d[a], d[b])]
        out[f] = agg(diff) + (len(diff),)
    return out


def imputation_ablation(protocols, models=("hier_b1", "hier_b0")):
    """Test AUROC on affected admissions with the held-out/unseen codes'
    imputed weights kept vs zeroed (what the baseline does), same model
    otherwise. AUROC ignores the intercept, so the saved weights suffice."""
    from hierprior.data import load
    X, y, subj, split, hier = load()
    te = split == 2
    Xte, yte = X[te], y[te]
    out = defaultdict(lambda: defaultdict(list))
    for p in sorted((OUT / "runs").glob("*_w.npz")):
        pr, f, seed = p.stem[:-2].split("_")
        if pr not in protocols:
            continue
        z = np.load(p)
        ood = z["ood"]
        aff = np.asarray(Xte[:, ood].sum(1)).ravel() > 0
        for m in models:
            w = z[m]
            w0 = w.copy()
            w0[ood] = 0.0
            a1 = roc_auc_score(yte[aff], Xte[aff] @ w)
            a0 = roc_auc_score(yte[aff], Xte[aff] @ w0)
            out[(pr, float(f))][m].append((a1, a0))
    return out


def univariate_figure(path):
    rows = json.loads((OUT / "univariate.json").read_text())
    models = ["baseline", "sibling_mean", "hier_b1", "hier_b0"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    for i, pr in enumerate(("A", "B")):
        rs = [r for r in rows if r["protocol"] == pr]
        fracs = sorted({r["frac"] for r in rs})
        n = [np.mean([r["n_train"] for r in rs if r["frac"] == f]) for f in fracs]
        for j, (metric, ylab) in enumerate((("dll_per_occ", "test log-lik gain per occurrence (nats)"),
                                            ("occ_auc", "AUROC over held-out-code occurrences"))):
            ax = axes[i, j]
            for m in models:
                mu, sd = zip(*[agg([r[f"{m}/test"][metric] for r in rs if r["frac"] == f]) for f in fracs])
                mu, sd = np.array(mu), np.array(sd)
                ax.fill_between(n, mu - sd, mu + sd, color=COLOR[m], alpha=0.12, lw=0)
                ax.plot(n, mu, color=COLOR[m], lw=2, marker=MARK[m], ms=6, mec="white", mew=1.2,
                        label=LABEL[m] if m != "hier_b0" else "Hierarchical prior (beta=0) [exploratory]")
            style(ax)
            ax.set_title(f"{pr}: one-feature models of held-out codes", fontsize=9, color=TEXT, loc="left")
            ax.set_xlabel("training admissions (log scale)", fontsize=8, color=MUTED)
            ax.set_ylabel(ylab, fontsize=8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("One-input-feature regime: each held-out code alone vs the prevalence prior",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def main():
    table, meta = load_runs()
    protos = [p for p in "ABN" if any(k[0] == p for k in table)]
    models = ["l2", "hier_b1", "hier_b0", "rollup"]
    scaling_figure(table, meta, [p for p in protos if p in "AB"], ["l2", "hier_b1"],
                   OUT / "scaling_main.png")
    scaling_figure(table, meta, protos, models, OUT / "scaling_all_models.png")
    md = []
    for pr in protos:
        md.append(f"### {PROTO[pr]}\n\n" + markdown_table(table, meta, pr, models))
        for other in ("l2",):
            for m in ("hier_b1", "hier_b0", "rollup"):
                for metric in ("test_auc", "test_auc_aff"):
                    d = paired_deltas(table, pr, m, other, metric)
                    md.append(f"- {m} - {other}, {metric}: " + ", ".join(
                        f"{f}: {mu:+.4f}±{sd:.4f}" for f, (mu, sd, k) in d.items()))
        md.append("- chosen hyperparameters (per frac, per seed): " + "; ".join(
            f"{f}: " + ", ".join(f"{m}={[r['hp'] for r in table[(pr, f)][m]]}" for m in models
                                 if m in table[(pr, f)])
            for (p, f) in sorted(table) if p == pr))
        md.append("- n held-out/unseen codes, affected test admissions: " + ", ".join(
            f"{f}: {meta[(pr, f)][0]['n_ood_codes']} codes, {meta[(pr, f)][0]['n_aff']['test']} adm"
            for (p, f) in sorted(meta) if p == pr))
    abl = imputation_ablation(protos)
    md.append("\n### Imputation ablation: test AUROC on affected admissions, imputed vs zeroed OOD weights\n")
    md.append("| protocol | frac | model | imputed | zeroed | gain (mean±sd over seeds) |")
    md.append("|---|---|---|---|---|---|")
    for (pr, f), d in sorted(abl.items()):
        for m, v in d.items():
            v = np.array(v)
            g = agg(v[:, 0] - v[:, 1])
            md.append(f"| {pr} | {f} | {m} | {v[:, 0].mean():.4f} | {v[:, 1].mean():.4f} | {g[0]:+.4f}±{g[1]:.4f} |")
    if (OUT / "univariate.json").exists():
        rows = univariate_figure(OUT / "univariate.png")
        md.append("### One-feature regime (test)\n")
        md.append("| protocol | frac | model | codes (>=20 occ) | dll/occ | occurrence AUROC | sign acc | codes improved |")
        md.append("|---|---|---|---|---|---|---|---|")
        for pr in ("A", "B"):
            for f in sorted({r["frac"] for r in rows if r["protocol"] == pr}):
                rs = [r for r in rows if r["protocol"] == pr and r["frac"] == f]
                for m in ("baseline", "sibling_mean", "hier_b1", "hier_b0"):
                    s = [r[f"{m}/test"] for r in rs]
                    md.append(f"| {pr} | {f} | {m} | {s[0]['n_codes']} ({s[0]['n_codes_20']}) | "
                              f"{np.mean([x['dll_per_occ'] for x in s]):+.4f} | "
                              f"{np.mean([x['occ_auc'] for x in s]):.3f} | "
                              f"{np.mean([x['sign_acc'] for x in s]):.3f} | "
                              f"{np.mean([x['frac_codes_improved'] for x in s]):.2f} |")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
