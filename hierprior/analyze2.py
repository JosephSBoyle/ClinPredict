"""Study 2 tables and figures from output/hier2/runs/*.json.

  .venv/Scripts/python -m hierprior.analyze2
"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hierprior.analyze import GRID_C, MUTED, TEXT, agg, style
from hierprior.experiment2 import OUT

MODELS = ["rollup", "hier", "hier0", "l2"]
# same slots as study 1 where the entity is the same (l2 blue, rollup yellow)
COLOR = {"l2": "#2a78d6", "hier": "#eb6834", "hier0": "#1baf7a", "rollup": "#eda100"}
MARK = {"l2": "o", "hier": "s", "hier0": "^", "rollup": "D"}
LABEL = {"rollup": "Ancestor-indicator LR, one L2 penalty (baseline)",
         "hier": "Learned group priors, centred on parent (enhanced)",
         "hier0": "Learned group variances, centred on 0 [exploratory]",
         "l2": "Codes-only L2 LR (study 1 baseline) [reference]"}
SHORT = {"rollup": "baseline", "hier": "enhanced", "hier0": "var-only", "l2": "codes-only"}
SETTING = {"R": "Random training subsets", "T": "Temporal: train < 2020, test 2020-2022"}
PANEL = {"R": "Random", "T": "Temporal"}


def load_runs():
    """(setting, frac) -> model -> [best-on-dev record per seed, seed order]."""
    table = defaultdict(lambda: defaultdict(list))
    for p in sorted((OUT / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        table[(r["setting"], r["frac"])][r["model"]].append(r)
    for d in table.values():
        for rs in d.values():
            rs.sort(key=lambda r: r["seed"])
    return table


def metric(r, name, split="test"):
    return r["best"][split].get(name, np.nan)


def fracs_of(table, st):
    return sorted(f for (s, f) in table if s == st)


def n_train(table, st, f):
    return np.mean([r["n_train"] for r in table[(st, f)]["rollup"]])


def scaling_figure(table, path):
    settings = [s for s in "RT" if any(k[0] == s for k in table)]
    fig, axes = plt.subplots(len(settings), 3, figsize=(15, 3.8 * len(settings)), squeeze=False)
    for i, st in enumerate(settings):
        fr = fracs_of(table, st)
        n = [n_train(table, st, f) for f in fr]
        for j, (name, title) in enumerate((("auc", "test AUROC, all admissions"),
                                           ("auc_aff", "test AUROC, admissions with an unseen code"))):
            ax = axes[i, j]
            for m in MODELS:
                mu, sd = map(np.array, zip(*[agg([metric(r, name) for r in table[(st, f)][m]]) for f in fr]))
                ax.fill_between(n, mu - sd, mu + sd, color=COLOR[m], alpha=0.12, lw=0)
                ax.plot(n, mu, color=COLOR[m], lw=2, ls="-" if m in ("rollup", "hier") else "--",
                        marker=MARK[m], ms=6, mec="white", mew=1.2, label=LABEL[m])
            style(ax)
            ax.set_title(f"{PANEL[st]}: {title}", fontsize=9, color=TEXT, loc="left")
            ax.set_ylabel("AUROC", fontsize=8, color=MUTED)
        ax = axes[i, 2]
        ax.axhline(0, color=MUTED, lw=1)
        for m in ("hier", "hier0"):
            for name, ls in (("auc", "-"), ("auc_aff", ":")):
                d = [agg([metric(a, name) - metric(b, name) for a, b in
                          zip(table[(st, f)][m], table[(st, f)]["rollup"])]) for f in fr]
                mu, sd = map(np.array, zip(*d))
                ax.plot(n, mu, color=COLOR[m], lw=2, ls=ls, marker=MARK[m], ms=6, mec="white", mew=1.2,
                        label=f"{SHORT[m]} - baseline, {'all' if name == 'auc' else 'unseen-code'} adm.")
                ax.fill_between(n, mu - sd, mu + sd, color=COLOR[m], alpha=0.08, lw=0)
        style(ax)
        ax.set_title(f"{PANEL[st]}: paired AUROC difference vs baseline", fontsize=9, color=TEXT, loc="left")
        ax.set_ylabel("AUROC difference", fontsize=8, color=MUTED)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        for ax in axes[i]:
            ax.set_xlabel("training admissions (log scale)", fontsize=8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("30-day readmission (MIMIC-IV, ICD-10-CM diagnoses), ancestor-indicator inputs: "
                 "AUROC vs training size (mean ± 1 sd over 3 subsamples)",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cell(rs, name, digits=3):
    mu, sd = agg([metric(r, name) for r in rs])
    return f"{mu:.{digits}f}±{sd:.{digits}f}" if len(rs) > 1 else f"{mu:.{digits}f}"


def dcell(a, b, name, sign=1):
    mu, sd = agg([sign * (metric(x, name) - metric(y, name)) for x, y in zip(a, b)])
    return f"{mu:+.4f}±{sd:.4f}" if len(a) > 1 else f"{mu:+.4f}"


def tables(table):
    md = []
    for st in "RT":
        fr = fracs_of(table, st)
        if not fr:
            continue
        md.append(f"### {SETTING[st]}\n")
        md.append("Test AUROC, all admissions / admissions with an unseen code (mean±sd over seeds).\n")
        md.append("| train adm | unseen codes | affected test adm | " + " | ".join(SHORT[m] for m in MODELS) + " |")
        md.append("|---|---|---|" + "---|" * len(MODELS))
        for f in fr:
            r0 = table[(st, f)]["rollup"]
            md.append(f"| {n_train(table, st, f):,.0f} | {np.mean([r['n_unseen_codes'] for r in r0]):,.0f} | "
                      f"{np.mean([r['n_aff']['test'] for r in r0]):,.0f} / {r0[0]['n_eval']['test']:,} | "
                      + " | ".join(f"{cell(table[(st, f)][m], 'auc')} / {cell(table[(st, f)][m], 'auc_aff')}"
                                   for m in MODELS) + " |")
        md.append("\nPaired differences vs the baseline (same subsample): AUROC all / unseen-code, "
                  "and log-loss improvement (baseline minus model, nats/admission) all / unseen-code.\n")
        md.append("| train adm | enhanced AUROC | enhanced log-loss | var-only AUROC | var-only log-loss |")
        md.append("|---|---|---|---|---|")
        for f in fr:
            d, b = table[(st, f)], table[(st, f)]["rollup"]
            md.append(f"| {n_train(table, st, f):,.0f} | "
                      + " | ".join(f"{dcell(d[m], b, 'auc')} / {dcell(d[m], b, 'auc_aff')} | "
                                   f"{dcell(d[m], b, 'll', -1)} / {dcell(d[m], b, 'll_aff', -1)}"
                                   for m in ("hier", "hier0")) + " |")
        md.append("\nImputation ablation (enhanced model, best dev alpha): unseen codes and groups "
                  "take their parent's weight vs weight 0, on test admissions with an unseen code.\n")
        md.append("| train adm | AUROC imputed | AUROC zeroed | gain | log-loss gain (nats) |")
        md.append("|---|---|---|---|---|")
        for f in fr:
            rs = table[(st, f)]["hier"]
            a1 = [r["best"]["test"]["auc_aff"] for r in rs]
            a0 = [r["best"]["test_zeroed"]["auc_aff"] for r in rs]
            l1 = [r["best"]["test"]["ll_aff"] for r in rs]
            l0 = [r["best"]["test_zeroed"]["ll_aff"] for r in rs]
            g, gs = agg(np.subtract(a1, a0))
            lg, lgs = agg(np.subtract(l0, l1))
            md.append(f"| {n_train(table, st, f):,.0f} | {np.mean(a1):.4f} | {np.mean(a0):.4f} | "
                      f"{g:+.4f}±{gs:.4f} | {lg:+.4f}±{lgs:.4f} |")
        md.append("\nChosen hyperparameter per seed (lambda for baseline / codes-only, alpha otherwise): "
                  + "; ".join(f"{n_train(table, st, f):,.0f}: " + ", ".join(
                      f"{SHORT[m]} {[r['best']['hp'] for r in table[(st, f)][m]]}" for m in MODELS)
                      for f in fr) + "\n")
    long = [(k, d) for k, d in sorted(table.items()) if "hier_em50" in d]
    if long:
        md.append("### [Exploratory] Longer empirical Bayes (50 EM steps), random setting\n")
        md.append("Test AUROC, all admissions / admissions with an unseen code.\n")
        md.append("| train adm | baseline | enhanced (10 EM) | enhanced (50 EM) | var-only (10 EM) | var-only (50 EM) |")
        md.append("|---|---|---|---|---|---|")
        for (st, f), d in long:
            md.append(f"| {n_train(table, st, f):,.0f} | " + " | ".join(
                f"{cell(d[m], 'auc')} / {cell(d[m], 'auc_aff')}"
                for m in ("rollup", "hier", "hier_em50", "hier0", "hier0_em50")) + " |")
        md.append("")
    secs = defaultdict(list)
    for (st, f), d in table.items():
        for m, rs in d.items():
            if m not in MODELS:
                continue
            secs[(f, m)] += [r["seconds"]["total"] for r in rs]
    md.append("### Run time (seconds per job: data load, empirical Bayes, 9-point grid, scoring; "
              "mean over settings and seeds)\n")
    md.append("| train frac | " + " | ".join(SHORT[m] for m in MODELS) + " |")
    md.append("|---|" + "---|" * len(MODELS))
    for f in sorted({k[0] for k in secs}):
        md.append(f"| {f} | " + " | ".join(f"{np.mean(secs[(f, m)]):.0f}" for m in MODELS) + " |")
    return "\n".join(md)


def main():
    table = load_runs()
    scaling_figure(table, OUT / "scaling.png")
    md = tables(table)
    (OUT / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
