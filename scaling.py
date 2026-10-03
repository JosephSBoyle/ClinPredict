"""Scaling experiment for the torch models in multilevel_model.py.

For each training fraction and seed: subsample training patients, fit each
model for a fixed number of Adam steps, keep the step with the best dev
log-loss, and record its dev metrics. Then plot mean +- SD over seeds against
training size.
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from hierprior.analyze import MUTED, TEXT, style
from hierprior.torch_stub import Data, evaluate
from multilevel_model import LogisticRegression, MultiLevel_ICD10, VariationalMultiLevel

MODELS = {"logistic regression": LogisticRegression,
          "multilevel (MAP)": MultiLevel_ICD10,
          "multilevel (variational)": VariationalMultiLevel}
COLORS = dict(zip(MODELS, ("#2a78d6", "#eb6834", "#1baf7a")))
FRACS = (0.01, 0.03, 0.1, 0.3, 1.0)
SEEDS = (0, 1, 2)


def subsample(d, full_split, frac, seed):
    """Keep a random `frac` of training patients; the rest of train is set to -1 (unused)."""
    split = full_split.copy()
    pats = np.unique(d.subj[split == 0])
    keep = np.random.default_rng(seed).choice(pats, max(1, round(frac * len(pats))), replace=False)
    split[(split == 0) & ~np.isin(d.subj, keep)] = -1
    d.split = split
    d.base_rate = float(d.y_np[split == 0].mean())
    return int((split == 0).sum())


def fit(model, d, steps, lr=1e-2, batch_size=4096, every=50):
    """Adam on BCE + penalty / n_train for `steps` steps; returns dev metrics at the best-dev-log-loss check."""
    n = int((d.split == 0).sum())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    best, step = None, 0
    while step < steps:
        for X, y in d.batches(0, batch_size, seed=step):
            model.train()
            opt.zero_grad()
            (bce(model(X), y) + model.penalty() / n).backward()
            opt.step()
            step += 1
            if step % every == 0 or step == steps:
                model.eval()
                m = evaluate(model, d, 1)
                if best is None or m["ll_reduction_pct"] > best["ll_reduction_pct"]:
                    best = m
            if step == steps:
                break
    return best


def plot(rows, task, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylab in zip(axes, ("auroc", "ll_reduction_pct"),
                             ("dev AUROC", "dev log-loss reduction vs base rate (%)")):
        for name in MODELS:
            r = [x for x in rows if x["model"] == name]
            fs = sorted({x["frac"] for x in r})
            n = [np.mean([x["n_train"] for x in r if x["frac"] == f]) for f in fs]
            v = [[x[key] for x in r if x["frac"] == f] for f in fs]
            mu, sd = np.array([np.mean(a) for a in v]), np.array([np.std(a) for a in v])
            ax.fill_between(n, mu - sd, mu + sd, color=COLORS[name], alpha=0.15, lw=0)
            ax.plot(n, mu, color=COLORS[name], lw=2, marker="o", ms=8, label=name)
        if key == "ll_reduction_pct":
            ax.axhline(0, color=MUTED, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("training admissions (log scale)", color=TEXT)
        ax.set_ylabel(ylab, color=TEXT)
        style(ax)
    axes[0].legend(frameon=False)
    title = {"mortality": "1-year mortality", "readmission": "30-day readmission"}[task]
    fig.suptitle(f"{title}: dev performance vs training size (mean ± SD over {len(SEEDS)} seeds)",
                 color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mortality", choices=["readmission", "mortality"])
    ap.add_argument("--mode", default="any", choices=["any", "primary"])
    ap.add_argument("--tree", default="prefix",
                    choices=["prefix", "official_block", "official_category", "official_prefix"])
    ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args()
    out = Path("output/torch_scaling") / f"{a.task}_{a.mode}_{a.tree}"
    out.mkdir(parents=True, exist_ok=True)

    d = Data(a.task, a.mode, a.tree)
    full_split = d.split.copy()
    rows = []
    for frac in FRACS:
        for seed in SEEDS:
            n_train = subsample(d, full_split, frac, seed)
            for name, cls in MODELS.items():
                torch.manual_seed(seed)
                m = fit(cls(d), d, a.steps)
                rows.append({"model": name, "frac": frac, "seed": seed, "n_train": n_train, **m})
                print(rows[-1], flush=True)
            (out / "results.json").write_text(json.dumps(rows, indent=1))
    plot(rows, a.task, out / "scaling.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
