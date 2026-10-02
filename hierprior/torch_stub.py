"""Stub for hand-written torch models of 30-day readmission and 1-year mortality.

Data (from hierprior.data, same cohort and patient split as studies 1-4):

  X      (n_adm, n_codes) 0/1 CSR, admission x ICD-10-CM leaf code
         ("any" diagnosis, or "primary" = seq_num 1 only)
  y      (n_adm,) float 0/1 label of the chosen task
  split  (n_adm,) 0 = train, 1 = dev, 2 = test, by patient
  hier   the code tree (prefix Hierarchy or official ICDTree): parent, depth,
         is_leaf, leaf_node, names, and A = ancestors_matrix(), the
         (n_codes, n_nodes) 0/1 map from a leaf to itself and its ancestors

Subclass HierModel and implement __init__ (parameters) and forward (logits);
optionally override penalty (the negative log prior, added to the loss).

  .venv/Scripts/python -m hierprior.torch_stub --task mort1y --mode any
"""
import argparse

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from sklearn.metrics import log_loss, roc_auc_score

from hierprior.data import ICDTree, build_cohort, load, mortality_1y
from hierprior.onecode import primary_design


def to_torch(M):
    """scipy CSR -> torch sparse CSR (float32)."""
    M = M.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr).long(), torch.from_numpy(M.indices).long(),
                                   torch.from_numpy(M.data), size=M.shape)


class Data:
    """Everything a model needs for one task.
    design mode ("any" or "primary") and tree ("prefix", "official_block", "official_category", "official_prefix")."""

    def __init__(self, task, mode="any", tree="prefix"):
        X, y_readmission, subj, split, hier = load()
        if tree != "prefix":
            hier = ICDTree(hier.leaves, category=tree != "official_block", subcategories=tree == "official_prefix")
        if mode == "primary":
            X = primary_design(hier)
        y = y_readmission if task == "readmission" else mortality_1y(build_cohort()[0])
        self.task, self.mode, self.tree = task, mode, tree
        self.X_sp, self.y_np, self.subj, self.split, self.hier = X.tocsr(), y, subj, split, hier
        self.A_sp = hier.ancestors_matrix().tocsr()            # (n_codes, n_nodes)
        self.n_codes, self.n_nodes = X.shape[1], hier.n_nodes
        self.parent = torch.from_numpy(hier.parent)            # (n_nodes,), root = -1
        self.depth = torch.from_numpy(hier.depth)              # (n_nodes,)
        self.A = to_torch(self.A_sp)
        self.base_rate = float(y[split == 0].mean())

    def part(self, s):
        """(X, y) of split s as torch tensors: sparse CSR X, dense float32 y."""
        m = self.split == s
        return to_torch(self.X_sp[m]), torch.from_numpy(self.y_np[m].astype(np.float32))

    def batches(self, s=0, batch_size=4096, seed=0):
        """Shuffled minibatches of split s; yields (X sparse CSR, y)."""
        idx = np.flatnonzero(self.split == s)
        np.random.default_rng(seed).shuffle(idx)
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            yield to_torch(self.X_sp[b]), torch.from_numpy(self.y_np[b].astype(np.float32))


class MultilevelModel(nn.Module):
    """Base: maps a sparse admission x code batch to one logit per admission."""

    def __init__(self, data: Data):
        super().__init__()
        self.data = data
        self.bias = nn.Parameter(torch.tensor(float(np.log(data.base_rate / (1 - data.base_rate)))))

    def forward(self, X):
        """X: (batch, n_codes) sparse CSR -> (batch,) logits."""
        raise NotImplementedError

    def penalty(self):
        """Negative log prior of the parameters (scalar). Default: none."""
        return torch.zeros(())


def evaluate(model, data, s=1):
    """AUROC and % log-loss reduction vs the training base rate on split s."""
    X, y = data.part(s)
    with torch.no_grad():
        p = torch.sigmoid(model(X)).numpy()
    y = y.numpy()
    ll, ll0 = log_loss(y, p), log_loss(y, np.full_like(p, data.base_rate))
    return {"auroc": roc_auc_score(y, p), "ll_reduction_pct": 100 * (1 - ll / ll0)}


def train(model, data, epochs=10, lr=1e-2, batch_size=4096):
    """Minimise mean BCE + penalty / n_train with SGD."""
    n = int((data.split == 0).sum())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for X, y in data.batches(0, batch_size, seed=ep):
            opt.zero_grad()
            loss = bce(model(X), y) + model.penalty() / n
            loss.backward()
            opt.step()
        model.eval()
        print(ep, evaluate(model, data, 1))
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="readmission", choices=["readmission", "mort1y"])
    ap.add_argument("--mode", default="any", choices=["any", "primary"])
    ap.add_argument("--tree", default="prefix", choices=["prefix", "official_block", "official_category", "official_prefix"])
    a = ap.parse_args()
    d = Data(a.task, a.mode, a.tree)
