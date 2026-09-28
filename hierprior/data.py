"""Cohort, labels, ICD-10-CM hierarchy and the held-out-sibling split.

Cohort: adult (anchor_age >= 18) MIMIC-IV admissions whose diagnoses are all
ICD-10-CM, excluding in-hospital deaths. Label: any later admission of the same
patient starting within 30 days of this discharge (the next admission may be
coded in either ICD version -- only its timestamp is used).

Hierarchy: character prefixes. root -> first letter -> 3-char category ->
4-char -> ... -> full code. An observed code that is also a prefix of another
observed code (non-billable header codes) becomes a leaf "<code>$" under the
internal node "<code>".
"""
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

ROOT = Path("data/mimiciv/3.1/hosp")
CACHE = Path("cache/hier")


def build_cohort(window_days=30):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "cohort.parquet"
    if out.exists():
        return pl.read_parquet(out), pl.read_parquet(CACHE / "dx.parquet")

    adm = pl.read_csv(ROOT / "admissions.csv.gz", columns=[
        "subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"],
        try_parse_dates=True)
    pat = pl.read_csv(ROOT / "patients.csv.gz", columns=["subject_id", "anchor_age"])
    dx = pl.read_csv(ROOT / "diagnoses_icd.csv.gz",
                     schema_overrides={"icd_code": pl.Utf8, "icd_version": pl.Int8})

    adm = adm.sort(["subject_id", "admittime"]).with_columns(
        next_admit=pl.col("admittime").shift(-1).over("subject_id"))
    adm = adm.with_columns(
        readmit=((pl.col("next_admit") - pl.col("dischtime")) <= pl.duration(days=window_days))
        .fill_null(False).cast(pl.Int8))

    versions = dx.group_by("hadm_id").agg(
        all10=(pl.col("icd_version") == 10).all())
    coh = (adm.join(pat, on="subject_id")
           .join(versions, on="hadm_id")
           .filter(pl.col("all10") & (pl.col("anchor_age") >= 18)
                   & (pl.col("hospital_expire_flag") == 0))
           .select("subject_id", "hadm_id", "admittime", "readmit"))
    dx = (dx.filter(pl.col("icd_version") == 10)
          .join(coh.select("hadm_id"), on="hadm_id")
          .select("subject_id", "hadm_id", pl.col("icd_code").str.strip_chars())
          .unique())
    coh.write_parquet(out)
    dx.write_parquet(CACHE / "dx.parquet")
    return coh, dx


class Hierarchy:
    """Prefix tree over the observed codes. Node 0 is the root.

    Leaves are indexed 0..n_leaves-1 in the design matrix; tree nodes are
    indexed separately with leaf_node[j] giving leaf j's node id.
    """

    def __init__(self, codes):
        codes = sorted(set(codes))
        cs = set(codes)
        prefixes = {c[:k] for c in codes for k in range(1, len(c))}
        self.leaves = codes
        self.leaf_name = [c + "$" if c in prefixes else c for c in codes]
        names = ["<root>"] + sorted(prefixes) + self.leaf_name
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        parent = np.zeros(len(names), dtype=np.int64)
        parent[0] = -1
        for i, n in enumerate(names[1:], 1):
            # "E11$" -> "E11", "E119" -> "E11", "E" -> root
            parent[i] = self.idx[n[:-1]] if len(n) > 1 else 0
        self.parent = parent
        self.leaf_node = np.array([self.idx[n] for n in self.leaf_name])
        self.is_leaf = np.zeros(len(names), bool)
        self.is_leaf[self.leaf_node] = True
        assert (parent[1:] < np.arange(1, len(names))).all()  # parents sort first
        depth = np.zeros(len(names), dtype=np.int64)
        for i in range(1, len(names)):
            depth[i] = depth[parent[i]] + 1
        self.depth = depth
        self.col = {c: j for j, c in enumerate(codes)}
        assert all(c in cs for c in self.leaves)

    @property
    def n_nodes(self):
        return len(self.names)

    def children(self):
        ch = [[] for _ in range(self.n_nodes)]
        for i, p in enumerate(self.parent):
            if p >= 0:
                ch[p].append(i)
        return ch

    def ancestors_matrix(self):
        """(n_leaves, n_nodes) 0/1: leaf j -> its node and every ancestor except root."""
        rows, cols = [], []
        for j, v in enumerate(self.leaf_node):
            while v > 0:
                rows.append(j)
                cols.append(v)
                v = self.parent[v]
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                             shape=(len(self.leaves), self.n_nodes))


def design(coh, dx, hier):
    """Binary admission x leaf-code CSR matrix aligned with coh row order."""
    row = {h: i for i, h in enumerate(coh["hadm_id"].to_list())}
    r = np.fromiter((row[h] for h in dx["hadm_id"].to_list()), np.int64, len(dx))
    c = np.fromiter((hier.col[x] for x in dx["icd_code"].to_list()), np.int64, len(dx))
    X = sp.csr_matrix((np.ones(len(r), np.float64), (r, c)), shape=(len(coh), len(hier.leaves)))
    X.data[:] = 1.0
    return X


def split_patients(subject_ids, fracs=(0.7, 0.1, 0.2), seed=0):
    """0 = train, 1 = dev, 2 = test, by patient."""
    u = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    cut = np.cumsum(np.array(fracs) * len(u)).astype(int)
    lab = np.empty(len(u), np.int8)
    lab[perm[:cut[0]]] = 0
    lab[perm[cut[0]:cut[1]]] = 1
    lab[perm[cut[1]:]] = 2
    return lab[np.searchsorted(u, subject_ids)]


def pick_held_out(hier, seed=0):
    """One random leaf per final-level sibling group (parent whose children
    are all leaves, >= 2 children). Returns leaf column indices."""
    rng = np.random.default_rng(seed)
    ch = hier.children()
    held = []
    node_to_col = {v: j for j, v in enumerate(hier.leaf_node)}
    for p in range(hier.n_nodes):
        kids = ch[p]
        if len(kids) >= 2 and all(hier.is_leaf[k] for k in kids):
            held.append(node_to_col[kids[rng.integers(len(kids))]])
    return np.array(sorted(held))


def load():
    coh, dx = build_cohort()
    hier = Hierarchy(dx["icd_code"].unique().to_list())
    X = design(coh, dx, hier)
    y = coh["readmit"].to_numpy().astype(np.float64)
    subj = coh["subject_id"].to_numpy()
    split = split_patients(subj)
    return X, y, subj, split, hier
