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


YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]


def year_group(coh):
    """Per admission (coh row order): index into YEAR_GROUPS of the patient's
    anchor_year_group, the 3-year bin their de-identified anchor year falls in."""
    pat = pl.read_csv(ROOT / "patients.csv.gz", columns=["subject_id", "anchor_year_group"])
    g = coh.select("subject_id").join(pat, on="subject_id", how="left")["anchor_year_group"]
    return np.array([YEAR_GROUPS.index(x) for x in g.to_list()], np.int8)


def load():
    coh, dx = build_cohort()
    hier = Hierarchy(dx["icd_code"].unique().to_list())
    X = design(coh, dx, hier)
    y = coh["readmit"].to_numpy().astype(np.float64)
    subj = coh["subject_id"].to_numpy()
    split = split_patients(subj)
    return X, y, subj, split, hier


ICD10CM = Path.home() / ".cache/pyhealth/medcode/ICD10CM.csv"


class ICDTree(Hierarchy):
    """The official ICD-10-CM levels: root -> chapter -> block -> 3-character
    category -> code, from PyHealth's ICD10CM.csv (every category's parent is
    a block such as "I30-I5A", whose parent is a chapter such as "I00-I99").
    With category=False the code hangs directly under its block (chapter ->
    block -> code). With subcategories=True the category is followed by the
    code's longer prefixes (4, 5, 6 characters) as in the prefix tree. A code that is also
    an inner node (a 3-character code under its own category, or a header
    code) becomes the leaf "<code>$". Leaves and their columns are the same
    as the prefix Hierarchy's for the same codes.

    var_group gives every node its prior-variance group: one per (leaf or
    inner node, depth), so the 4-level tree has one variance each for
    chapter, block, category and code."""

    def __init__(self, codes, category=True, subcategories=False):
        d = pl.read_csv(ICD10CM, columns=["code", "parent_code"], schema_overrides={"parent_code": pl.Utf8})
        par = dict(zip(d["code"].to_list(), d["parent_code"].to_list()))
        codes = sorted(set(codes))
        paths = {}
        for c in codes:
            block = par[c[:3]]
            path = [par[block], block] + ([c[:3]] if category else [])
            if subcategories:
                path += [c[:k] for k in range(4, len(c))]
            paths[c] = path
        inner = {a for p in paths.values() for a in p}
        self.leaves = codes
        self.leaf_name = [c + "$" if c in inner else c for c in codes]
        parent_of = {}
        for c, name in zip(codes, self.leaf_name):
            chain = ["<root>"] + paths[c] + [name]
            for a, b in zip(chain[:-1], chain[1:]):
                parent_of[b] = a
        def depth(n):
            k = 0
            while n != "<root>":
                n, k = parent_of[n], k + 1
            return k
        names = ["<root>"] + sorted(parent_of, key=lambda n: (depth(n), n))     # parents first
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        parent = np.array([-1] + [self.idx[parent_of[n]] for n in names[1:]], dtype=np.int64)
        self.parent = parent
        self.leaf_node = np.array([self.idx[n] for n in self.leaf_name])
        self.is_leaf = np.zeros(len(names), bool)
        self.is_leaf[self.leaf_node] = True
        assert (parent[1:] < np.arange(1, len(names))).all()
        dep = np.zeros(len(names), dtype=np.int64)
        for i in range(1, len(names)):
            dep[i] = dep[parent[i]] + 1
        self.depth = dep
        self.col = {c: j for j, c in enumerate(codes)}
        kinds = sorted({(bool(l), int(k)) for l, k in zip(self.is_leaf[1:], dep[1:])})
        self.var_kinds = kinds                     # [(is_leaf, depth)] in theta order
        g = {kd: i for i, kd in enumerate(kinds)}
        self.var_group = np.array([-1] + [g[(bool(l), int(k))] for l, k in zip(self.is_leaf[1:], dep[1:])])


def mortality_1y(coh):
    """Per admission (coh row order): 1 if the patient's date of death is
    within 365 days of this discharge. MIMIC-IV records out-of-hospital deaths
    up to a year after a patient's last discharge, so the label is observed
    for every admission. The cohort already excludes in-hospital deaths."""
    adm = pl.read_csv(ROOT / "admissions.csv.gz", columns=["hadm_id", "dischtime"], try_parse_dates=True)
    pat = pl.read_csv(ROOT / "patients.csv.gz", columns=["subject_id", "dod"], try_parse_dates=True)
    c = (coh.select("subject_id", "hadm_id").join(adm, on="hadm_id", how="left")
         .join(pat, on="subject_id", how="left"))
    days = (pl.col("dod") - pl.col("dischtime").dt.date()).dt.total_days()
    assert (c["hadm_id"] == coh["hadm_id"]).all()
    return c.select((days <= 365).fill_null(False).cast(pl.Float64))[:, 0].to_numpy()
