#!/usr/bin/env python3
# ============================================================
# LINEAR-DIRECTION BASELINE COMPARISON  (companion to abide_pattern_index_FINAL_v3.py)
# ============================================================
# Question (reviewer): is the simple centroid direction as good as shrinkage LDA /
# regularised logistic regression on held-out data? If not, the method should change.
#
# IDENTICAL pipeline to the main v3 script, so numbers are comparable:
#   Fisher-z edges -> [inside each training fold, TD training subjects only]
#   median imputation -> ridge regression of every edge on site dummies (+ age, sex,
#   and head motion if a column exists) -> robust scaling -> PCA (<=150)
#   -> direction w (UNIT NORM, no intercept, oriented ASD > TD) -> score = U @ w
# Same outer folds (seed 42, 10-fold, stratified by diagnosis), same inner 5-fold
# cross-fit giving out-of-sample TD / ASD reference distributions for I, Z and R.
# Held-out ASD AND TD participants are scored once each.
#
# Directions compared (all computed on the SAME preprocessed fold, so differences are
# due to the direction only):
#   centroid_site_balanced  site-balanced ASD-TD centroid contrast (the paper's Stage II)
#   centroid_pooled         pooled ASD-TD centroid contrast (shows effect of site balancing)
#   lda                     shrinkage (Ledoit-Wolf) LDA
#   logistic                L2 logistic regression, class-balanced, C by inner 5-fold CV
#                           (set FIXED_LOGISTIC_C to skip tuning)
#
# Reported per method: Cohen's d and AUC (95% bootstrap CIs) for the raw OOF score, I
# and Z; per-fold AUC; within-site AUC meta-analysis; paired bootstrap CI of the AUC
# difference vs the reference method (logistic); reference calibration (TD mean I and
# ASD mean R should be ~50); edge-pattern stability across folds; rank agreement.
#
# Accuracy / F1 are deliberately NOT reported: a score threshold at zero is
# uncalibrated across methods and would read as diagnostic performance.
# These scores are research descriptors, not diagnostic or severity measures.
#
# Quick test:  BOOTSTRAP_N=200 python linearbaseline_v3.py
# ============================================================
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ---------------- configuration ----------------
DATA_PATH = os.environ.get("ABIDE_FC_PATH", "content/ABIDE_AAL_FC_with_phenotype1.csv")
PHENO_PATH = os.environ.get("ABIDE_PHENO_PATH", "content/Phenotypic_V1_0b.csv")
OUTPUT_DIR = os.environ.get("LINEAR_BASELINE_OUT_DIR", "content/LINEAR_BASELINE_OUTPUT_v3")
SEED = 42
OUTER = 10
INNER = 5
PCA_MAX = 150
RIDGE_ALPHA = 10.0
COV_PENALTY_SCALE = 10.0
LR_C_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]   # same as the main script
FIXED_C = os.environ.get("FIXED_LOGISTIC_C")
BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", 2000))
COVARIATE_ADJUST = os.environ.get("COVARIATE_ADJUST", "1") == "1"
MOTION_COLUMNS = ["func_mean_fd", "FUNC_MEAN_FD", "MEAN_FD", "mean_fd"]
MISSING_CODES = [-9999, -9999.0, "-9999", "-9999.0"]
METHODS = ["centroid_site_balanced", "centroid_pooled", "lda", "logistic"]
REFERENCE_METHOD = "logistic"
os.makedirs(OUTPUT_DIR, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ---------------- statistics helpers ----------------
def normalize_subject_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def percentile(ref, v):
    ref = np.asarray(ref, float); ref = np.sort(ref[np.isfinite(ref)])
    v = np.asarray(v, float)
    out = np.searchsorted(ref, v, side="right") / len(ref) * 100.0
    out[~np.isfinite(v)] = np.nan
    return out


def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def auc_rank(a, b):
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def boot_ci(a, b, fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    for i in range(n_boot):
        v[i] = fn(rng.choice(a, len(a)), rng.choice(b, len(b)))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def effect(score, y, n_boot, seed):
    a, t = score[y == 1], score[y == 0]
    d_lo, d_hi = boot_ci(a, t, cohen_d, n_boot, seed)
    a_lo, a_hi = boot_ci(a, t, auc_rank, n_boot, seed + 1)
    return {"D": cohen_d(a, t), "D_LO": d_lo, "D_HI": d_hi,
            "AUC": auc_rank(a, t), "AUC_LO": a_lo, "AUC_HI": a_hi}


def paired_auc_diff(s1, s2, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    ia, it = np.where(y == 1)[0], np.where(y == 0)[0]
    base = auc_rank(s1[ia], s1[it]) - auc_rank(s2[ia], s2[it])
    v = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(ia, len(ia)); t = rng.choice(it, len(it))
        v[i] = auc_rank(s1[a], s1[t]) - auc_rank(s2[a], s2[t])
    return base, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def dl_meta(d, se):
    d = np.asarray(d, float); se = np.asarray(se, float)
    w = 1 / se ** 2; fixed = (w * d).sum() / w.sum()
    Q = float((w * (d - fixed) ** 2).sum()); df = len(d) - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    ws = 1 / (se ** 2 + tau2); p = (ws * d).sum() / ws.sum(); sp = np.sqrt(1 / ws.sum())
    return p, p - 1.96 * sp, p + 1.96 * sp, (max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0)


def hanley_se(auc, n1, n0):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return float(np.sqrt(max(v, 1e-12)))


def site_auc_meta(score, y, sites):
    aucs, ses = [], []
    for s in sorted(np.unique(sites)):
        a = score[(sites == s) & (y == 1)]; t = score[(sites == s) & (y == 0)]
        if len(a) < 3 or len(t) < 3:
            continue
        au = auc_rank(a, t); aucs.append(au); ses.append(hanley_se(au, len(a), len(t)))
    if len(aucs) < 2:
        return {}
    p, lo, hi, i2 = dl_meta(aucs, ses)
    return {"SITE_RE_AUC": p, "SITE_RE_AUC_LO": lo, "SITE_RE_AUC_HI": hi,
            "SITE_RE_AUC_I2": i2, "SITES_USED": len(aucs)}


def pair_corr_mean(P):
    c = np.corrcoef(P)
    return float(c[np.triu_indices(len(P), 1)].mean())


def unit(v):
    v = np.asarray(v, float).ravel()
    return v / np.linalg.norm(v)


# ---------------- pipeline (same as main script) ----------------
class Prep:
    """Imputer -> ridge(site [+ age, sex, motion]) -> robust scaler -> PCA; TD training only."""

    def __init__(self, seed=SEED):
        self.seed = seed

    def _design(self, sites, C):
        D = (sites[:, None] == self.levels[None, :]).astype(float)
        if self.use_cov:
            Cc = np.where(np.isfinite(C), C, self.cov_med)
            D = np.hstack([D, COV_PENALTY_SCALE * (Cc - self.cov_mu) / self.cov_sd])
        return D

    def fit(self, X, y, sites, C):
        td = y == 0
        self.use_cov = C is not None
        if self.use_cov:
            self.cov_med = np.nanmedian(C[td], axis=0)
            Ct = np.where(np.isfinite(C), C, self.cov_med)
            self.cov_mu = Ct[td].mean(0); self.cov_sd = Ct[td].std(0, ddof=1)
            self.cov_sd[self.cov_sd == 0] = 1.0
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[td])
        X = self.imputer.transform(X)
        self.levels = np.unique(sites)
        D = self._design(sites, C)
        self.ridge = Ridge(alpha=RIDGE_ALPHA).fit(D[td], X[td])
        X = X - self.ridge.predict(D)
        self.scaler = RobustScaler().fit(X[td])
        Z = self.scaler.transform(X)
        k = min(PCA_MAX, int(td.sum()) - 1, Z.shape[1])
        self.pca = PCA(n_components=k, svd_solver="randomized", random_state=self.seed).fit(Z[td])
        return self

    def transform(self, X, sites, C):
        X = self.imputer.transform(X)
        X = X - self.ridge.predict(self._design(sites, C))
        return self.pca.transform(self.scaler.transform(X))

    def back(self, w):
        scaled = self.pca.components_.T @ w
        sc = np.asarray(self.scaler.scale_, float).copy(); sc[sc == 0] = 1.0
        return scaled / sc, scaled


def centroid_dir(U, y, sites, balanced):
    if not balanced:
        return unit(U[y == 1].mean(0) - U[y == 0].mean(0))
    a, t = [], []
    for s in np.unique(sites):
        m = sites == s
        if (m & (y == 1)).sum() and (m & (y == 0)).sum():
            a.append(U[m & (y == 1)].mean(0)); t.append(U[m & (y == 0)].mean(0))
    return unit(np.mean(a, 0) - np.mean(t, 0))


def fit_directions(U, y, sites, seed):
    """All directions: unit norm, no intercept, oriented so ASD training scores > TD."""
    d = {"centroid_site_balanced": centroid_dir(U, y, sites, True),
         "centroid_pooled": centroid_dir(U, y, sites, False),
         "lda": unit(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(U, y).coef_)}
    if FIXED_C:
        lr = LogisticRegression(C=float(FIXED_C), class_weight="balanced",
                                max_iter=3000).fit(U, y)
        c_sel = float(FIXED_C)
    else:
        lr = LogisticRegressionCV(
            Cs=LR_C_GRID, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            scoring="roc_auc", class_weight="balanced", max_iter=3000, n_jobs=1).fit(U, y)
        c_sel = float(np.ravel(lr.C_)[0])
    d["logistic"] = unit(lr.coef_)
    for k, w in d.items():
        s = U @ w
        if s[y == 1].mean() < s[y == 0].mean():
            d[k] = -w
    return d, c_sel


def inner_reference(X, y, sites, C, seed):
    skf = StratifiedKFold(INNER, shuffle=True, random_state=seed)
    ref = {m: np.full(len(y), np.nan) for m in METHODS}
    for a, b in skf.split(X, y):
        prep = Prep(seed).fit(X[a], y[a], sites[a], None if C is None else C[a])
        dirs, _ = fit_directions(prep.transform(X[a], sites[a], None if C is None else C[a]),
                                 y[a], sites[a], seed)
        Ub = prep.transform(X[b], sites[b], None if C is None else C[b])
        for m, w in dirs.items():
            ref[m][b] = Ub @ w
    return ref


# ---------------- load ----------------
def main():
    log("Loading data")
    df = pd.read_csv(DATA_PATH)
    df["SUB_ID"] = normalize_subject_id(df["SUB_ID"])
    df = df[pd.to_numeric(df["DX_GROUP"], errors="coerce").isin([1, 2])].reset_index(drop=True)
    y = (pd.to_numeric(df["DX_GROUP"]).values == 1).astype(int)
    sites = np.asarray(df["SITE_ID"].fillna("UNKNOWN").astype(str).values, dtype=object)
    fc = [c for c in df.columns if str(c).startswith("AAL_FC__")] or \
         [c for c in df.columns if "__" in str(c)]
    X = df[fc].apply(pd.to_numeric, errors="coerce").values.astype(float)
    X[~np.isfinite(X)] = np.nan
    asd, td = y == 1, y == 0
    log(f"N={len(y)} ASD={asd.sum()} TD={td.sum()} sites={len(np.unique(sites))} edges={X.shape[1]}")

    C = None; cov_names = []
    if COVARIATE_ADJUST and os.path.exists(PHENO_PATH):
        ph = pd.read_csv(PHENO_PATH)
        ph["SUB_ID"] = normalize_subject_id(ph["SUB_ID"])
        ph = ph.drop_duplicates("SUB_ID").replace(MISSING_CODES, np.nan)
        mg = df[["SUB_ID"]].merge(ph, on="SUB_ID", how="left")
        age = pd.to_numeric(mg["AGE_AT_SCAN"], errors="coerce").values
        sex = pd.to_numeric(mg["SEX"], errors="coerce").values
        cols = [age, np.where(np.isfinite(sex), (sex == 2).astype(float), np.nan)]
        cov_names = ["AGE", "SEX_F"]
        for mc in MOTION_COLUMNS:
            if mc in mg.columns:
                cols.append(pd.to_numeric(mg[mc], errors="coerce").values); cov_names.append("MOTION")
                break
        C = np.column_stack(cols)
    log(f"Edge nuisance regression on: site + {cov_names if cov_names else 'NOTHING (site only)'}")
    if "MOTION" not in cov_names:
        print("NOTE: no head-motion measure available; motion is not regressed (limitation).")

    # ---------------- outer CV ----------------
    raw = {m: np.full(len(y), np.nan) for m in METHODS}
    I = {m: np.full(len(y), np.nan) for m in METHODS}
    Zs = {m: np.full(len(y), np.nan) for m in METHODS}
    R = {m: np.full(len(y), np.nan) for m in METHODS}
    W = {m: [] for m in METHODS}
    fold = np.zeros(len(y), int); c_sel_list = []
    skf = StratifiedKFold(OUTER, shuffle=True, random_state=SEED)
    for f, (tr, te) in enumerate(skf.split(X, y), start=1):
        Ctr = None if C is None else C[tr]; Cte = None if C is None else C[te]
        prep = Prep(SEED).fit(X[tr], y[tr], sites[tr], Ctr)
        dirs, c_sel = fit_directions(prep.transform(X[tr], sites[tr], Ctr), y[tr], sites[tr], SEED)
        c_sel_list.append(c_sel)
        Ute = prep.transform(X[te], sites[te], Cte)
        ref = inner_reference(X[tr], y[tr], sites[tr], Ctr, SEED + 100 * f)
        fold[te] = f
        for m, w in dirs.items():
            s = Ute @ w
            raw[m][te] = s
            r_td = ref[m][y[tr] == 0]; r_asd = ref[m][y[tr] == 1]
            I[m][te] = percentile(r_td, s)
            Zs[m][te] = (s - r_td.mean()) / r_td.std(ddof=1)
            a_te = y[te] == 1
            R[m][te[a_te]] = percentile(r_asd, s[a_te])
            W[m].append(prep.back(w)[0])
        log(f"  fold {f:02d}/{OUTER} done (logistic C = {c_sel:g})")
    for m in METHODS:
        assert np.isfinite(raw[m]).all() and np.isfinite(I[m]).all()

    # ---------------- summaries ----------------
    log("Summarising")
    rows = []
    for m in METHODS:
        e_raw = effect(raw[m], y, BOOTSTRAP_N, SEED)
        e_I = effect(I[m], y, BOOTSTRAP_N, SEED + 10)
        e_Z = effect(Zs[m], y, BOOTSTRAP_N, SEED + 20)
        fa = [auc_rank(raw[m][(fold == f) & asd], raw[m][(fold == f) & td])
              for f in range(1, OUTER + 1)]
        if m == REFERENCE_METHOD:
            dd = (0.0, np.nan, np.nan)
        else:
            dd = paired_auc_diff(raw[m], raw[REFERENCE_METHOD], y, BOOTSTRAP_N, SEED + 30)
        row = {"METHOD": m,
               "RAW_D": e_raw["D"], "RAW_D_LO": e_raw["D_LO"], "RAW_D_HI": e_raw["D_HI"],
               "RAW_AUC": e_raw["AUC"], "RAW_AUC_LO": e_raw["AUC_LO"], "RAW_AUC_HI": e_raw["AUC_HI"],
               "I_D": e_I["D"], "I_D_LO": e_I["D_LO"], "I_D_HI": e_I["D_HI"],
               "I_AUC": e_I["AUC"], "I_AUC_LO": e_I["AUC_LO"], "I_AUC_HI": e_I["AUC_HI"],
               "Z_D": e_Z["D"], "Z_D_LO": e_Z["D_LO"], "Z_D_HI": e_Z["D_HI"],
               "PER_FOLD_AUC_MEAN": float(np.mean(fa)), "PER_FOLD_AUC_SD": float(np.std(fa, ddof=1)),
               "AUC_DIFF_VS_" + REFERENCE_METHOD.upper(): dd[0],
               "AUC_DIFF_CI_LO": dd[1], "AUC_DIFF_CI_HI": dd[2],
               "MEAN_I_ASD": float(I[m][asd].mean()), "MEAN_I_TD": float(I[m][td].mean()),
               "MEAN_R_ASD": float(R[m][asd].mean()), "SD_R_ASD": float(R[m][asd].std(ddof=1)),
               "STABILITY_EDGE_PATTERN_MEAN_R": pair_corr_mean(np.array(W[m]))}
        row.update(site_auc_meta(raw[m], y, sites))
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUTPUT_DIR, "baseline_comparison_v3.csv"), index=False)

    corr = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
    for a in METHODS:
        for b in METHODS:
            corr.loc[a, b] = spearmanr(raw[a], raw[b]).statistic
    corr.to_csv(os.path.join(OUTPUT_DIR, "baseline_score_rank_agreement.csv"))

    out = pd.DataFrame({"SUB_ID": df["SUB_ID"].values, "SITE_ID": sites,
                        "GROUP": np.where(asd, "ASD", "TD"), "FOLD": fold})
    for m in METHODS:
        out[f"RAW_{m}"] = raw[m]; out[f"I_{m}"] = I[m]
        out[f"Z_{m}"] = Zs[m]; out[f"R_{m}"] = np.where(asd, R[m], np.nan)
    out.to_csv(os.path.join(OUTPUT_DIR, "baseline_oof_scores_v3.csv"), index=False)
    pd.DataFrame({"FOLD": range(1, OUTER + 1), "LOGISTIC_C": c_sel_list}).to_csv(
        os.path.join(OUTPUT_DIR, "baseline_selected_C.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n" + "=" * 90)
    print("LINEAR DIRECTION COMPARISON (same folds, same preprocessing, all participants scored)")
    print("=" * 90)
    show = ["METHOD", "RAW_D", "RAW_D_LO", "RAW_D_HI", "RAW_AUC", "RAW_AUC_LO", "RAW_AUC_HI",
            "AUC_DIFF_VS_" + REFERENCE_METHOD.upper(), "AUC_DIFF_CI_LO", "AUC_DIFF_CI_HI"]
    print(summ[show].round(3).to_string(index=False))
    print("\nCalibration and stability:")
    print(summ[["METHOD", "MEAN_I_TD", "MEAN_I_ASD", "MEAN_R_ASD", "SD_R_ASD",
                "PER_FOLD_AUC_MEAN", "PER_FOLD_AUC_SD", "STABILITY_EDGE_PATTERN_MEAN_R",
                "SITE_RE_AUC", "SITE_RE_AUC_LO", "SITE_RE_AUC_HI"]].round(3).to_string(index=False))
    print("\nSpearman agreement of OOF scores between methods:")
    print(corr.round(3).to_string())
    print(f"\nSelected logistic C per fold: {sorted(set(c_sel_list))}")
    print("Research descriptor only - not diagnostic, not a severity measure. "
          "AUC CIs are conditional on the OOF scores (no pipeline-refit uncertainty).")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    log("Done")


if __name__ == "__main__":
    main()