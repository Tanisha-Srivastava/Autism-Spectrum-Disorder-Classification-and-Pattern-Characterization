# ============================================================
# ABIDE-I  MRI ASD Pattern Index (I) and Relative Expression Index (R)
# FINAL SCRIPT v3 -- control-anchored PCA space + supervised linear direction
#
# v3 adds (reviewer points): TD-fitted age/sex (+ optional motion) regression
# of every edge inside each training fold, optional per-site scale harmonisation,
# tangent-space comparator, CIs for Cohen's d / AUC, within-site AUC meta-analysis,
# normative-deviation score, ASD heterogeneity / subgroup analysis, method table.
# THE INDEX IS A RESEARCH DESCRIPTOR, NOT A DIAGNOSTIC OR SEVERITY MEASURE.
# ============================================================
#
# One script for the whole paper. Direction estimator in the PCA space:
#
#   DIRECTION=logistic     (default) L2 logistic regression, class-balanced,
#                          C tuned by inner shuffled stratified 5-fold CV
#   DIRECTION=centroid     site-balanced ASD-TD centroid contrast (Stage II)
#   DIRECTION=lda          shrinkage LDA
#   DIRECTION=logistic_sw  logistic regression with site-balanced sample
#                          weights (each site, each class, equal total weight)
#
# Run the SAME script with DIRECTION=centroid to regenerate the Stage II
# centroid numbers (expected: mean I ASD 60.5 / TD 50.1, d = 0.38). If it
# does not reproduce them, something differs from the Stage II run; do not
# mix numbers from different runs in one paper.
#
# Differences from the ASD-only scripts:
#   * ALL held-out participants are scored (TD too), so the ASD-vs-TD
#     analyses of the Stage II write-up are available for the chosen
#     direction (group test, d, AUC, site meta-analysis, confound models,
#     effect-size permutation test, leave-one-site-out contrast).
#   * R is still defined for ASD participants only (never for TD).
#   * Extra continuous output Z_VS_TD_REFERENCE (no ceiling at 100).
#   * Selected logistic C per fold is logged (fold_information.csv).
#   * Kruskal-Wallis effect size is now labelled correctly (eta^2_H).
#   * Legacy ASD-only file names/layouts are kept (TD rows blank) so
#     downstream ASD-only scripts still read them; all-participant tables
#     are in *_ALL.csv and the group-level tables.
#
# Everything is fitted inside the training folds only. Same folds (seed 42),
# same TD-fitted imputation / ridge site correction / robust scaling / PCA,
# same inner 5-fold cross-fit for out-of-sample reference distributions.
#
# Outputs go to content/ABIDE_FINAL_OUTPUT_<DIRECTION> (set ABIDE_OUT_DIR
# to override).
#
# PowerShell:
#   $env:DIRECTION = "centroid"; python abide_pattern_index_FINAL.py
# Quick test (minutes):
#   $env:N_PERMUTATIONS = "2"; $env:N_REPEAT_SEEDS = "2"; $env:BOOTSTRAP_N = "200"
# ============================================================

import os
os.environ["PYTHONWARNINGS"] = "ignore"
import glob
import random
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binom, chi2, kruskal, mannwhitneyu, norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DIRECTION = os.environ.get("DIRECTION", "logistic").lower()
if DIRECTION not in ("logistic", "logistic_sw", "lda", "centroid"):
    raise ValueError(f"Unknown DIRECTION: {DIRECTION}")

DATA_PATH = os.environ.get(
    "ABIDE_FC_PATH", "content/ABIDE_AAL_FC_with_phenotype1.csv")
PHENOTYPE_PATH = os.environ.get(
    "ABIDE_PHENO_PATH", "content/Phenotypic_V1_0b.csv")
OUTPUT_DIR = os.environ.get(
    "ABIDE_OUT_DIR", f"content/ABIDE_FINAL_OUTPUT_{DIRECTION.upper()}_v4")

RANDOM_STATE = 42
N_SPLITS = 10
INNER_SPLITS = 5
PCA_MAX_COMPONENTS = 150
RIDGE_ALPHA = 10.0
# First full run selected the old minimum (1e-4) in all 10 folds, so the grid
# is extended downward. Check SELECTED_C_COUNTS: if the smallest value is
# still selected, report that heavy regularisation is preferred.
LR_C_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", 2000))
N_PERMUTATIONS = int(os.environ.get("N_PERMUTATIONS", 100))
N_REPEAT_SEEDS = int(os.environ.get("N_REPEAT_SEEDS", 10))
N_JOBS = int(os.environ.get("N_JOBS", -1))

SENSITIVITY_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
MISSING_CODES = [-9999, -9999.0, "-9999", "-9999.0"]
RUN_LOSO = os.environ.get("RUN_LOSO", "1") == "1"   # set RUN_LOSO=0 to skip
RUN_TANGENT = os.environ.get("RUN_TANGENT", "1") == "1"
RUN_SUBTYPES = os.environ.get("RUN_SUBTYPES", "1") == "1"
# Nuisance regression of every edge (fitted on TD TRAINING subjects only).
# COVARIATE_ADJUST=0 reproduces the v2 behaviour (site only).
COVARIATE_ADJUST = os.environ.get("COVARIATE_ADJUST", "1") == "1"
# Optional motion column in the phenotype file (none exists in V1_0b).
MOTION_COLUMNS = ["func_mean_fd", "FUNC_MEAN_FD", "MEAN_FD", "mean_fd"]
COV_PENALTY_SCALE = 10.0   # covariate columns are scaled up => ~100x less ridge shrinkage than site dummies
NORM_K = int(os.environ.get("NORM_K", 20))          # leading PCs used for normative deviation
TANGENT_SHRINK = 0.10      # Ledoit-style shrinkage toward identity (scans can be shorter than 116 volumes)
N_CLUSTER_NULL = int(os.environ.get("N_CLUSTER_NULL", 200))
N_CLUSTER_BOOT = int(os.environ.get("N_CLUSTER_BOOT", 100))

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_subject_id(series):
    s = series.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def empirical_percentile(reference, values):
    """100 * |{reference <= value}| / |reference|."""
    ref = np.asarray(reference, float)
    ref = np.sort(ref[np.isfinite(ref)])
    v = np.asarray(values, float)
    if len(ref) == 0:
        return np.full(len(v), np.nan)
    out = np.searchsorted(ref, v, side="right") / len(ref) * 100.0
    out[~np.isfinite(v)] = np.nan
    return out


def bootstrap_mean_ci(x, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    means = rng.choice(x, (n_boot, len(x))).mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def bootstrap_diff_ci(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def quartile_group(v):
    v = np.asarray(v, float)
    out = np.full(len(v), -1, int)
    ok = np.isfinite(v)
    out[ok] = np.digitize(v[ok], [25, 50, 75])
    return out


GROUP_NAMES = ["Low", "Moderate", "High", "Very high"]


def group_label(v):
    if not np.isfinite(v):
        return np.nan
    return GROUP_NAMES[int(np.digitize(v, [25, 50, 75]))]


def pooled_asd_percentile(oof, asd):
    out = np.full(len(oof), np.nan)
    out[asd] = empirical_percentile(oof[asd], oof[asd])
    return out


def pair_corr_summary(patterns):
    c = np.corrcoef(patterns)
    return c[np.triu_indices(len(patterns), k=1)]


def icc_2_1(M):
    """ICC(2,1), two-way random effects, absolute agreement. M: subjects x repeats."""
    M = np.asarray(M, float)
    n, k = M.shape
    gm = M.mean()
    msr = k * np.sum((M.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((M.mean(0) - gm) ** 2) / (k - 1)
    sse = np.sum((M - M.mean(1, keepdims=True)
                  - M.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def group_stats(score, y):
    """ASD (y==1) vs TD (y==0) on a score defined for everybody."""
    a = score[(y == 1) & np.isfinite(score)]
    t = score[(y == 0) & np.isfinite(score)]
    u = mannwhitneyu(a, t, alternative="two-sided")
    return {
        "N_ASD": len(a), "N_TD": len(t),
        "MEAN_ASD": a.mean(), "MEAN_TD": t.mean(),
        "MEDIAN_ASD": np.median(a), "MEDIAN_TD": np.median(t),
        "SD_ASD": a.std(ddof=1), "SD_TD": t.std(ddof=1),
        "MEAN_DIFF": a.mean() - t.mean(),
        "COHENS_D": cohen_d(a, t),
        "MWU_U": float(u.statistic), "MWU_P": float(u.pvalue),
        "AUC_DESCRIPTIVE": float(u.statistic / (len(a) * len(t))),
    }


def dl_meta(d, se):
    """DerSimonian-Laird random-effects meta-analysis."""
    d = np.asarray(d, float); se = np.asarray(se, float)
    k = len(d)
    w = 1.0 / se ** 2
    fixed = np.sum(w * d) / np.sum(w)
    Q = float(np.sum(w * (d - fixed) ** 2))
    df = k - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    ws = 1.0 / (se ** 2 + tau2)
    pooled = np.sum(ws * d) / np.sum(ws)
    se_p = np.sqrt(1.0 / np.sum(ws))
    return {"K_SITES": k, "POOLED_DIFF": pooled,
            "CI_LOW": pooled - 1.96 * se_p, "CI_HIGH": pooled + 1.96 * se_p,
            "P": float(2 * norm.sf(abs(pooled / se_p))),
            "Q": Q, "Q_DF": df, "Q_P": float(chi2.sf(Q, df)) if df > 0 else np.nan,
            "I2_PERCENT": float(max(0.0, (Q - df) / Q) * 100) if Q > 0 else 0.0,
            "TAU2": tau2}


def sign_test(k_pos, n):
    p_one = float(binom.sf(k_pos - 1, n, 0.5))
    p_low = float(binom.cdf(k_pos, n, 0.5))
    return p_one, min(1.0, 2 * min(p_one, p_low))


def site_effects(score, y, sites):
    """Within-site ASD-TD difference of a score (sites with both groups, n>=2)."""
    rows = []
    for s in sorted(np.unique(sites)):
        a = score[(sites == s) & (y == 1)]; t = score[(sites == s) & (y == 0)]
        a = a[np.isfinite(a)]; t = t[np.isfinite(t)]
        if len(a) < 2 or len(t) < 2:
            continue
        se = np.sqrt(a.var(ddof=1) / len(a) + t.var(ddof=1) / len(t))
        rows.append({"SITE": s, "N_ASD": len(a), "N_TD": len(t),
                     "MEAN_ASD": a.mean(), "MEAN_TD": t.mean(),
                     "DIFF": a.mean() - t.mean(), "SE": se})
    return pd.DataFrame(rows)


def site_summary(se_df):
    """Sign test + random-effects meta-analysis from a site_effects table."""
    out = {}
    if len(se_df) < 2:
        return out
    k_pos = int((se_df["DIFF"] > 0).sum())
    p1, p2 = sign_test(k_pos, len(se_df))
    out.update({"SITES_POSITIVE": k_pos, "SITES_TOTAL": len(se_df),
                "SIGN_P_ONE_SIDED": p1, "SIGN_P_TWO_SIDED": p2})
    out.update({f"RE_{k}": v for k, v in
                dl_meta(se_df["DIFF"].values, se_df["SE"].values).items()})
    return out


def pooled_ref_index(oof, y):
    """Index of every participant against the pooled OOF TD scores."""
    return empirical_percentile(oof[y == 0], oof)


def ols_hc3(y, X):
    """OLS with HC3 standard errors (normal approximation). Column 1 of X is
    the term of interest (diagnosis). Returns effect, CI, p, adj R2, n."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, p = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    res = y - X @ beta
    h = np.clip(np.einsum("ij,jk,ik->i", X, xtx_inv, X), 0, 1 - 1e-8)
    Xs = X * (res / (1 - h))[:, None]
    cov = xtx_inv @ (Xs.T @ Xs) @ xtx_inv
    se = float(np.sqrt(cov[1, 1]))
    r2 = 1 - (res @ res) / np.sum((y - y.mean()) ** 2)
    adj = 1 - (1 - r2) * (n - 1) / (n - np.linalg.matrix_rank(X))
    return {"N": n, "ASD_EFFECT": float(beta[1]),
            "CI_LOW": float(beta[1] - 1.96 * se),
            "CI_HIGH": float(beta[1] + 1.96 * se),
            "P": float(2 * norm.sf(abs(beta[1] / se))), "ADJ_R2": float(adj)}


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[idx] * n / rank)
        q[idx] = prev
    return q



def boot_ci_stat(a, b, fn, n_boot=2000, seed=42):
    """Percentile CI of fn(a, b), resampling each group with replacement."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    v = np.empty(n_boot)
    for i in range(n_boot):
        v[i] = fn(rng.choice(a, len(a)), rng.choice(b, len(b)))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def auc_rank(a, b):
    """P(a > b) + 0.5 P(a == b) via ranks."""
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def auc_se_hanley_mcneil(auc, n1, n0):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2)
         + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return float(np.sqrt(max(v, 1e-12)))


def effect_table(score, y, n_boot, seed=42, label=""):
    """Cohen's d and AUC (ASD > TD) with bootstrap 95% CIs."""
    a = score[(y == 1) & np.isfinite(score)]; t = score[(y == 0) & np.isfinite(score)]
    d = cohen_d(a, t); auc = auc_rank(a, t)
    d_lo, d_hi = boot_ci_stat(a, t, cohen_d, n_boot, seed)
    a_lo, a_hi = boot_ci_stat(a, t, auc_rank, n_boot, seed + 1)
    return {"SCORE": label, "COHENS_D": d, "D_CI_LOW": d_lo, "D_CI_HIGH": d_hi,
            "AUC": auc, "AUC_CI_LOW": a_lo, "AUC_CI_HIGH": a_hi,
            "N_ASD": len(a), "N_TD": len(t)}


def site_auc_meta(score, y, sites):
    """Within-site AUC (Hanley-McNeil SE) + random-effects pooling."""
    rows = []
    for s_ in sorted(np.unique(sites)):
        a = score[(sites == s_) & (y == 1)]; t = score[(sites == s_) & (y == 0)]
        if len(a) < 3 or len(t) < 3:
            continue
        auc = auc_rank(a, t)
        rows.append({"SITE": s_, "N_ASD": len(a), "N_TD": len(t), "AUC": auc,
                     "SE": auc_se_hanley_mcneil(auc, len(a), len(t))})
    t_ = pd.DataFrame(rows)
    if len(t_) < 2:
        return t_, {}
    m = dl_meta(t_["AUC"].values, t_["SE"].values)
    return t_, {"RE_AUC": m["POOLED_DIFF"], "RE_AUC_CI_LOW": m["CI_LOW"],
                "RE_AUC_CI_HIGH": m["CI_HIGH"], "RE_AUC_I2": m["I2_PERCENT"],
                "K_SITES": m["K_SITES"]}


def per_fold_auc(oof, y, fold):
    v = [auc_rank(oof[(fold == f) & (y == 1)], oof[(fold == f) & (y == 0)])
         for f in np.unique(fold)
         if ((fold == f) & (y == 1)).any() and ((fold == f) & (y == 0)).any()]
    return float(np.mean(v)), float(np.std(v, ddof=1))


# ---- tangent-space features (strictly training-referenced) ----
def _sym_fun(M, fn):
    w, V = np.linalg.eigh(M)
    return (V * fn(w)) @ V.T


def corr_from_z(X, n_roi=116):
    """Fisher-z edge vector -> shrunk correlation matrices (SPD)."""
    iu = np.triu_indices(n_roi, 1)
    r = np.tanh(X)
    out = np.empty((len(X), n_roi, n_roi))
    for i in range(len(X)):
        C = np.eye(n_roi)
        C[iu] = r[i]; C[(iu[1], iu[0])] = r[i]
        out[i] = (1 - TANGENT_SHRINK) * C + TANGENT_SHRINK * np.eye(n_roi)
    return out


def tangent_reference(C_td):
    """Log-Euclidean mean of the TRAINING TD matrices (the tangent point)."""
    L = np.mean([_sym_fun(c, np.log) for c in C_td], axis=0)
    M = _sym_fun(L, np.exp)
    return _sym_fun(M, lambda w: 1.0 / np.sqrt(w))      # M^{-1/2}


def tangent_features(C_all, Minv_half):
    n_roi = C_all.shape[1]
    iu = np.triu_indices(n_roi)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    F = np.empty((len(C_all), len(iu[0])))
    for i, c in enumerate(C_all):
        F[i] = _sym_fun(Minv_half @ c @ Minv_half, np.log)[iu] * scale
    return F

# ============================================================
# CORE MODEL
# ============================================================

def site_balanced_weights(y, sites):
    """Every site-by-class cell gets the same total weight (sites lacking a
    class get weight 0), normalised to sum to n."""
    sw = np.zeros(len(y))
    for s in np.unique(sites):
        m = sites == s
        n1 = int((m & (y == 1)).sum()); n0 = int((m & (y == 0)).sum())
        if n1 == 0 or n0 == 0:
            continue
        sw[m & (y == 1)] = 1.0 / n1
        sw[m & (y == 0)] = 1.0 / n0
    if sw.sum() == 0:
        raise ValueError("No site contains both ASD and TD subjects.")
    return sw * len(y) / sw.sum()


def learn_pattern(U, y, sites, weighting="equal", min_group_n=1):
    """Site-balanced ASD-TD centroid contrast in PCA space (unit norm)."""
    if weighting == "pooled":
        c_asd = U[y == 1].mean(0); c_td = U[y == 0].mean(0)
    else:
        a_list, t_list = [], []
        for s in np.unique(sites):
            m = sites == s
            ma = m & (y == 1); mt = m & (y == 0)
            if ma.sum() < max(min_group_n, 1) or mt.sum() < max(min_group_n, 1):
                continue
            a_list.append(U[ma].mean(0)); t_list.append(U[mt].mean(0))
        if not a_list:
            raise ValueError("No site contains both ASD and TD subjects.")
        c_asd = np.mean(a_list, axis=0); c_td = np.mean(t_list, axis=0)
    w = c_asd - c_td
    n = np.linalg.norm(w)
    if n == 0:
        raise ValueError("Pattern has zero norm.")
    return w / n


def learn_direction(U, y, sites, direction, weighting="equal",
                    min_group_n=1, seed=RANDOM_STATE):
    """Returns (unit-norm direction, selected logistic C or NaN)."""
    c_sel = np.nan
    if direction == "centroid":
        return learn_pattern(U, y, sites, weighting, min_group_n), c_sel
    if direction == "lda":
        w = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto").fit(U, y).coef_.ravel()
    elif direction in ("logistic", "logistic_sw"):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        if direction == "logistic":
            lr = LogisticRegressionCV(
                Cs=LR_C_GRID, cv=cv, scoring="roc_auc",
                class_weight="balanced", max_iter=3000, n_jobs=1).fit(U, y)
        else:
            lr = LogisticRegressionCV(
                Cs=LR_C_GRID, cv=cv, scoring="roc_auc",
                class_weight=None, max_iter=3000, n_jobs=1).fit(
                U, y, sample_weight=site_balanced_weights(y, sites))
        w = lr.coef_.ravel()
        c_sel = float(np.ravel(lr.C_)[0])
    else:
        raise ValueError(f"Unknown DIRECTION: {direction}")
    n = np.linalg.norm(w)
    if n == 0:
        raise ValueError("Pattern has zero norm.")
    return w / n, c_sel


class ControlAnchoredModel:
    """Imputer -> ridge site correction -> robust scaler -> PCA, all fitted on
    TD training subjects only; direction w learned on all training subjects
    and sign-oriented so ASD training scores exceed TD training scores."""

    def __init__(self, alpha=RIDGE_ALPHA, site_correct=True,
                 weighting="equal", min_group_n=1, seed=RANDOM_STATE,
                 direction=None, covar_adjust=None, site_scale=False):
        self.direction = direction or DIRECTION
        self.covar_adjust = COVARIATE_ADJUST if covar_adjust is None else covar_adjust
        self.site_scale = site_scale
        self.alpha = alpha
        self.site_correct = site_correct
        self.weighting = weighting
        self.min_group_n = min_group_n
        self.seed = seed

    def _site_matrix(self, sites):
        return (np.asarray(sites)[:, None]
                == self.site_levels[None, :]).astype(float)

    def _design(self, sites, C):
        """Site dummies (+ TD-training-standardised covariates)."""
        D = self._site_matrix(sites)
        if self.covar_adjust and C is not None:
            Cc = np.where(np.isfinite(C), C, self.cov_med)
            D = np.hstack([D, COV_PENALTY_SCALE * (Cc - self.cov_mu) / self.cov_sd])
        return D

    def _adjust(self, X, sites, C):
        if self.imputer is not None:
            X = self.imputer.transform(X)
        if self.site_correct:
            X = X - self.ridge.predict(self._design(sites, C))
            if self.site_scale:
                f = np.array([self.site_factor.get(s_, 1.0) for s_ in sites])
                X = X / f[:, None]
        return X

    def scaled(self, X, sites, C=None):
        return self.scaler.transform(self._adjust(X, sites, C))

    def deviation(self, X, sites, C=None, k=None):
        """Normative deviation: RMS of TD-standardised leading PC scores."""
        k = min(k or NORM_K, self.n_components)
        U = self.pca.transform(self.scaled(X, sites, C))[:, :k]
        z = U / np.sqrt(self.pca.explained_variance_[:k])
        return np.sqrt((z ** 2).mean(1))

    def fit(self, X, y, sites, C=None):
        td = y == 0
        if self.covar_adjust and C is not None:
            self.cov_med = np.nanmedian(C[td], axis=0)
            Ct = np.where(np.isfinite(C), C, self.cov_med)
            self.cov_mu = Ct[td].mean(0)
            self.cov_sd = Ct[td].std(0, ddof=1)
            self.cov_sd[self.cov_sd == 0] = 1.0
        self.imputer = None
        if HAS_MISSING:
            self.imputer = SimpleImputer(
                strategy="median", keep_empty_features=True).fit(X[td])
            X = self.imputer.transform(X)
        self.site_levels = np.unique(sites)
        if self.site_correct:
            D = self._design(sites, C)
            self.ridge = Ridge(alpha=self.alpha).fit(D[td], X[td])
            X = X - self.ridge.predict(D)
            if self.site_scale:
                # one scalar per site: TD residual SD / pooled TD residual SD
                # (absorbs scan-length dependent Fisher-z noise; ComBat-like scale step)
                pooled = np.sqrt(np.mean(X[td].var(0, ddof=1)))
                self.site_factor = {}
                for s_ in self.site_levels:
                    m_ = td & (sites == s_)
                    self.site_factor[s_] = (
                        np.sqrt(np.mean(X[m_].var(0, ddof=1))) / pooled
                        if m_.sum() >= 5 else 1.0)
                f = np.array([self.site_factor[s_] for s_ in sites])
                X = X / f[:, None]
        self.scaler = RobustScaler().fit(X[td])
        Z = self.scaler.transform(X)
        k = min(PCA_MAX_COMPONENTS, int(td.sum()) - 1, Z.shape[1])
        self.pca = PCA(n_components=k, svd_solver="randomized",
                       random_state=self.seed).fit(Z[td])
        U = self.pca.transform(Z)
        w, c_sel = learn_direction(U, y, sites, self.direction,
                                   self.weighting, self.min_group_n, self.seed)
        s_train = U @ w
        if s_train[y == 1].mean() < s_train[y == 0].mean():
            w = -w
            s_train = -s_train
        self.w = w
        self.train_scores = s_train
        self.c_selected = c_sel
        self.explained = float(self.pca.explained_variance_ratio_.sum())
        self.n_components = k
        return self

    def score(self, X, sites, C=None):
        return self.pca.transform(self.scaled(X, sites, C)) @ self.w

    def edge_weights(self):
        """Back-projection to edge space: (raw-unit, scaled-unit)."""
        scaled = self.pca.components_.T @ self.w
        sc = np.asarray(self.scaler.scale_, float).copy()
        sc[sc == 0] = 1.0
        return scaled / sc, scaled


# ============================================================
# OOF ENGINE
# ============================================================

def inner_reference_scores(X, y, sites, seed, params, C=None, with_dev=False):
    """Out-of-sample scores (and normative deviation) for every training
    subject via inner K-fold."""
    skf = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
    s = np.full(len(y), np.nan); dv = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        Ctr = None if C is None else C[tr]; Cte = None if C is None else C[te]
        m = ControlAnchoredModel(seed=seed, **params).fit(
            X[tr], y[tr], sites[tr], Ctr)
        s[te] = m.score(X[te], sites[te], Cte)
        if with_dev:
            dv[te] = m.deviation(X[te], sites[te], Cte)
    return (s, dv) if with_dev else s


def run_oof(X, y, sites, seed=RANDOM_STATE, params=None,
            inner_reference=False, collect_patterns=True, C=None):
    """10-fold OOF. Every held-out participant gets a raw score; I and Z are
    defined for all (TD reference); R for ASD only (ASD reference)."""
    params = params or {}
    n = len(y)
    oof = np.full(n, np.nan)
    fold_id = np.zeros(n, int)
    I_in = np.full(n, np.nan); R_in = np.full(n, np.nan)
    Z_in = np.full(n, np.nan)
    I_ins = np.full(n, np.nan); R_ins = np.full(n, np.nan)
    dev_oof = np.full(n, np.nan); dev_I = np.full(n, np.nan)
    om_raw, om_scaled, info = [], [], []

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(skf.split(X, y), start=1):
        ytr = y[tr]
        Ctr = None if C is None else C[tr]; Cte = None if C is None else C[te]
        m = ControlAnchoredModel(seed=seed, **params).fit(
            X[tr], ytr, sites[tr], Ctr)
        fold_id[te] = f
        s_te = m.score(X[te], sites[te], Cte)
        dev_oof[te] = m.deviation(X[te], sites[te], Cte)
        oof[te] = s_te
        te_asd = y[te] == 1

        I_ins[te] = empirical_percentile(m.train_scores[ytr == 0], s_te)
        R_ins[te[te_asd]] = empirical_percentile(
            m.train_scores[ytr == 1], s_te[te_asd])

        if inner_reference:
            s_ref, d_ref = inner_reference_scores(
                X[tr], ytr, sites[tr], seed + 100 * f, params, Ctr, True)
            dev_I[te] = empirical_percentile(d_ref[ytr == 0], dev_oof[te])
            ref_td, ref_asd = s_ref[ytr == 0], s_ref[ytr == 1]
            I_in[te] = empirical_percentile(ref_td, s_te)
            Z_in[te] = (s_te - ref_td.mean()) / ref_td.std(ddof=1)
            R_in[te[te_asd]] = empirical_percentile(ref_asd, s_te[te_asd])

        if collect_patterns:
            raw, scl = m.edge_weights()
            om_raw.append(raw); om_scaled.append(scl)

        info.append({
            "FOLD": f, "TRAIN_N": len(tr), "TEST_N": len(te),
            "TRAIN_ASD": int((ytr == 1).sum()), "TRAIN_TD": int((ytr == 0).sum()),
            "TEST_ASD": int(te_asd.sum()), "TEST_TD": int((~te_asd).sum()),
            "PCA_COMPONENTS": m.n_components, "PCA_VARIANCE_TD": m.explained,
            "SELECTED_LOGISTIC_C": m.c_selected,
        })

    return {"oof": oof, "fold": fold_id, "I_inner": I_in, "R_inner": R_in,
            "Z_inner": Z_in, "dev": dev_oof, "dev_I": dev_I, "I_insample": I_ins, "R_insample": R_ins,
            "omega_raw": np.array(om_raw), "omega_scaled": np.array(om_scaled),
            "fold_info": info}


# ============================================================
# PARALLEL WORKERS
# ============================================================

def permute_within_site(y, sites, rng):
    yp = y.copy()
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        yp[idx] = rng.permutation(y[idx])
    return yp


def permutation_job(X, y, sites, seed, C=None):
    """Whole pipeline rerun on within-site label-shuffled data; statistics are
    computed against the SHUFFLED labels."""
    rng = np.random.default_rng(seed)
    yp = permute_within_site(y, sites, rng)
    r = run_oof(X, yp, sites, seed=RANDOM_STATE, params={},
                inner_reference=False, collect_patterns=True, C=C)
    oof = r["oof"]
    d_raw = cohen_d(oof[yp == 1], oof[yp == 0])
    ip = pooled_ref_index(oof, yp)
    diff = ip[yp == 1].mean() - ip[yp == 0].mean()
    return (d_raw, diff, pair_corr_summary(r["omega_raw"]).mean(),
            pair_corr_summary(r["omega_scaled"]).mean())


def sensitivity_job(X, y, sites, label, params, C=None):
    r = run_oof(X, y, sites, seed=RANDOM_STATE, params=params,
                inner_reference=False, collect_patterns=False, C=C)
    return label, r["oof"]


def repeat_job(X, y, sites, seed, C=None):
    return run_oof(X, y, sites, seed=seed, params={},
                   inner_reference=False, collect_patterns=False, C=C)["oof"]


def loso_job(X, y, sites, held_site, C=None):
    """Hold out one site entirely; score ALL its participants (the site gets
    the intercept shift only, i.e. the 'new scanner' case)."""
    te = np.where(sites == held_site)[0]
    tr = np.where(sites != held_site)[0]
    Ctr = None if C is None else C[tr]; Cte = None if C is None else C[te]
    m = ControlAnchoredModel(seed=RANDOM_STATE).fit(X[tr], y[tr], sites[tr], Ctr)
    s = m.score(X[te], sites[te], Cte)
    s_ref = inner_reference_scores(
        X[tr], y[tr], sites[tr], RANDOM_STATE + 777, {}, Ctr)
    ytr = y[tr]
    I_te = empirical_percentile(s_ref[ytr == 0], s)
    R_te = np.full(len(te), np.nan)
    a = y[te] == 1
    R_te[a] = empirical_percentile(s_ref[ytr == 1], s[a])
    return te, s, I_te, R_te


# ============================================================
# A. LOAD DATA
# ============================================================

log(f"A. Loading data (direction = {DIRECTION})")
df = pd.read_csv(DATA_PATH)
for col in ["SUB_ID", "DX_GROUP"]:
    if col not in df.columns:
        raise ValueError(f"{col} is required in the MRI file.")
df["SUB_ID"] = normalize_subject_id(df["SUB_ID"])

fc_columns = [c for c in df.columns if "__" in str(c)]
if len(fc_columns) != 6670:
    print(f"WARNING: expected 6670 FC edges, found {len(fc_columns)}")

y = np.where(df["DX_GROUP"].values == 1, 1, 0)
if "SITE_ID" in df.columns:
    sites = df["SITE_ID"].fillna("UNKNOWN").astype(str).values
else:
    sites = np.array(["UNKNOWN"] * len(df))

X = df[fc_columns].apply(pd.to_numeric, errors="coerce").values.astype(float)
X[~np.isfinite(X)] = np.nan
HAS_MISSING = bool(np.isnan(X).any())
asd = y == 1
td = y == 0
log(f"N={len(y)} (ASD {asd.sum()}, TD {td.sum()}), "
    f"{len(np.unique(sites))} sites, {X.shape[1]} edges, "
    f"missing values: {int(np.isnan(X).sum())}")

covars = pd.DataFrame({
    "N_TIMEPOINTS": (pd.to_numeric(df["N_TIMEPOINTS"], errors="coerce").values
                     if "N_TIMEPOINTS" in df.columns
                     else np.full(len(df), np.nan)),
    "GLOBAL_MEAN_FC": np.nanmean(X, axis=1),
    "AGE": np.nan, "SEX_F": np.nan, "FIQ": np.nan,
})
HAVE_COVARS = False
try:
    pheno_path = PHENOTYPE_PATH
    if not os.path.exists(pheno_path):
        cand = glob.glob("/content/*Phenotypic*.csv")
        pheno_path = cand[0] if cand else None
    pheno = pd.read_csv(pheno_path)
    pheno["SUB_ID"] = normalize_subject_id(pheno["SUB_ID"])
    pheno = pheno.drop_duplicates("SUB_ID").replace(MISSING_CODES, np.nan)
    mg = df[["SUB_ID"]].merge(
        pheno[["SUB_ID", "AGE_AT_SCAN", "SEX", "FIQ"]],
        on="SUB_ID", how="left")
    covars["AGE"] = pd.to_numeric(mg["AGE_AT_SCAN"], errors="coerce").values
    sex = pd.to_numeric(mg["SEX"], errors="coerce").values
    covars["SEX_F"] = np.where(np.isfinite(sex), (sex == 2).astype(float), np.nan)
    covars["FIQ"] = pd.to_numeric(mg["FIQ"], errors="coerce").values
    HAVE_COVARS = bool(covars["AGE"].notna().any())
    for mc in MOTION_COLUMNS:                      # optional motion regressor
        if mc in pheno.columns:
            covars["MOTION"] = pd.to_numeric(
                df[["SUB_ID"]].merge(pheno[["SUB_ID", mc]], on="SUB_ID",
                                     how="left")[mc], errors="coerce").values
            break
except Exception as e:
    print("Covariates unavailable:", e)

# nuisance matrix regressed out of every edge (TD-fitted inside each fold)
MOTION_AVAILABLE = "MOTION" in covars.columns
if COVARIATE_ADJUST and HAVE_COVARS:
    cov_names = ["AGE", "SEX_F"] + (["MOTION"] if MOTION_AVAILABLE else [])
    COV = covars[cov_names].values.astype(float)
    log(f"Nuisance regression of edges on {cov_names} (+ site); "
        f"motion available: {MOTION_AVAILABLE}")
else:
    COV = None
    if COVARIATE_ADJUST:
        print("WARNING: age/sex unavailable -> edges are NOT covariate-adjusted")
if not MOTION_AVAILABLE:
    print("NOTE: no head-motion measure in the phenotype file; motion cannot be "
          "regressed. State this as a limitation (scan length and global mean FC "
          "are not motion measures).")

# ============================================================
# B-G. MAIN OOF RUN WITH OUT-OF-SAMPLE REFERENCE
# ============================================================

log("B-G. Main 10-fold OOF run with inner cross-fitted reference")
main = run_oof(X, y, sites, seed=RANDOM_STATE, params={},
               inner_reference=True, collect_patterns=True, C=COV)
oof = main["oof"]; I = main["I_inner"]; R = main["R_inner"]; Z = main["Z_inner"]
R_pooled = pooled_asd_percentile(oof, asd)

assert np.isfinite(oof).all() and np.isfinite(I).all() and np.isfinite(Z).all(), \
    "Every participant needs one OOF score, I and Z"
assert np.isfinite(R[asd]).all() and np.isnan(R[td]).all(), \
    "R must exist for ASD participants only"
log("Integrity checks passed (I for all, R for ASD only, one score each)")

fold_df = pd.DataFrame(main["fold_info"])
fold_df.to_csv(os.path.join(OUTPUT_DIR, "fold_information.csv"), index=False)

results = pd.DataFrame({
    "SUB_ID": df["SUB_ID"].values,
    "ASD_STATUS": np.where(asd, "ASD", "TD"),
    "SITE_ID": sites, "FOLD": main["fold"],
    "OOF_PATTERN_SCORE": oof,
    "MRI_ASD_PATTERN_INDEX_I": I,
    "ASD_RELATIVE_EXPRESSION_INDEX_R": R,
    "Z_VS_TD_REFERENCE": Z,
    "NORMATIVE_DEVIATION_RMS_Z": main["dev"],
    "NORMATIVE_DEVIATION_PERCENTILE_VS_TD": main["dev_I"],
    "I_INSAMPLE_REFERENCE_ORIGINAL": main["I_insample"],
    "R_INSAMPLE_REFERENCE_ORIGINAL": main["R_insample"],
    "R_POOLED_ASD_PERCENTILE": R_pooled,
})
results["I_GROUP"] = [group_label(v) for v in I]
results["R_GROUP"] = [group_label(v) for v in R]
results.to_csv(os.path.join(OUTPUT_DIR, "subject_level_results_ALL.csv"),
               index=False)
# legacy layout: TD rows blank, so ASD-only downstream scripts keep working
legacy = results.copy()
for c in ["OOF_PATTERN_SCORE", "MRI_ASD_PATTERN_INDEX_I", "Z_VS_TD_REFERENCE",
          "I_INSAMPLE_REFERENCE_ORIGINAL", "I_GROUP"]:
    legacy.loc[td, c] = np.nan
legacy.to_csv(os.path.join(OUTPUT_DIR, "subject_level_results.csv"), index=False)
results[asd].drop(columns=["ASD_STATUS"]).to_csv(
    os.path.join(OUTPUT_DIR, "subject_level_results_ASD_only.csv"), index=False)

summary = {"PATTERN_DIRECTION_ESTIMATOR": DIRECTION,
           "N_ASD": int(asd.sum()), "N_TD": int(td.sum()),
           "N_SITES": int(len(np.unique(sites)))}
if DIRECTION.startswith("logistic"):
    summary["SELECTED_C_COUNTS"] = str(
        fold_df["SELECTED_LOGISTIC_C"].value_counts().to_dict())
    _c = fold_df["SELECTED_LOGISTIC_C"]
    if (_c <= min(LR_C_GRID)).any() or (_c >= max(LR_C_GRID)).any():
        print("WARNING: selected C hit the edge of LR_C_GRID in "
              f"{int(((_c <= min(LR_C_GRID)) | (_c >= max(LR_C_GRID))).sum())} "
              "fold(s); the grid truncates the optimum.")

# ============================================================
# H. GROUP-LEVEL SEPARATION (Table 1)
# ============================================================

log("H. Group-level separation")
gs = group_stats(I, y)
ci_lo, ci_hi = bootstrap_diff_ci(I[asd], I[td], BOOTSTRAP_N, RANDOM_STATE)
gs["MEAN_DIFF_CI_LOW"] = ci_lo; gs["MEAN_DIFF_CI_HIGH"] = ci_hi
gs_z = group_stats(Z, y)
pd.DataFrame([{"SCORE": "I (percentile vs TD reference)", **gs},
              {"SCORE": "Z (vs TD reference, no ceiling)", **gs_z}]).to_csv(
    os.path.join(OUTPUT_DIR, "group_level_results.csv"), index=False)
summary.update({f"I_{k}": v for k, v in gs.items()})
summary["Z_COHENS_D"] = gs_z["COHENS_D"]

# --- effect sizes with CIs (reviewer point 6) ---
eff = pd.DataFrame([
    effect_table(I, y, BOOTSTRAP_N, RANDOM_STATE, "I (percentile vs TD reference)"),
    effect_table(Z, y, BOOTSTRAP_N, RANDOM_STATE, "Z (vs TD reference, continuous)"),
    effect_table(oof, y, BOOTSTRAP_N, RANDOM_STATE, "Raw OOF pattern score"),
    effect_table(main["dev"], y, BOOTSTRAP_N, RANDOM_STATE,
                 "Normative deviation (RMS z, leading PCs)"),
])
fm_, fs_ = per_fold_auc(oof, y, main["fold"])
eff["PER_FOLD_AUC_MEAN"] = np.nan; eff["PER_FOLD_AUC_SD"] = np.nan
eff.loc[2, ["PER_FOLD_AUC_MEAN", "PER_FOLD_AUC_SD"]] = [fm_, fs_]
eff.to_csv(os.path.join(OUTPUT_DIR, "effect_sizes_with_CI.csv"), index=False)
auc_site_tab, auc_site_sum = site_auc_meta(oof, y, sites)
auc_site_tab.to_csv(os.path.join(OUTPUT_DIR, "site_auc_table.csv"), index=False)
pd.DataFrame([auc_site_sum]).to_csv(
    os.path.join(OUTPUT_DIR, "site_auc_meta_analysis.csv"), index=False)
summary.update({f"EFFECT_{r_['SCORE'][:1]}_{k}": v for r_ in eff.to_dict("records")
                for k, v in r_.items() if k not in ("SCORE",)} )
summary.update({f"SITE_AUC_{k}": v for k, v in auc_site_sum.items()})

a_I, a_R = I[asd], R[asd]
summary.update({
    "MEAN_R_ASD": a_R.mean(), "MEDIAN_R_ASD": np.median(a_R),
    "SD_R_ASD": a_R.std(ddof=1), "MIN_R_ASD": a_R.min(), "MAX_R_ASD": a_R.max(),
    "N_ASD_I_AT_CEILING_GE_99.9": int((a_I >= 99.9).sum()),
    "N_TD_I_AT_CEILING_GE_99.9": int((I[td] >= 99.9).sum()),
    "SHARE_ASD_I_GT_50": float((a_I > 50).mean()),
})

bias_df = pd.DataFrame([
    ["Out-of-sample reference (inner cross-fit) [primary]",
     I[td].mean(), a_R.mean(), I[asd].mean()],
    ["In-sample training reference (original)",
     main["I_insample"][td].mean(), main["R_insample"][asd].mean(),
     main["I_insample"][asd].mean()],
], columns=["REFERENCE", "MEAN_I_TD", "MEAN_R_ASD", "MEAN_I_ASD"])
bias_df.to_csv(os.path.join(OUTPUT_DIR, "reference_bias_check.csv"), index=False)

grp = pd.DataFrame({
    "GROUP": GROUP_NAMES,
    "N_ASD_BY_I": [int((results.loc[asd, "I_GROUP"] == g_).sum()) for g_ in GROUP_NAMES],
    "N_TD_BY_I": [int((results.loc[td, "I_GROUP"] == g_).sum()) for g_ in GROUP_NAMES],
    "N_ASD_BY_R": [int((results.loc[asd, "R_GROUP"] == g_).sum()) for g_ in GROUP_NAMES],
})
grp.to_csv(os.path.join(OUTPUT_DIR, "index_group_counts.csv"), index=False)
grp[["GROUP", "N_ASD_BY_I", "N_ASD_BY_R"]].to_csv(
    os.path.join(OUTPUT_DIR, "index_group_counts_ASD.csv"), index=False)

# ============================================================
# I. SITE CONSISTENCY + RANDOM-EFFECTS META-ANALYSIS (Table 2)
# ============================================================

log("I. Site consistency")
site_tab = site_effects(I, y, sites)
site_tab.to_csv(os.path.join(OUTPUT_DIR, "site_consistency_table.csv"), index=False)
ss = site_summary(site_tab)
summary.update({f"SITE10F_{k}": v for k, v in ss.items()})

site_rows = []
for s in sorted(np.unique(sites)):
    ma = (sites == s) & asd
    if ma.sum() < 1:
        continue
    site_rows.append({"SITE": s, "N_ASD": int(ma.sum()),
                      "MEAN_I": I[ma].mean(), "MEDIAN_I": np.median(I[ma]),
                      "SD_I": I[ma].std(ddof=1) if ma.sum() > 1 else np.nan,
                      "MEAN_R": R[ma].mean()})
site_df = pd.DataFrame(site_rows)
site_df.to_csv(os.path.join(OUTPUT_DIR, "site_level_table_ASD_only.csv"), index=False)
kw = [I[(sites == s) & asd] for s in site_df.loc[site_df["N_ASD"] >= 2, "SITE"]]
if len(kw) >= 2:
    H_, p_kw = kruskal(*kw)
    n_tot = sum(len(g_) for g_ in kw)
    summary.update({"KRUSKAL_I_ACROSS_SITES_ASD_H": float(H_),
                    "KRUSKAL_I_ACROSS_SITES_ASD_P": float(p_kw),
                    "KRUSKAL_ETA_SQUARED_H": float((H_ - len(kw) + 1) / (n_tot - len(kw)))})

# ============================================================
# J. CONFOUND MODELS (Table 4) AND WITHIN-ASD ASSOCIATIONS
# ============================================================

if HAVE_COVARS:
    log("J. Confound models and covariate associations")
    D_site = pd.get_dummies(pd.Series(sites), drop_first=True).values.astype(float)
    yf = y.astype(float)

    def fit_model(name, mask, use_site, cols):
        parts = [np.ones(mask.sum()), yf[mask]]
        mats = [np.column_stack(parts)]
        if use_site:
            mats.append(D_site[mask])
        for c in cols:
            mats.append(covars[c].values[mask][:, None])
        res = ols_hc3(I[mask], np.hstack(mats))
        return {"MODEL": name, **res}

    allm = np.ones(len(y), bool)
    ok = lambda cols: allm & np.all(
        [np.isfinite(covars[c].values) for c in cols], axis=0)
    rows = [fit_model("M0 unadjusted", allm, False, []),
            fit_model("M1 + site", allm, True, [])]
    m2 = ok(["AGE", "SEX_F"])
    if m2.sum() > 30:
        rows.append(fit_model("M2 + site + age + sex", m2, True, ["AGE", "SEX_F"]))
        cols3 = ["AGE", "SEX_F", "N_TIMEPOINTS", "GLOBAL_MEAN_FC"]
        m3 = ok(cols3)
        if m3.sum() > 30:
            rows.append(fit_model("M3 + site + age + sex + scan length + global mean FC",
                                  m3, True, cols3))
        m4 = ok(["AGE", "SEX_F", "FIQ"])
        if m4.sum() > 30:
            rows.append(fit_model("M4 + site + age + sex + FIQ (subset)",
                                  m4, True, ["AGE", "SEX_F", "FIQ"]))
            rows.append(fit_model("M4-ref: M1 on the same subset", m4, True, []))
    conf_df = pd.DataFrame(rows)
    conf_df.to_csv(os.path.join(OUTPUT_DIR, "confound_models.csv"), index=False)

    cov_rows = []
    for nm, v in [("AGE", covars["AGE"]), ("SEX_FEMALE", covars["SEX_F"]),
                  ("FIQ", covars["FIQ"]),
                  ("SCAN_LENGTH_N_TIMEPOINTS", covars["N_TIMEPOINTS"]),
                  ("GLOBAL_MEAN_FC", covars["GLOBAL_MEAN_FC"])]:
        v = np.asarray(v.values, float)
        for grp_name, gm in [("ASD", asd), ("TD", td), ("ALL", allm)]:
            okv = np.isfinite(v) & gm
            if okv.sum() < 10:
                continue
            ri = spearmanr(I[okv], v[okv])
            cov_rows.append({"VARIABLE": nm, "GROUP": grp_name, "N": int(okv.sum()),
                             "SPEARMAN_RHO_WITH_I": ri.statistic, "P": ri.pvalue})
    cov_df = pd.DataFrame(cov_rows)
    cov_df["Q_BH_WITHIN_GROUP"] = np.nan
    for g_ in cov_df["GROUP"].unique():
        mk = cov_df["GROUP"] == g_
        cov_df.loc[mk, "Q_BH_WITHIN_GROUP"] = benjamini_hochberg(cov_df.loc[mk, "P"].values)
    cov_df.to_csv(os.path.join(OUTPUT_DIR, "covariate_associations.csv"), index=False)
    cov_df[cov_df["GROUP"] == "ASD"].to_csv(
        os.path.join(OUTPUT_DIR, "asd_covariate_associations.csv"), index=False)

# ============================================================
# K. SITE-HANDLING / DIRECTION SENSITIVITY
# ============================================================

log("K. Site-handling and direction sensitivity")
configs = [(f"Ridge alpha = {a:g}", {"alpha": a})
           for a in SENSITIVITY_ALPHAS if a != RIDGE_ALPHA]
configs.append(("No site correction", {"site_correct": False}))
configs.append(("No age/sex adjustment (v2 behaviour)", {"covar_adjust": False}))
configs.append(("+ per-site scale harmonisation", {"site_scale": True}))
for d_ in ["centroid", "lda", "logistic", "logistic_sw"]:
    if d_ != DIRECTION:
        configs.append((f"Direction: {d_}", {"direction": d_}))
sens = Parallel(n_jobs=N_JOBS)(
    delayed(sensitivity_job)(X, y, sites, lbl, prm, COV) for lbl, prm in configs)
sens = [(f"Default ({DIRECTION}, alpha=10)", oof)] + list(sens)
default_asd = oof[asd]
default_q = quartile_group(empirical_percentile(default_asd, default_asd))
srows, grows = [], []
for lbl, sc_ in sens:
    s_a = sc_[asd]
    q_ = quartile_group(empirical_percentile(s_a, s_a))
    srows.append({"CONFIGURATION": lbl,
                  "SPEARMAN_WITH_DEFAULT_ASD_SCORES": spearmanr(s_a, default_asd).statistic,
                  "QUARTILE_GROUP_AGREEMENT_WITH_DEFAULT": float((q_ == default_q).mean())})
    ip = pooled_ref_index(sc_, y)
    st = site_effects(ip, y, sites)
    grows.append({"CONFIGURATION": lbl,
                  "MEAN_DIFF_I_POOLED_REF": ip[asd].mean() - ip[td].mean(),
                  "COHENS_D_I_POOLED_REF": cohen_d(ip[asd], ip[td]),
                  "SITES_POSITIVE": int((st["DIFF"] > 0).sum()),
                  "SITES_TOTAL": len(st),
                  "RE_POOLED_DIFF": dl_meta(st["DIFF"].values, st["SE"].values)["POOLED_DIFF"]})
sens_df = pd.DataFrame(srows)
sens_df.to_csv(os.path.join(OUTPUT_DIR, "site_sensitivity_ASD_only.csv"), index=False)
sens_grp_df = pd.DataFrame(grows)
sens_grp_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_group_effect.csv"), index=False)

# ============================================================
# L. REPEATED-PARTITION RELIABILITY
# ============================================================

log(f"L. Reliability over {N_REPEAT_SEEDS} random partitions")
rel = {}
if N_REPEAT_SEEDS >= 2:
    seeds = [RANDOM_STATE + 1 + i for i in range(N_REPEAT_SEEDS)]
    rep = Parallel(n_jobs=N_JOBS)(
        delayed(repeat_job)(X, y, sites, s, COV) for s in seeds)
    raw_all = np.column_stack(rep)
    ip_all = np.column_stack([pooled_ref_index(r, y) for r in rep])
    k_ = ip_all.shape[1]
    pair_all = [spearmanr(raw_all[:, i], raw_all[:, j]).statistic
                for i in range(k_) for j in range(i + 1, k_)]
    pair_asd = [spearmanr(raw_all[asd, i], raw_all[asd, j]).statistic
                for i in range(k_) for j in range(i + 1, k_)]
    rep_R = np.column_stack([empirical_percentile(r[asd], r[asd]) for r in rep])
    qs = [quartile_group(rep_R[:, j]) for j in range(k_)]
    qa = [float((qs[i] == qs[j]).mean()) for i in range(k_) for j in range(i + 1, k_)]
    d_each = [cohen_d(ip_all[asd, j], ip_all[td, j]) for j in range(k_)]
    rel = {
        "N_PARTITIONS": k_,
        "GROUP_EFFECT_D_MEAN": float(np.mean(d_each)),
        "GROUP_EFFECT_D_SD": float(np.std(d_each, ddof=1)),
        "ICC_2_1_ALL_PARTICIPANTS_I": float(icc_2_1(ip_all)),
        "ICC_2_1_ASD_ONLY_I": float(icc_2_1(ip_all[asd])),
        "MEAN_PAIRWISE_SPEARMAN_ALL": float(np.mean(pair_all)),
        "MIN_PAIRWISE_SPEARMAN_ALL": float(np.min(pair_all)),
        "MEAN_PAIRWISE_SPEARMAN_ASD": float(np.mean(pair_asd)),
        "MEAN_SUBJECT_SD_OF_I_POINTS": float(ip_all.std(1, ddof=1).mean()),
        "MEAN_ASD_QUARTILE_GROUP_AGREEMENT_R": float(np.mean(qa)),
    }
    pd.DataFrame([rel]).to_csv(
        os.path.join(OUTPUT_DIR, "reliability_repeated_partitions.csv"), index=False)
    summary.update({f"RELIABILITY_{k}": v for k, v in rel.items()})

# ============================================================
# M. EDGE-LEVEL INTERPRETATION AND PATTERN STABILITY
# ============================================================

log("M. Edge-level interpretation (full-data fit, interpretation only)")
full = ControlAnchoredModel(seed=RANDOM_STATE).fit(X, y, sites, COV)
om_raw, om_scaled = full.edge_weights()
Zfull = full.scaled(X, sites, COV)
s_full = full.score(X, sites, COV)
sc = s_full - s_full.mean()
haufe = ((Zfull - Zfull.mean(0)).T @ sc) / (len(sc) - 1) / sc.var(ddof=1)
edge_df = pd.DataFrame({"FC_EDGE": fc_columns,
                        "LOADING_RAW_UNITS_OMEGA": om_raw,
                        "WEIGHT_SCALED_UNITS": om_scaled,
                        "HAUFE_ACTIVATION": haufe})
for nm, col in [("RAW", "LOADING_RAW_UNITS_OMEGA"), ("SCALED", "WEIGHT_SCALED_UNITS"),
                ("HAUFE", "HAUFE_ACTIVATION")]:
    edge_df[f"RANK_BY_{nm}_ABS"] = edge_df[col].abs().rank(ascending=False).astype(int)
edge_df.sort_values("RANK_BY_RAW_ABS").to_csv(
    os.path.join(OUTPUT_DIR, "edge_loadings.csv"), index=False)
for nm, col in [("RAW_OMEGA", "LOADING_RAW_UNITS_OMEGA"),
                ("SCALED", "WEIGHT_SCALED_UNITS"), ("HAUFE", "HAUFE_ACTIVATION")]:
    top = edge_df.reindex(edge_df[col].abs().sort_values(ascending=False).index).head(25)
    summary[f"TOP25_NEGATIVE_{nm}"] = int((top[col] < 0).sum())
    summary[f"ALLEDGES_FRACTION_NEGATIVE_{nm}"] = float((edge_df[col] < 0).mean())
summary["FULL_DATA_FIT_SELECTED_C"] = full.c_selected
# how much of the full-data score is just global mean FC
summary["SPEARMAN_FULLFIT_SCORE_VS_GLOBAL_MEAN_FC"] = float(
    spearmanr(s_full, covars["GLOBAL_MEAN_FC"].values).statistic)

sp = {}
for nm, arr in [("RAW", main["omega_raw"]), ("SCALED", main["omega_scaled"])]:
    v = pair_corr_summary(arr)
    sp[nm] = v
    summary[f"STABILITY_{nm}_MEAN"] = float(v.mean())
    summary[f"STABILITY_{nm}_MEDIAN"] = float(np.median(v))
    summary[f"STABILITY_{nm}_MIN"] = float(v.min())
    summary[f"STABILITY_{nm}_MAX"] = float(v.max())
pd.DataFrame({"PAIR_CORR_RAW_OMEGA": sp["RAW"], "PAIR_CORR_SCALED": sp["SCALED"]}).to_csv(
    os.path.join(OUTPUT_DIR, "stability_pairwise.csv"), index=False)

# ============================================================
# N. PERMUTATION NULLS (whole pipeline, labels shuffled within site)
# ============================================================

obs_d_raw = cohen_d(oof[asd], oof[td])
obs_diff_pool = (pooled_ref_index(oof, y)[asd].mean() - pooled_ref_index(oof, y)[td].mean())
obs_stab_raw = float(sp["RAW"].mean()); obs_stab_scl = float(sp["SCALED"].mean())
perm_summary = None
if N_PERMUTATIONS > 0:
    log(f"N. {N_PERMUTATIONS} full-pipeline permutations")
    perm = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(permutation_job)(X, y, sites, RANDOM_STATE + 5000 + i, COV)
        for i in range(N_PERMUTATIONS)))
    pd.DataFrame(perm, columns=["NULL_D_RAW_OOF", "NULL_MEAN_DIFF_I_POOLED",
                                "NULL_STABILITY_RAW", "NULL_STABILITY_SCALED"]).to_csv(
        os.path.join(OUTPUT_DIR, "permutation_null.csv"), index=False)

    def perm_p(obs, null):
        return (1 + np.sum(null >= obs)) / (1 + len(null))

    perm_summary = pd.DataFrame([
        {"STATISTIC": "Cohen's d of raw OOF score", "OBSERVED": obs_d_raw, "COL": 0},
        {"STATISTIC": "Mean diff of I (pooled reference)", "OBSERVED": obs_diff_pool, "COL": 1},
        {"STATISTIC": "Fold-pair correlation (raw-unit omega)", "OBSERVED": obs_stab_raw, "COL": 2},
        {"STATISTIC": "Fold-pair correlation (scaled)", "OBSERVED": obs_stab_scl, "COL": 3},
    ])
    perm_summary["NULL_MEAN"] = [perm[:, c].mean() for c in perm_summary["COL"]]
    perm_summary["NULL_95TH_PCT"] = [np.percentile(perm[:, c], 95) for c in perm_summary["COL"]]
    perm_summary["PERM_P_ONE_SIDED"] = [perm_p(o, perm[:, c]) for o, c in
                                        zip(perm_summary["OBSERVED"], perm_summary["COL"])]
    perm_summary["N_PERMUTATIONS"] = N_PERMUTATIONS
    perm_summary = perm_summary.drop(columns="COL")
    perm_summary.to_csv(os.path.join(OUTPUT_DIR, "permutation_summary.csv"), index=False)

# ============================================================
# O. LEAVE-ONE-SITE-OUT (Table 3)
# ============================================================
# A held-out site gets the intercept shift only, so ABSOLUTE I levels are
# offset; interpret the within-site ASD-TD contrast, not the level.

if RUN_LOSO:
    log("O. Leave-one-site-out scoring of all participants")
    loso_sites = sorted(np.unique(sites))
    loso_res = Parallel(n_jobs=N_JOBS)(
        delayed(loso_job)(X, y, sites, s_, COV) for s_ in loso_sites)
    loso_score = np.full(len(y), np.nan)
    loso_I = np.full(len(y), np.nan); loso_R = np.full(len(y), np.nan)
    for idx, s_te, I_te, R_te in loso_res:
        loso_score[idx] = s_te; loso_I[idx] = I_te; loso_R[idx] = R_te
    assert np.isfinite(loso_I).all() and np.isnan(loso_R[td]).all()

    loso_tab = site_effects(loso_I, y, sites)
    loso_tab.to_csv(os.path.join(OUTPUT_DIR, "loso_group_table.csv"), index=False)
    loso_gs = group_stats(loso_I, y)
    loso_ss = site_summary(loso_tab)
    pd.DataFrame([{**loso_gs, **loso_ss}]).to_csv(
        os.path.join(OUTPUT_DIR, "loso_group_summary.csv"), index=False)
    summary.update({f"LOSO_{k}": v for k, v in {**loso_gs, **loso_ss}.items()})

    loso_rows = []
    for s_ in loso_sites:
        ma = (sites == s_) & asd
        rho = spearmanr(loso_I[ma], I[ma]).statistic if ma.sum() >= 3 else np.nan
        loso_rows.append({"HELD_OUT_SITE": s_, "N_ASD": int(ma.sum()),
                          "MEAN_I_LOSO": loso_I[ma].mean(), "MEAN_R_LOSO": loso_R[ma].mean(),
                          "MEAN_I_10FOLD_FOR_COMPARISON": I[ma].mean(),
                          "SPEARMAN_I_LOSO_VS_10FOLD_WITHIN_SITE": rho})
    loso_df = pd.DataFrame(loso_rows)
    loso_df.to_csv(os.path.join(OUTPUT_DIR, "loso_site_table_ASD_only.csv"), index=False)
    loso_q = quartile_group(loso_R[asd]); main_q = quartile_group(R[asd])
    loso_summary = {
        "SPEARMAN_I_LOSO_VS_10FOLD_ALL_ASD": spearmanr(loso_I[asd], I[asd]).statistic,
        "SPEARMAN_R_LOSO_VS_10FOLD_ALL_ASD": spearmanr(loso_R[asd], R[asd]).statistic,
        "MEAN_ABS_DIFF_I_POINTS": float(np.abs(loso_I[asd] - I[asd]).mean()),
        "MEAN_I_LOSO_ASD": float(loso_I[asd].mean()),
        "MEAN_R_LOSO_ASD": float(loso_R[asd].mean()),
        "R_QUARTILE_GROUP_AGREEMENT": float((loso_q == main_q).mean()),
    }
    pd.DataFrame([loso_summary]).to_csv(
        os.path.join(OUTPUT_DIR, "loso_summary_ASD_only.csv"), index=False)
    summary.update({f"LOSO_ASDONLY_{k}": v for k, v in loso_summary.items()})

    loso_all = pd.DataFrame({"SUB_ID": df["SUB_ID"].values, "SITE_ID": sites,
                             "ASD_STATUS": np.where(asd, "ASD", "TD"),
                             "LOSO_SCORE": loso_score, "LOSO_I": loso_I, "LOSO_R": loso_R})
    loso_all.to_csv(os.path.join(OUTPUT_DIR, "loso_subject_level_ALL.csv"), index=False)
    leg = loso_all.copy()
    leg.loc[td, ["LOSO_SCORE", "LOSO_I"]] = np.nan
    leg.to_csv(os.path.join(OUTPUT_DIR, "loso_subject_level.csv"), index=False)


# ============================================================
# P. METHOD COMPARISON WITH CIs + TANGENT-SPACE COMPARATOR (reviewer point 1)
# ============================================================
# All methods use the SAME outer folds, the SAME TD-fitted site/age/sex
# adjustment and report d / AUC of the raw out-of-fold score with bootstrap CIs
# and a paired bootstrap CI of the AUC difference to the default method.
# NOTE: the default direction was chosen after looking at these same data;
# report every row, not only the best one (selection optimism).

def paired_auc_diff_ci(s1, s2, y, n_boot, seed=42):
    rng = np.random.default_rng(seed)
    ia, it = np.where(y == 1)[0], np.where(y == 0)[0]
    base = auc_rank(s1[ia], s1[it]) - auc_rank(s2[ia], s2[it])
    v = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(ia, len(ia)); t = rng.choice(it, len(it))
        v[i] = auc_rank(s1[a], s1[t]) - auc_rank(s2[a], s2[t])
    return base, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


method_scores = {}
for lbl, sc_ in sens:
    if lbl.startswith("Default") or lbl.startswith("Direction:") \
            or lbl in ("No age/sex adjustment (v2 behaviour)",
                       "+ per-site scale harmonisation", "No site correction"):
        method_scores[lbl] = sc_

if RUN_TANGENT:
    log("P. Tangent-space comparator (reference point fitted on TRAINING TD only)")
    C_mats = corr_from_z(X)
    tang = np.full(len(y), np.nan)
    skf_t = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf_t.split(X, y):
        Minv = tangent_reference(C_mats[tr][y[tr] == 0])
        F = tangent_features(C_mats, Minv)
        Ctr = None if COV is None else COV[tr]; Cte = None if COV is None else COV[te]
        mt = ControlAnchoredModel(seed=RANDOM_STATE).fit(F[tr], y[tr], sites[tr], Ctr)
        tang[te] = mt.score(F[te], sites[te], Cte)
    method_scores[f"Tangent space + {DIRECTION}"] = tang
    pd.DataFrame({"SUB_ID": df["SUB_ID"].values, "TANGENT_OOF_SCORE": tang}).to_csv(
        os.path.join(OUTPUT_DIR, "tangent_oof_scores.csv"), index=False)

dflt_key = [k for k in method_scores if k.startswith("Default")][0]
mrows = []
for lbl, sc_ in method_scores.items():
    e = effect_table(sc_, y, BOOTSTRAP_N, RANDOM_STATE, lbl)
    if lbl == dflt_key:
        e.update({"AUC_DIFF_VS_DEFAULT": 0.0, "AUC_DIFF_CI_LOW": np.nan,
                  "AUC_DIFF_CI_HIGH": np.nan})
    else:
        dd, lo, hi = paired_auc_diff_ci(sc_, method_scores[dflt_key], y,
                                        BOOTSTRAP_N, RANDOM_STATE)
        e.update({"AUC_DIFF_VS_DEFAULT": dd, "AUC_DIFF_CI_LOW": lo,
                  "AUC_DIFF_CI_HIGH": hi})
    mrows.append(e)
method_df = pd.DataFrame(mrows).rename(columns={"SCORE": "METHOD"})
method_df.to_csv(os.path.join(OUTPUT_DIR, "method_comparison_with_CI.csv"), index=False)

# ============================================================
# Q. HETEROGENEITY: NORMATIVE DEVIATION + ASD SUBGROUP ANALYSIS (reviewer point 5)
# ============================================================
# (i) Normative deviation: RMS of TD-standardised leading PC scores, computed
#     out-of-fold with an out-of-sample TD reference (no single direction forced).
# (ii) Variance ratio of the continuous score Z (ASD/TD): >1 = more spread than
#      a pure mean shift.
# (iii) Exploratory subgrouping of ASD in the TD-normative PC space. The PCA is
#      fitted on TD only, so ASD participants are out-of-sample for it. Reported
#      whether or not subgroups are found; compared with a single-cluster
#      (Gaussian) null.

log("Q. Normative deviation and ASD heterogeneity")
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2_contingency

het = {}
het["DEV_ASD_VS_TD_COHENS_D"] = cohen_d(main["dev"][asd], main["dev"][td])
het["DEV_ASD_VS_TD_AUC"] = auc_rank(main["dev"][asd], main["dev"][td])
het["SPEARMAN_DEV_VS_PATTERN_SCORE_ASD"] = float(
    spearmanr(main["dev"][asd], oof[asd]).statistic)
vr = lambda a, b: np.var(a, ddof=1) / np.var(b, ddof=1)
het["VARIANCE_RATIO_Z_ASD_OVER_TD"] = float(vr(Z[asd], Z[td]))
vlo, vhi = boot_ci_stat(Z[asd], Z[td], vr, BOOTSTRAP_N, RANDOM_STATE)
het["VARIANCE_RATIO_CI_LOW"], het["VARIANCE_RATIO_CI_HIGH"] = vlo, vhi
het["SHARE_ASD_DEV_ABOVE_TD_95TH_REF"] = float((main["dev_I"][asd] > 95).mean())
het["SHARE_TD_DEV_ABOVE_TD_95TH_REF"] = float((main["dev_I"][td] > 95).mean())

kk = min(NORM_K, full.n_components)
U_all = full.pca.transform(full.scaled(X, sites, COV))[:, :kk]
Zn = U_all / np.sqrt(full.pca.explained_variance_[:kk])
ZA = Zn[asd]
sub_rows, labels_by_k = [], {}
for k in range(2, 6):
    km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(ZA)
    labels_by_k[k] = km.labels_
    gm = GaussianMixture(k, n_init=3, random_state=RANDOM_STATE).fit(ZA)
    sub_rows.append({"K": k, "SILHOUETTE": silhouette_score(ZA, km.labels_),
                     "GMM_BIC": gm.bic(ZA),
                     "MIN_CLUSTER_SIZE": int(np.bincount(km.labels_).min())})
gm1 = GaussianMixture(1, random_state=RANDOM_STATE).fit(ZA)
sub_df = pd.DataFrame(sub_rows)
sub_df = pd.concat([pd.DataFrame([{"K": 1, "SILHOUETTE": np.nan,
                                   "GMM_BIC": gm1.bic(ZA),
                                   "MIN_CLUSTER_SIZE": len(ZA)}]), sub_df])
sub_df.to_csv(os.path.join(OUTPUT_DIR, "asd_subgroup_model_selection.csv"), index=False)
k_best = int(sub_df.dropna(subset=["SILHOUETTE"]).sort_values(
    "SILHOUETTE", ascending=False).iloc[0]["K"])
obs_sil = float(sub_df.loc[sub_df["K"] == k_best, "SILHOUETTE"].iloc[0])
het["SUBGROUP_K_BY_SILHOUETTE"] = k_best
het["SUBGROUP_OBS_SILHOUETTE"] = obs_sil
het["SUBGROUP_K_BY_GMM_BIC"] = int(sub_df.sort_values("GMM_BIC").iloc[0]["K"])

# single-cluster Gaussian null for the best-k silhouette
rng_c = np.random.default_rng(RANDOM_STATE)
mu_c, cov_c = ZA.mean(0), np.cov(ZA.T)
null_sil = np.empty(N_CLUSTER_NULL)
for i in range(N_CLUSTER_NULL):
    S = rng_c.multivariate_normal(mu_c, cov_c, len(ZA))
    null_sil[i] = silhouette_score(
        S, KMeans(k_best, n_init=3, random_state=i).fit(S).labels_)
het["SUBGROUP_NULL_SILHOUETTE_MEAN"] = float(null_sil.mean())
het["SUBGROUP_NULL_SILHOUETTE_95TH"] = float(np.percentile(null_sil, 95))
het["SUBGROUP_SILHOUETTE_PERM_P"] = float((1 + np.sum(null_sil >= obs_sil))
                                          / (1 + N_CLUSTER_NULL))

# bootstrap stability (ARI of bootstrap-fitted vs full-sample partition)
km_full = KMeans(k_best, n_init=10, random_state=RANDOM_STATE).fit(ZA)
rng_b = np.random.default_rng(RANDOM_STATE + 1)
ari = []
for i in range(N_CLUSTER_BOOT):
    idx = rng_b.integers(0, len(ZA), len(ZA))
    kb = KMeans(k_best, n_init=3, random_state=i).fit(ZA[idx])
    ari.append(adjusted_rand_score(km_full.labels_, kb.predict(ZA)))
het["SUBGROUP_BOOTSTRAP_ARI_MEAN"] = float(np.mean(ari))
het["SUBGROUP_BOOTSTRAP_ARI_SD"] = float(np.std(ari, ddof=1))

# characterise the partition (exploratory; BH within table)
lab = km_full.labels_
asd_idx = np.where(asd)[0]
char = [{"CLUSTER": c, "N": int((lab == c).sum()),
         "MEAN_I_OOF": float(I[asd_idx][lab == c].mean()),
         "MEAN_Z_OOF": float(Z[asd_idx][lab == c].mean()),
         "MEAN_DEVIATION_OOF": float(main["dev"][asd_idx][lab == c].mean())}
        for c in range(k_best)]
pd.DataFrame(char).to_csv(os.path.join(OUTPUT_DIR, "asd_subgroup_profiles.csv"), index=False)
site_ct = pd.crosstab(lab, sites[asd_idx])
chi, p_site, _, _ = chi2_contingency(site_ct)
het["SUBGROUP_VS_SITE_CRAMERS_V"] = float(np.sqrt(chi / (site_ct.values.sum()
                                         * (min(site_ct.shape) - 1))))
het["SUBGROUP_VS_SITE_CHI2_P"] = float(p_site)
assoc = []
cl_cov = {"AGE": covars["AGE"].values[asd_idx], "FIQ": covars["FIQ"].values[asd_idx],
          "SEX_FEMALE": covars["SEX_F"].values[asd_idx]}
if HAVE_COVARS:
    try:
        ph = df[["SUB_ID"]].merge(pheno, on="SUB_ID", how="left")
        for c_ in ["ADOS_GOTHAM_SEVERITY", "ADOS_TOTAL", "SRS_RAW_TOTAL"]:
            if c_ in ph.columns:
                cl_cov[c_] = pd.to_numeric(ph[c_], errors="coerce").values[asd_idx]
    except Exception:
        pass
for nm, v in cl_cov.items():
    ok_ = np.isfinite(v)
    if ok_.sum() > 10 and len(np.unique(lab[ok_])) > 1:
        p_ = kruskal(*[v[ok_ & (lab == c)] for c in range(k_best)
                       if (ok_ & (lab == c)).sum() > 1]).pvalue
        assoc.append({"VARIABLE": nm, "N": int(ok_.sum()), "KRUSKAL_P": float(p_)})
if assoc:
    ad = pd.DataFrame(assoc); ad["Q_BH"] = benjamini_hochberg(ad["KRUSKAL_P"].values)
    ad.to_csv(os.path.join(OUTPUT_DIR, "asd_subgroup_associations.csv"), index=False)
pd.DataFrame(list(het.items()), columns=["Metric", "Value"]).to_csv(
    os.path.join(OUTPUT_DIR, "heterogeneity_summary.csv"), index=False)
summary.update({f"HET_{k}": v for k, v in het.items()})

# ============================================================
# R. REPORTING NOTES (written to disk with the results)
# ============================================================
notes = f"""
RESEARCH NOTES (auto-generated)
* The pattern index (I), relative expression index (R), Z and normative deviation are
  RESEARCH DESCRIPTORS of group-level ASD-TD connectivity differences. They are not
  diagnostic and not measures of clinical severity; exploratory associations with
  ADOS/SRS are reported below whatever their size.
* Edges are Fisher-z transformed at extraction (np.arctanh of Pearson r), i.e. before
  any averaging, regression or subtraction.
* Inside every training fold, and nowhere else: TD-fitted imputation, site dummies
  {'+ age + sex' + (' + motion' if MOTION_AVAILABLE else '') if (COVARIATE_ADJUST and COV is not None) else '(no covariates)'}
  regression, robust scaling, PCA, direction, and the reference distributions (inner cross-fit).
  Held-out participants only receive these fitted transforms.
* Head motion available: {MOTION_AVAILABLE}. If False, motion-related variance is NOT
  removed; say so as a limitation (N_TIMEPOINTS / global mean FC are not motion).
* The default direction ({DIRECTION}) and the C-grid were chosen after inspecting these
  data (direction comparison and the grid extension). Report all methods in
  method_comparison_with_CI.csv; treat headline performance as optimistic.
* Report all analyses including null/weak ones: clinical associations, sites with small
  or negative effects, subgroup null test (SUBGROUP_SILHOUETTE_PERM_P), tangent comparator.
"""
open(os.path.join(OUTPUT_DIR, "RESEARCH_NOTES.txt"), "w").write(notes)


pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_csv(
    os.path.join(OUTPUT_DIR, "SUMMARY.csv"), index=False)

# ============================================================
# PRINT KEY RESULTS
# ============================================================

pd.set_option("display.width", 220, "display.max_columns", 40)
print("\n" + "=" * 78)
print(f"KEY RESULTS   direction = {DIRECTION}")
print("=" * 78)
print(f"N = {len(y)} (ASD {asd.sum()}, TD {td.sum()}), {len(np.unique(sites))} sites")
print(f"Mean I: ASD {gs['MEAN_ASD']:.1f}, TD {gs['MEAN_TD']:.1f}; diff "
      f"{gs['MEAN_DIFF']:.1f} (95% bootstrap CI {ci_lo:.1f} to {ci_hi:.1f}); "
      f"Mann-Whitney p = {gs['MWU_P']:.2g}; Cohen's d = {gs['COHENS_D']:.2f}; "
      f"AUC (descriptive) = {gs['AUC_DESCRIPTIVE']:.2f}")
print(f"Z vs TD reference: d = {gs_z['COHENS_D']:.2f}")
print(f"I at ceiling (>=99.9): ASD {summary['N_ASD_I_AT_CEILING_GE_99.9']}, "
      f"TD {summary['N_TD_I_AT_CEILING_GE_99.9']}")
print(f"R (ASD): mean {a_R.mean():.1f}, median {np.median(a_R):.1f}, "
      f"SD {a_R.std(ddof=1):.1f}, range {a_R.min():.1f}-{a_R.max():.1f}")
print("\nReference check (TD mean I and ASD mean R should be ~50 with the "
      "out-of-sample reference):")
print(bias_df.round(2).to_string(index=False))
print("\nWithin-site ASD-TD contrast of I (10-fold):")
print(site_tab.round(2).to_string(index=False))
if ss:
    print(f"Sites positive {ss['SITES_POSITIVE']}/{ss['SITES_TOTAL']} "
          f"(sign p one-sided {ss['SIGN_P_ONE_SIDED']:.3f}); random-effects "
          f"{ss['RE_POOLED_DIFF']:.1f} ({ss['RE_CI_LOW']:.1f} to {ss['RE_CI_HIGH']:.1f}), "
          f"I2 = {ss['RE_I2_PERCENT']:.0f}%")
if HAVE_COVARS:
    print("\nConfound models (HC3):")
    print(conf_df.round(3).to_string(index=False))
    print("\nAssociations within ASD (BH q within group):")
    print(cov_df[cov_df["GROUP"] == "ASD"].round(3).to_string(index=False))
print("\nSensitivity of the ASD ranking:")
print(sens_df.round(3).to_string(index=False))
print("\nSensitivity of the group effect (pooled-reference index; compare rows "
      "with each other, not with headline I):")
print(sens_grp_df.round(3).to_string(index=False))
if rel:
    print("\nReliability across random partitions:")
    for k, v in rel.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
print(f"\nPattern stability (observed): raw {obs_stab_raw:.3f}, scaled {obs_stab_scl:.3f}")
if perm_summary is not None:
    print("\nPermutation tests:")
    print(perm_summary.round(4).to_string(index=False))
print(f"\nTop-25 negative loadings: raw {summary['TOP25_NEGATIVE_RAW_OMEGA']}, scaled "
      f"{summary['TOP25_NEGATIVE_SCALED']}, Haufe {summary['TOP25_NEGATIVE_HAUFE']} | "
      f"fraction of all edges negative (raw): "
      f"{summary['ALLEDGES_FRACTION_NEGATIVE_RAW_OMEGA']:.2f}")
print(f"Full-data score vs global mean FC (Spearman): "
      f"{summary['SPEARMAN_FULLFIT_SCORE_VS_GLOBAL_MEAN_FC']:.2f}")
if DIRECTION.startswith("logistic"):
    print(f"Selected C per fold: {summary['SELECTED_C_COUNTS']}")
if RUN_LOSO:
    print("\nLeave-one-site-out (all participants):")
    print(loso_tab.round(2).to_string(index=False))
    print(f"LOSO pooled: mean I ASD {loso_gs['MEAN_ASD']:.1f}, TD {loso_gs['MEAN_TD']:.1f}, "
          f"d = {loso_gs['COHENS_D']:.2f}, AUC = {loso_gs['AUC_DESCRIPTIVE']:.2f}")
    if loso_ss:
        print(f"LOSO sites positive {loso_ss['SITES_POSITIVE']}/{loso_ss['SITES_TOTAL']} "
              f"(sign p one-sided {loso_ss['SIGN_P_ONE_SIDED']:.3f}); random-effects "
              f"{loso_ss['RE_POOLED_DIFF']:.1f} ({loso_ss['RE_CI_LOW']:.1f} to "
              f"{loso_ss['RE_CI_HIGH']:.1f}), I2 = {loso_ss['RE_I2_PERCENT']:.0f}%")
    print(f"LOSO vs 10-fold within ASD: Spearman I = "
          f"{loso_summary['SPEARMAN_I_LOSO_VS_10FOLD_ALL_ASD']:.3f}, R quartile "
          f"agreement {loso_summary['R_QUARTILE_GROUP_AGREEMENT'] * 100:.0f}%")
print("\nEffect sizes with 95% bootstrap CIs (ASD vs TD, out-of-fold):")
print(eff.round(3).to_string(index=False))
if auc_site_sum:
    print(f"Within-site AUC meta-analysis: {auc_site_sum['RE_AUC']:.3f} "
          f"({auc_site_sum['RE_AUC_CI_LOW']:.3f} to {auc_site_sum['RE_AUC_CI_HIGH']:.3f}), "
          f"I2 = {auc_site_sum['RE_AUC_I2']:.0f}%")
print("\nMethod comparison (raw OOF score, same folds, same adjustment):")
print(method_df[["METHOD", "COHENS_D", "D_CI_LOW", "D_CI_HIGH", "AUC", "AUC_CI_LOW",
                 "AUC_CI_HIGH", "AUC_DIFF_VS_DEFAULT", "AUC_DIFF_CI_LOW",
                 "AUC_DIFF_CI_HIGH"]].round(3).to_string(index=False))
print("\nHeterogeneity / normative / subgroup summary:")
for k_, v_ in het.items():
    print(f"  {k_}: {v_:.3f}" if isinstance(v_, float) else f"  {k_}: {v_}")
print("  (subgroups are exploratory; interpret against SUBGROUP_SILHOUETTE_PERM_P)")
print("\nREMINDER: research descriptor only - not diagnostic, not a severity measure.")

print(f"\nOutputs written to: {OUTPUT_DIR}")
log("Done")