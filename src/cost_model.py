# -*- coding: utf-8 -*-
"""
Cost modeling & contractor effect analysis
+ Cost-Performance score追加
"""

import os
import math
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available; MixedLM will be skipped.")

# ------------------------------------------------------------
# Data loading & feature engineering
# ------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")

def build_feature_matrix(df: pd.DataFrame, exclude_cost=True) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = df["cost"].astype(float)
    contractor = df["contractor_id"].astype(str)

    cols_be = [c for c in df.columns if c.startswith("be_")]
    cols_af = [c for c in df.columns if c.startswith("af_")]
    cols_bld = [c for c in df.columns if c in ["マンション","一戸建て","オフィス等・その他"]]
    cols_num = ["age_years"]

    X = df[cols_num + cols_bld + cols_be + cols_af].copy().astype(float)
    return X, y, contractor

# ------------------------------------------------------------
# Target encoding for contractor_id
# ------------------------------------------------------------

def add_contractor_target_encoding(
    X: pd.DataFrame, y: pd.Series, contractor: pd.Series, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    enc = pd.Series(index=X.index, dtype=float)
    global_mean = y.mean()

    for train_idx, valid_idx in kf.split(X):
        tr_c = contractor.iloc[train_idx]
        tr_y = y.iloc[train_idx]
        means = tr_y.groupby(tr_c).mean()
        counts = tr_c.value_counts()

        m = contractor.iloc[valid_idx].map(means).fillna(global_mean)
        c = contractor.iloc[valid_idx].map(counts).fillna(1.0)
        enc.iloc[valid_idx] = (c * m + 5.0 * global_mean) / (c + 5.0)

    X_enc = X.copy()
    X_enc["contractor_te_mean_cost"] = enc.values
    X_enc["contractor_case_count"] = contractor.map(contractor.value_counts()).values
    return X_enc

# ------------------------------------------------------------
# Cross-validated training & evaluation
# ------------------------------------------------------------

def cross_validate_rf(
    X: pd.DataFrame, y: pd.Series, contractor: pd.Series,
    use_contractor: bool = False, n_splits: int = 5, random_state: int = 42,
    n_estimators: int = 400, max_depth: int | None = None,
) -> Dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true_all, y_pred_all = [], []
    importances = []

    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, y_tr = X.iloc[tr].copy(), y.iloc[tr].copy()
        X_va, y_va = X.iloc[va].copy(), y.iloc[va].copy()

        if use_contractor:
            X_tr = add_contractor_target_encoding(X_tr, y_tr, contractor.iloc[tr], n_splits=5, random_state=random_state)
            X_va = X_va.copy()
            tr_means = y_tr.groupby(contractor.iloc[tr]).mean()
            tr_counts = contractor.iloc[tr].value_counts()
            gmean = y_tr.mean()
            m = contractor.iloc[va].map(tr_means).fillna(gmean)
            c = contractor.iloc[va].map(tr_counts).fillna(1.0)
            X_va["contractor_te_mean_cost"] = (c * m + 5.0 * gmean) / (c + 5.0)
            X_va["contractor_case_count"] = contractor.iloc[va].map(tr_counts).fillna(1.0).values

        scaler = StandardScaler(with_mean=True, with_std=True)
        if "age_years" in X_tr.columns:
            X_tr["age_years"] = scaler.fit_transform(X_tr[["age_years"]])
            X_va["age_years"] = scaler.transform(X_va[["age_years"]])

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=random_state, n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)

        y_true_all.append(y_va.values)
        y_pred_all.append(pred)
        importances.append(pd.Series(model.feature_importances_, index=X_tr.columns))

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    rmse = mean_squared_error(y_true_all, y_pred_all, squared=False)
    mae  = mean_absolute_error(y_true_all, y_pred_all)
    r2   = r2_score(y_true_all, y_pred_all)
    imp_mean = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)

    return {"rmse": rmse, "mae": mae, "r2": r2,
            "feature_importances": imp_mean,
            "y_true": y_true_all, "y_pred": y_pred_all}

# ------------------------------------------------------------
# ANOVA on residuals by contractor
# ------------------------------------------------------------

def anova_on_residuals(y_true: np.ndarray, y_pred: np.ndarray, contractor: pd.Series) -> Dict:
    resid = y_true - y_pred
    df = pd.DataFrame({"resid": resid, "contractor": contractor.values})
    groups = df.groupby("contractor")["resid"]
    group_means = groups.mean()
    counts = groups.count()
    overall_mean = df["resid"].mean()

    ss_between = ((group_means - overall_mean) ** 2 * counts).sum()
    ss_within = groups.apply(lambda g: ((g - g.mean())**2).sum()).sum()
    df_between = group_means.size - 1
    df_within = df.shape[0] - group_means.size
    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)
    F = ms_between / ms_within if ms_within > 0 else np.inf
    eta_p2 = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else np.nan

    return {"F": F, "df_between": int(df_between), "df_within": int(df_within),
            "eta_partial_sq": float(eta_p2)}

# ------------------------------------------------------------
# Cost-Performance Score追加
# ------------------------------------------------------------

def oof_predictions_baseline(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                             random_state: int = 42, n_estimators: int = 200,
                             max_depth: int | None = None) -> np.ndarray:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)

    for tr, va in kf.split(X):
        X_tr, y_tr = X.iloc[tr].copy(), y.iloc[tr].copy()
        X_va = X.iloc[va].copy()
        scaler = StandardScaler(with_mean=True, with_std=True)
        if "age_years" in X_tr.columns:
            X_tr["age_years"] = scaler.fit_transform(X_tr[["age_years"]])
            X_va["age_years"] = scaler.transform(X_va[["age_years"]])
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=random_state, n_jobs=-1)
        model.fit(X_tr, y_tr)
        oof[va] = model.predict(X_va)
    return oof

def compute_cost_performance_scores(df: pd.DataFrame, n_splits: int = 5,
                                    seed: int = 42, n_estimators: int = 200) -> pd.DataFrame:
    X, y, contractor = build_feature_matrix(df)
    oof_pred = oof_predictions_baseline(X, y, n_splits=n_splits,
                                        random_state=seed, n_estimators=n_estimators)
    delta = oof_pred - y.values
    dmin, dmax = float(np.min(delta)), float(np.max(delta))
    if dmax - dmin < 1e-9:
        score = np.full_like(delta, 50.0)
    else:
        score = (delta - dmin) / (dmax - dmin) * 100.0
    df_out = df.copy()
    df_out["cp_score_0_100"] = np.round(score).astype(int)
    df_out["cp_delta_expected_minus_actual"] = delta
    return df_out

def write_scored_csv(input_csv: str, output_csv: str,
                     n_splits: int = 5, seed: int = 42, n_estimators: int = 200) -> str:
    dfin = load_data(input_csv)
    dfout = compute_cost_performance_scores(dfin, n_splits=n_splits,
                                            seed=seed, n_estimators=n_estimators)
    dfout.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv
