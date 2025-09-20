import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# データを読み込む
df = pd.read_csv("normalized_listings_with_y.csv", encoding="utf-8-sig")

# 特徴量とターゲット
X_cols = ["station_walk","built_ym","exclusive_area_sqm","LDK","1R","K","S","nando",
          "region_type","x_min","x_max","y_min","y_max"]
X = df[X_cols].copy()
y = df["y1"]

# 欠損値を除外
data = pd.concat([X,y], axis=1).dropna()
X = data[X_cols]
y = data["y1"]

# --- 1. ジニ重要度（決定木ベース） ---
clf = DecisionTreeClassifier(random_state=0, max_depth=4)
clf.fit(X, y)
gini_importance = pd.Series(clf.feature_importances_, index=X_cols)

# --- 2. p値（ロジスティック回帰） ---
X_const = sm.add_constant(X)  # 切片項
model = sm.Logit(y, X_const)
result = model.fit(disp=0)
pvalues = result.pvalues

# --- 結果をまとめる ---
summary = pd.DataFrame({
    "gini_importance": gini_importance,
    "p_value": pvalues[X_cols]  # 切片以外
}).sort_values("gini_importance", ascending=False)

#print(summary)

##########################################################

recalculate = False

if recalculate:
    # train_models.py
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from pprint import pprint

    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score, precision_recall_curve,
        average_precision_score, classification_report, confusion_matrix
    )
    import joblib

    # -----------------------------
    # 1) 入力と特徴量
    # -----------------------------
    CSV_PATH = "normalized_listings_with_y.csv"
    TARGET = "y1"  # ← y2 に切替えても動きます

    X_cols = [
        "station_walk","built_ym","exclusive_area_sqm","LDK","1R","K","S","nando",
        "region_type","x_min","x_max","y_min","y_max"
    ]

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    data = df[X_cols + [TARGET]].copy()

    # 目的変数の欠損だけを落とす（特徴量欠損は Imputer で処理）
    data = data.dropna(subset=[TARGET])
    X = data[X_cols]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -----------------------------
    # 2) モデルとチューニング範囲
    # -----------------------------
    pipelines_and_params = [
        (
            "LogisticRegression",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, class_weight="balanced", solver="lbfgs"
                ))
            ]),
            {
                "clf__C": [0.1, 1.0, 3.0, 10.0]
            }
        ),
        (
            "RandomForest",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    random_state=42, n_jobs=-1, class_weight="balanced"
                ))
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 6, 12],
                "clf__min_samples_leaf": [1, 3, 5]
            }
        ),
        (
            "HistGradientBoosting",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(
                    random_state=42, validation_fraction=0.1, early_stopping=True
                ))
            ]),
            {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [None, 6, 12],
                "clf__max_iter": [200, 400]
            }
        ),
    ]

    results = []
    best_overall = None  # (name, grid.best_estimator_, test_scores_dict, best_threshold)

    def fit_and_eval(name, pipe, param_grid):
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        # 予測確率
        proba = best.predict_proba(X_test)[:, 1]
        # デフォルトしきい値0.5
        y_pred_05 = (proba >= 0.5).astype(int)

        # F1最適化しきい値
        prec, rec, thr = precision_recall_curve(y_test, proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = np.nanargmax(f1s)
        best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
        y_pred_opt = (proba >= best_thr).astype(int)

        scores = {
            "cv_best_params": grid.best_params_,
            "test_roc_auc": roc_auc_score(y_test, proba),
            "test_avg_precision": average_precision_score(y_test, proba),
            "test_f1@0.5": f1_score(y_test, y_pred_05),
            "test_acc@0.5": accuracy_score(y_test, y_pred_05),
            "test_f1@opt": f1_score(y_test, y_pred_opt),
            "test_acc@opt": accuracy_score(y_test, y_pred_opt),
            "best_threshold": float(best_thr),
            "report@opt": classification_report(y_test, y_pred_opt, digits=4),
            "confusion_matrix@opt": confusion_matrix(y_test, y_pred_opt).tolist(),
        }
        return best, scores

    for name, pipe, grid in pipelines_and_params:
        best_model, scores = fit_and_eval(name, pipe, grid)
        results.append((name, scores))
        print(f"\n=== {name} ===")
        pprint(scores)

    # ベスト（ROC-AUC基準）を選ぶ
    best_name, best_scores = max(results, key=lambda x: x[1]["test_roc_auc"])
    # 再フィットして保存用オブジェクト化（Grid内のbest_estimator_が欲しいため再取得）
    for name, pipe, grid in pipelines_and_params:
        if name == best_name:
            final_grid = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)
            best_estimator = final_grid.best_estimator_
            break

    bundle = {
        "model_name": best_name,
        "model": best_estimator,
        "features": X_cols,
        "target": TARGET,
        "threshold": best_scores["best_threshold"],
        "cv_best_params": best_scores["cv_best_params"],
    }
    Path("models").mkdir(exist_ok=True)
    joblib.dump(bundle, f"models/best_model_{TARGET}.joblib")
    print(f"\nSaved: models/best_model_{TARGET}.joblib")

    # 参考：係数/重要度（可用なら表示）
    clf = best_estimator.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=X_cols).sort_values(ascending=False)
        print("\nFeature importances:")
        print(fi.to_string())
    elif hasattr(clf, "coef_"):
        coef = pd.Series(clf.coef_[0], index=X_cols).sort_values(key=np.abs, ascending=False)
        print("\nLogistic coefficients (sorted by |coef|):")
        print(coef.to_string())

##########################################################

# === ここから追記（ファイル末尾にそのまま貼り付け） =========================
import os, json, platform
from datetime import datetime
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_recall_curve, confusion_matrix
)
import sklearn
import statsmodels

# ------------------------------------------------------------
# 1) モデルとデータのロード
# ------------------------------------------------------------
MODEL_PATH = "models/best_model_y1.joblib"
CSV_PATH   = "normalized_listings_with_y.csv"
assert os.path.exists(MODEL_PATH), f"Not found: {MODEL_PATH}"
assert os.path.exists(CSV_PATH),   f"Not found: {CSV_PATH}"

bundle = joblib.load(MODEL_PATH)
model_name   = bundle.get("model_name")
features     = bundle.get("features")
target       = bundle.get("target")
threshold    = float(bundle.get("threshold"))
cv_best      = bundle.get("cv_best_params", {})
pipeline     = bundle.get("model")

# --- ここから差し替え（y_all の作り方）--------------------------------

df_full = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
# infは学習に使えないのでNaNへ
df_full.replace([np.inf, -np.inf], np.nan, inplace=True)

# 特徴量存在チェック
missing = [c for c in features if c not in df_full.columns]
if missing:
    raise ValueError(f"Missing feature columns in CSV: {missing}")
if target not in df_full.columns:
    raise ValueError(f"Target column '{target}' not found in CSV.")

# ターゲットを数値化（0/1以外や文字→ NaN）
y_raw = pd.to_numeric(df_full[target], errors="coerce")

# 学習に使える行＝ y が {0,1} のみ
valid_mask = y_raw.isin([0, 1])

# 参考用に、落ちた行をCSVに出す（デバッグ用）
dropped = df_full.loc[~valid_mask, ["property_name","location",target]]
if len(dropped):
    dropped.to_csv("rows_dropped_missing_or_invalid_target.csv",
                   index=False, encoding="utf-8-sig")
    print(f"[WARN] Dropped {len(dropped)} rows due to invalid '{target}'. "
          f"Saved: rows_dropped_missing_or_invalid_target.csv")

# 学習・評価に使うデータ
X_all = df_full.loc[valid_mask, features].copy()
y_all = y_raw.loc[valid_mask].astype(int).values
# --- ここまで差し替え ---------------------------------------------------


# ------------------------------------------------------------
# 2) メトリクス算出（OOF：5-fold の out-of-fold でより妥当な推定）
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# pipeline は自動で fold ごとに再学習される
oof_proba = cross_val_predict(
    pipeline, X_all, y_all, cv=skf, method="predict_proba", n_jobs=-1
)[:, 1]

roc_auc = float(roc_auc_score(y_all, oof_proba))
ap      = float(average_precision_score(y_all, oof_proba))

# しきい値＝bundle保存値での評価
y_pred_thr = (oof_proba >= threshold).astype(int)
acc_thr = float(accuracy_score(y_all, y_pred_thr))
f1_thr  = float(f1_score(y_all, y_pred_thr))
cm_thr  = confusion_matrix(y_all, y_pred_thr).tolist()

# F1最適化しきい値（参考）
prec, rec, thr = precision_recall_curve(y_all, oof_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1s))
best_threshold_oof = float(thr[best_idx]) if best_idx < len(thr) else threshold
y_pred_oof = (oof_proba >= best_threshold_oof).astype(int)
acc_oof = float(accuracy_score(y_all, y_pred_oof))
f1_oof  = float(f1_score(y_all, y_pred_oof))
cm_oof  = confusion_matrix(y_all, y_pred_oof).tolist()

# ------------------------------------------------------------
# 3) 推論入力サンプル（pred_payload_example.csv）
#    - features の並びで 1 行ダミー（中央値など）を出力
# ------------------------------------------------------------
payload_dir = Path(".")
payload_path = payload_dir / "pred_payload_example.csv"

# 中央値ベースで一行組み立て（整数っぽい列は四捨五入）
median_vals = X_all.median(numeric_only=True)

# 整数扱いしたい列（必要に応じて調整）
int_like_cols = {"station_walk","built_ym","LDK","1R","K","S","nando","region_type"}
row = {}
for col in features:
    val = median_vals.get(col, 0.0)
    if col in int_like_cols:
        row[col] = int(round(val))
    else:
        # 座標や面積は小数をそのまま
        row[col] = float(val)

pd.DataFrame([row], columns=features).to_csv(payload_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 4) メタデータ JSON（MODEL_METADATA_y1.json）
#    - しきい値、CV設定、OOFメトリクス、環境バージョンなど
# ------------------------------------------------------------
meta = {
    "model_file": str(MODEL_PATH),
    "model_name": model_name,
    "target": target,
    "features": features,
    "threshold": threshold,
    "cv_best_params": cv_best,
    "training_data": {
        "csv_path": str(CSV_PATH),
        "n_samples_total": int(len(df_full)),       # 総件数
        "n_samples_used":  int(len(X_all)),         # ラベル有効で使えた件数
        "n_positive_used": int(int(y_all.sum())),
        "positive_rate_used": float(y_all.mean())
    },
    "metrics_oof": {
        "roc_auc": roc_auc,
        "avg_precision": ap,
        "f1_at_saved_threshold": f1_thr,
        "acc_at_saved_threshold": acc_thr,
        "confusion_matrix_at_saved_threshold": cm_thr,
        "f1_opt": f1_oof,
        "acc_opt": acc_oof,
        "best_threshold_oof": best_threshold_oof,
        "confusion_matrix_at_best_threshold": cm_oof,
    },
    "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "joblib": joblib.__version__,
    },
    "artifacts": {
        "pred_payload_example_csv": str(payload_path),
    },
    "file_times_utc": {
        "model_mtime": datetime.utcfromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat() + "Z",
        "data_mtime":  datetime.utcfromtimestamp(os.path.getmtime(CSV_PATH)).isoformat() + "Z",
        "metadata_generated_at": datetime.utcnow().isoformat() + "Z",
    }
}

with open("MODEL_METADATA_y1.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n[OK] Wrote files:")
print(f" - pred_payload_example.csv")
print(f" - MODEL_METADATA_y1.json")
# === 追記ここまで =============================================================
