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

print(summary)
