# excel_gui_v2.py

import os
import time
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ====== 設定 ======
MODEL_NAME = "gpt-4o-mini"   # 速さ/コスト重視の現実解
TEMPERATURE = 0.2
PER_ROW_DELAY_SEC = 0.2      # 行ごとのインターバル（簡易レート制御）
THIS_YEAR = 2025             # built_ym 推定用（運用年に合わせて変更）

# 日本語(論理名) → 物理名 のマッピング
LOGICAL_TO_PHYSICAL = {
    "住所": "location",
    "築年": "building_age",
    "面積": "floor_area",
    "築造": "structure",
    "状態": "current_condition",
    "希望用": "desired_use",
}

PHYSICAL_TO_LOGICAL = {v: k for k, v in LOGICAL_TO_PHYSICAL.items()}

# 出力列（10列に拡張）
OUTPUT_COLS = [
    "案", "初期費用目安", "工期感", "メリット", "デメリット",
    "主なリスク", "向いている施主", "解体メリット要約",
    "解体の適否（暫定）", "根拠（データ/LLM）",
    # --- ここから追加 ---
    "モデル根拠TOP1", "モデル根拠TOP2", "モデル根拠TOP3"
]

# ====== OpenAI 初期化 ======
load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """あなたは建築・不動産の実務に詳しい営業アシスタントAIです。
与えられた物件情報とデータ分析シグナル（確率・スコア）を踏まえ、
「建て替え / リノベ / 解体・更地売却」の3案を比較し、さらに解体案の是非を説明します。

【必ず守ること】
- 出力は **厳密に JSON配列**（長さ3）と **JSONオブジェクト**（解体是非の付加情報）の2要素を同時に返すのではなく、
  **1つの JSON オブジェクト**で返すこと。キーは：
  - "plans": 3案の配列（順に 建て替え/リノベ/解体・更地売却）
  - "demolition_summary": { "解体メリット要約", "解体の適否（暫定）", "根拠（データ/LLM）" }
- "plans" 内の各要素のキーは日本語で固定：
  「案」「初期費用目安」「工期感」「メリット」「デメリット」「主なリスク」「向いている施主」
- 数値は不確実なら幅で。制度・規制は断定せず「可能性」「要確認」を用いる。
- セル表示を意識し、簡潔に箇条書き寄りで。"""

USER_TEMPLATE = """次の入力に対して出力してください。

[物件情報]
- 所在地: {location}
- 築年数: {building_age} 年
- 延床面積: {floor_area} ㎡
- 構造: {structure}
- 現況: {current_condition}
- 施主の希望: {desired_use}

[データ分析シグナル（参考）]
- land_like_probability: {land_prob:.3f}
- demolition_score(0-100): {demo_score}
- 推定 built_ym: {built_ym}
- x_max（不足時は平均で補完）: {x_max_filled}
- region_type: {region_type}
- 簡易根拠: {brief_basis}

【出力仕様（JSON オブジェクトのみ）】
{{
  "plans": [
    {{
      "案": "建て替え",
      "初期費用目安": "〜〜万円（根拠）",
      "工期感": "〜ヶ月",
      "メリット": "・・・",
      "デメリット": "・・・",
      "主なリスク": "・・・",
      "向いている施主": "・・・"
    }},
    {{
      "案": "リノベ",
      "初期費用目安": "〜〜万円（根拠）",
      "工期感": "〜ヶ月",
      "メリット": "・・・",
      "デメリット": "・・・",
      "主なリスク": "・・・",
      "向いている施主": "・・・"
    }},
    {{
      "案": "解体・更地売却",
      "初期費用目安": "〜〜万円（根拠）",
      "工期感": "〜ヶ月",
      "メリット": "・・・",
      "デメリット": "・・・",
      "主なリスク": "・・・",
      "向いている施主": "・・・"
    }}
  ],
  "demolition_summary": {{
    "解体メリット要約": "・・・",
    "解体の適否（暫定）": "解体前向き/要検討/優先度低 など",
    "根拠（データ/LLM）": "land_like確率・築年・状態キーワード 等を簡潔に"
  }}
}}"""

# =========================
# データ前処理 & ML 部分
# =========================
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ML_FEATURES = ["x_max", "exclusive_area_sqm", "built_ym", "region_type"]
TARGET_NAME = "y1"

def find_csv_path(base_excel_path: str) -> str:
    """Excel と同じフォルダにある normalized_listings_with_y.csv を優先探索"""
    base_dir = os.path.dirname(base_excel_path)
    candidate = os.path.join(base_dir, "normalized_listings_with_y.csv")
    if os.path.exists(candidate):
        return candidate
    # 見つからない場合はファイルダイアログで選ばせる
    path = filedialog.askopenfilename(
        title="normalized_listings_with_y.csv を選択してください",
        filetypes=[("CSV files", "*.csv")]
    )
    if not path:
        raise FileNotFoundError("normalized_listings_with_y.csv が見つかりません。")
    return path

# ==== 追加：設定 ====
USE_PRETRAINED_MODEL = True
MODEL_BUNDLE_PATH = "models/best_model_y1.joblib"
MODEL_METADATA_PATH = "MODEL_METADATA_y1.json"


def explain_topk_for_row(model, X_pred: pd.DataFrame, feature_names: list, defaults: dict, k: int = 3):
    """
    1行予測に対する上位k要因を返す（文字列リスト）。
    可能なら SHAP（クラス1=土地寄り）の寄与絶対値TOPK、失敗時は
    feature_importances_ × 値の差分 のヒューリスティック。
    """
    # imputer があれば同じ前処理で数値化
    try:
        clf = model.named_steps.get("clf", model)
        imputer = model.named_steps.get("imputer", None) if hasattr(model, "named_steps") else None
        if imputer is not None:
            X_imp = pd.DataFrame(imputer.transform(X_pred), columns=feature_names)
        else:
            X_imp = X_pred.copy()

        # --- 試行1: SHAP ---
        try:
            import shap  # Optional依存。なければ except に落ちる
            try:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_imp)
                # binary のとき list で [class0, class1]
                if isinstance(shap_values, list):
                    sv = shap_values[1]
                else:
                    sv = shap_values
                vals = sv[0]
            except Exception:
                # 新API（shap.Explainer）にフォールバック
                explainer = shap.Explainer(clf)
                sv = explainer(X_imp)
                vals = sv.values[0]
            abs_idx = np.argsort(np.abs(vals))[::-1][:k]
            out = []
            for i in abs_idx:
                feat = feature_names[i]
                val = X_imp.iloc[0, i]
                contrib = float(vals[i])
                direction = "土地寄りに＋" if contrib >= 0 else "土地寄りに－（=建物寄り）"
                out.append(f"{feat}={round(float(val), 3)} / 影響={contrib:+.3f} [{direction}]")
            return out

        except Exception:
            # shap 未導入・失敗 → ヒューリスティックへ
            pass

        # --- 試行2: 重要度×差分の簡易ヒューリスティック ---
        if hasattr(clf, "feature_importances_"):
            fi = np.asarray(clf.feature_importances_, dtype=float)
            diffs = []
            for i, f in enumerate(feature_names):
                x = float(X_imp.iloc[0, i]) if pd.notnull(X_imp.iloc[0, i]) else 0.0
                mu = float(defaults.get(f, 0.0))
                diffs.append(abs(x - mu))
            score = fi * np.asarray(diffs)
            idx = np.argsort(score)[::-1][:k]
            out = []
            for i in idx:
                feat = feature_names[i]
                val = float(X_imp.iloc[0, i]) if pd.notnull(X_imp.iloc[0, i]) else 0.0
                out.append(f"{feat}={round(val,3)} / 重要度×差分={score[i]:.3f}")
            return out

        # --- 最終フォールバック ---
        return ["built_ym（古いほど解体寄り）", "exclusive_area_sqm（面積特性）", "x_max（立地指標）"][:k]

    except Exception as e:
        return [f"解釈生成に失敗: {e}"]



import joblib

# ==== 追加：bundle 読み込み & 代表値ロード ====
def load_bundle_and_defaults(csv_path: str, model_path: str, meta_path: str):
    bundle = joblib.load(model_path)  # {"model","features","threshold",...}
    model = bundle["model"]
    features = bundle["features"]
    threshold = float(bundle.get("threshold", 0.5))

    # メタがあれば features/threshold を優先（将来の差替えに強い）
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        features = meta.get("features", features)
        threshold = float(meta.get("threshold", threshold))

    # 学習CSVから代表値（欠損補完に使う）
    df_train = pd.read_csv(csv_path, encoding="utf-8-sig")
    defaults = {}
    for c in features:
        if c in df_train.columns:
            s = pd.to_numeric(df_train[c], errors="coerce")
            if c in {"station_walk","built_ym","LDK","1R","K","S","nando","region_type"}:
                defaults[c] = int(round(s.dropna().median())) if s.notna().any() else 0
            else:
                defaults[c] = float(s.dropna().median()) if s.notna().any() else 0.0
        else:
            # 未知列は無難なデフォルト
            defaults[c] = 0

    return model, features, threshold, defaults

# ==== 追加：region_type の数値化（学習時は数値特徴） ====
def _region_type_to_int(region_type_str: str, location: str) -> int:
    # 学習データが数値カテゴリのため、簡易ルールで 0/1 に寄せる
    s = (region_type_str or "").strip().lower()
    if s.isdigit():
        return int(s)
    if any(k in location for k in ["東京", "東京都", "千代田", "中央", "港", "新宿", "渋谷", "世田谷"]):
        return 1
    if any(k in location for k in ["横浜","川崎","大阪","名古屋","札幌","福岡"]):
        return 1
    return 0

# ==== 追加：Excel 1行 → features 13列の DF 1行に成形 ====
def build_feature_row(row: pd.Series, features: list, defaults: dict, x_max_mean: float) -> pd.DataFrame:
    d = {f: defaults.get(f, 0) for f in features}

    # 既存GUIの推定値を活用
    building_age = _to_float(row.get("building_age", ""))
    floor_area = _to_float(row.get("floor_area", ""))
    location = str(row.get("location", "") or "")
    structure = str(row.get("structure", "") or "")

    # built_ym 推定（YYYYMM）
    built_year = None
    if pd.notnull(building_age):
        try:
            built_year = int(THIS_YEAR - float(building_age))
        except Exception:
            built_year = None
    if built_year and built_year > 0:
        built_ym = int(built_year) * 100 + 1  # 例：1980 -> 198001
    else:
        built_ym = np.nan

    # 可能な限り上書き（なければ defaults のまま）
    if "built_ym" in d and not pd.isna(built_ym):
        d["built_ym"] = int(built_ym)
    if "exclusive_area_sqm" in d and pd.notnull(floor_area):
        d["exclusive_area_sqm"] = float(floor_area)
    if "x_max" in d and pd.notnull(x_max_mean):
        d["x_max"] = float(x_max_mean)

    # region_type → 数値
    if "region_type" in d:
        d["region_type"] = _region_type_to_int(str(row.get("region_type", "")), location)

    # station_walk があれば使う（なければ defaults）
    sw = _to_float(row.get("station_walk", ""))
    if "station_walk" in d and pd.notnull(sw):
        d["station_walk"] = int(round(sw))

    # LDK/K/1R/S/nando は Excelに無い想定 → デフォルト（0）でOK

    X_pred = pd.DataFrame([d], columns=features)
    # 数値化（安全のため）
    for c in X_pred.columns:
        X_pred[c] = pd.to_numeric(X_pred[c], errors="coerce")
    return X_pred

def load_and_train_model(csv_path: str):
    if USE_PRETRAINED_MODEL and os.path.exists(MODEL_BUNDLE_PATH):
        model, features, threshold, defaults = load_bundle_and_defaults(
            csv_path=csv_path,
            model_path=MODEL_BUNDLE_PATH,
            meta_path=MODEL_METADATA_PATH
        )
        # 代表値（既存GUIの x_max_filled に流用）
        x_max_mean = defaults.get("x_max", 0.0)
        # 併せて格納（後で使うため）
        load_and_train_model._features = features
        load_and_train_model._threshold = threshold
        load_and_train_model._defaults = defaults
        return model, x_max_mean

    # ←フォールバックとして従来の簡易学習（あなたの元コード）を残す場合は、
    # ここに以前の実装を置いてください
    raise FileNotFoundError("事前学習モデルが見つかりません。MODEL_BUNDLE_PATH を確認してください。")

def estimate_signals_for_row(row: pd.Series, model, x_max_mean: float):
    features = getattr(load_and_train_model, "_features")
    threshold = getattr(load_and_train_model, "_threshold")
    defaults = getattr(load_and_train_model, "_defaults")

    # 既存の説明用値
    building_age = _to_float(row.get("building_age", ""))
    floor_area = _to_float(row.get("floor_area", ""))
    location = str(row.get("location", "") or "")
    structure = str(row.get("structure", "") or "")
    desired_use = str(row.get("desired_use", "") or "")

    # 本番推論 1行分の特徴行を構築
    X_pred = build_feature_row(row, features, defaults, x_max_mean)

    try:
        land_prob = float(model.predict_proba(X_pred)[0, 1])  # y1=1 確率
    except Exception:
        land_prob = 0.5

    # 既存のペナルティ/ボーナス
    penalty = 0
    condition = str(row.get("current_condition", "") or "")
    kw_map = {"クラック":12,"ひび":8,"配管":10,"漏水":12,"シロアリ":15,"全面改修":15,"雨漏り":12,"傾き":15,"違法":15}
    for k, w in kw_map.items():
        if k in condition:
            penalty += w

    age_bonus = 0
    if pd.notnull(building_age):
        if building_age >= 35: age_bonus = 15
        elif building_age >= 25: age_bonus = 8
        elif building_age >= 15: age_bonus = 3

    demo_score = int(np.clip(100 * land_prob + penalty + age_bonus, 0, 100))

    brief_basis = (
        f"P(land_like)={land_prob:.2f}, penalty={penalty}, age_bonus={age_bonus}, "
        f"築年={building_age}, 面積={floor_area}, 構造={structure}, 希望={desired_use}, "
        f"thr={threshold:.3f}"
    )

    # ★ 追加：モデル根拠 TOP3
    top3 = explain_topk_for_row(model, X_pred, features, defaults, k=3)

    return {
        "land_prob": land_prob,
        "demo_score": demo_score,
        "built_ym": int(X_pred.iloc[0]["built_ym"]) if pd.notnull(X_pred.iloc[0]["built_ym"]) else "N/A",
        "x_max_filled": round(float(X_pred.iloc[0]["x_max"]), 2) if pd.notnull(X_pred.iloc[0]["x_max"]) else "N/A",
        "region_type": int(X_pred.iloc[0]["region_type"]) if pd.notnull(X_pred.iloc[0]["region_type"]) else "N/A",
        "brief_basis": brief_basis,
        # ここで返す
        "model_top": top3
    }


def _to_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return float(str(v).replace(",", "").strip())
    except Exception:
        return np.nan

# =========================
# 既存ロジック（列正規化/LLM呼び出し/Excel保存）
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        col_strip = str(col).strip()
        if col_strip in PHYSICAL_TO_LOGICAL:
            col_map[col] = col_strip
            continue
        if col_strip in LOGICAL_TO_PHYSICAL:
            col_map[col] = LOGICAL_TO_PHYSICAL[col_strip]
            continue
        col_map[col] = col_strip
    return df.rename(columns=col_map)

def row_to_payload(row: pd.Series) -> dict:
    return {
        "location": row.get("location", ""),
        "building_age": row.get("building_age", ""),
        "floor_area": row.get("floor_area", ""),
        "structure": row.get("structure", ""),
        "current_condition": row.get("current_condition", ""),
        "desired_use": row.get("desired_use", ""),
    }

def call_gpt(payload: dict, signals: dict) -> dict:
    """GPT 呼び出し。JSONオブジェクト（plans + demolition_summary）を返す。"""
    user_prompt = USER_TEMPLATE.format(
        location=payload.get("location", ""),
        building_age=payload.get("building_age", ""),
        floor_area=payload.get("floor_area", ""),
        structure=payload.get("structure", ""),
        desired_use=payload.get("desired_use", ""),
        current_condition=payload.get("current_condition", ""),
        land_prob=signals["land_prob"],
        demo_score=signals["demo_score"],
        built_ym=signals["built_ym"],
        x_max_filled=signals["x_max_filled"],
        region_type=signals["region_type"],
        brief_basis=signals["brief_basis"],
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()

    # JSONだけが返るとは限らないので安全にパース
    if content.startswith("```"):
        content = content.strip("`")
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline+1:]
    data = json.loads(content)

    # 期待構造：
    # {"plans":[{...},{...},{...}],
    #  "demolition_summary":{"解体メリット要約":"...","解体の適否（暫定）":"...","根拠（データ/LLM）":"..."}}
    if not isinstance(data, dict) or "plans" not in data or "demolition_summary" not in data:
        raise ValueError("GPTの出力が仕様のJSONオブジェクトではありません")

    # plans 正規化
    plans = []
    for item in data.get("plans", []):
        plans.append({
            "案": item.get("案", ""),
            "初期費用目安": item.get("初期費用目安", ""),
            "工期感": item.get("工期感", ""),
            "メリット": item.get("メリット", ""),
            "デメリット": item.get("デメリット", ""),
            "主なリスク": item.get("主なリスク", ""),
            "向いている施主": item.get("向いている施主", ""),
        })
    # demolition_summary 正規化
    ds = data.get("demolition_summary", {})
    demolition_summary = {
        "解体メリット要約": ds.get("解体メリット要約", ""),
        "解体の適否（暫定）": ds.get("解体の適否（暫定）", ""),
        "根拠（データ/LLM）": ds.get("根拠（データ/LLM）", ""),
    }
    return {"plans": plans, "demolition_summary": demolition_summary}

def plans_to_multiline(plans: list, key: str) -> str:
    vals = [str(p.get(key, "")).strip() for p in plans]
    return "\n".join(vals)

def process_excel(path: str) -> str:
    # 学習データロード & 学習
    csv_path = find_csv_path(path)
    model, x_max_mean = load_and_train_model(csv_path)

    # 入力読み込み（最初のシート）
    df = pd.read_excel(path)
    df = normalize_columns(df)

    # 必要カラムがあるかチェック（最低限6項目のどれか）
    required_phys = ["location", "building_age", "floor_area", "structure", "current_condition", "desired_use"]
    if not any(col in df.columns for col in required_phys):
        raise ValueError("入力Excelに必要な列（住所/築年/面積/築造/状態/希望用 または location/building_age/...）が見つかりません。")

    # 出力列を用意（既にあるなら上書き）
    for col in OUTPUT_COLS:
        df[col] = ""

    # 行ごとに処理
    for idx, row in df.iterrows():
        payload = row_to_payload(row)
        try:
            signals = estimate_signals_for_row(row, model, x_max_mean)
            result = call_gpt(payload, signals)

            # plans → 7列
            plans = result["plans"]
            for col in ["案", "初期費用目安", "工期感", "メリット", "デメリット", "主なリスク", "向いている施主"]:
                df.at[idx, col] = plans_to_multiline(plans, col)

            # demolition_summary → 3列
            ds = result["demolition_summary"]
            df.at[idx, "解体メリット要約"] = ds.get("解体メリット要約", "")
            df.at[idx, "解体の適否（暫定）」"] = ds.get("解体の適否（暫定）", "")  # ←後で正規名に揃える
            df.at[idx, "根拠（データ/LLM）"] = ds.get("根拠（データ/LLM）", "")

            # 列名のタイポ修正（上で一瞬ズレを作った場合に備える）
            if "解体の適否（暫定）」" in df.columns and "解体の適否（暫定）" in df.columns:
                # 万一両方ある場合はマージ
                m = df["解体の適否（暫定）"].astype(str).str.strip()
                n = df["解体の適否（暫定）」"].astype(str).str.strip()
                df["解体の適否（暫定）"] = np.where(m.eq(""), n, m)
                df.drop(columns=["解体の適否（暫定）」"], inplace=True)
            elif "解体の適否（暫定）」" in df.columns:
                df.rename(columns={"解体の適否（暫定）」": "解体の適否（暫定）"}, inplace=True)
            # ▼ここから追記：モデル根拠 TOP1〜3 を出力
            tops = signals.get("model_top", [])
            df.at[idx, "モデル根拠TOP1"] = tops[0] if len(tops) > 0 else ""
            df.at[idx, "モデル根拠TOP2"] = tops[1] if len(tops) > 1 else ""
            df.at[idx, "モデル根拠TOP3"] = tops[2] if len(tops) > 2 else ""
            # ▲ここまで

        except Exception as e:
            msg = f"生成失敗: {e}"
            for col in OUTPUT_COLS:
                df.at[idx, col] = msg
        time.sleep(PER_ROW_DELAY_SEC)

    # 保存先
    base, ext = os.path.splitext(path)
    out_path = f"{base}_処理済み.xlsx"
    df.to_excel(out_path, index=False)
    return out_path

# ====== GUI (tkinter) ======
def run_gui():
    root = tk.Tk()
    root.title("空き物件アドバイザー（Excel処理：データ×LLM）")

    frm = tk.Frame(root, padx=16, pady=16)
    frm.pack()

    lbl = tk.Label(frm, text="Excelファイル（.xlsx）を選んで処理します。学習CSVは同フォルダの normalized_listings_with_y.csv を自動探索します。")
    lbl.grid(row=0, column=0, sticky="w")

    status_var = tk.StringVar(value="待機中")
    status = tk.Label(frm, textvariable=status_var, fg="gray")
    status.grid(row=1, column=0, sticky="w", pady=(8, 16))

    def on_select():
        path = filedialog.askopenfilename(
            title="Excelを選択",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not path:
            return
        try:
            status_var.set("処理中…（行ごとに分析→LLM生成）")
            root.update_idletasks()
            out = process_excel(path)
            status_var.set(f"完了: {out}")
            messagebox.showinfo("完了", f"処理が完了しました。\n\n保存先:\n{out}")
        except Exception as e:
            status_var.set("エラー発生")
            messagebox.showerror("エラー", str(e))

    btn = tk.Button(frm, text="Excelを選択して処理", command=on_select)
    btn.grid(row=2, column=0, sticky="w")

    root.mainloop()

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("※ OPENAI_API_KEY が環境変数または .env に設定されていません。")
    run_gui()
