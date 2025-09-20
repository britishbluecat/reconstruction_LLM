# address_normalize.py

import pandas as pd
from pathlib import Path

# 関数群をインポート
from address_normalize_functions import (
    # 既存
    clean_property_name,
    normalize_location_numbers,
    normalize_exclusive_area,
    normalize_price,
    normalize_station_walk,
    normalize_built_year,
    normalize_layout,
    add_ward_city_and_city_town,
    # 追加で使う
    build_folder_level_csvs,
    TARGET_COLUMNS,
    DATA_DIR,
)

# まず各サブフォルダの CSV を 1 本にまとめる（data/<subdir>.csv を作る）
build_folder_level_csvs()

# 処理対象（フォルダ名＝ファイル源 表示名）
GROUPS = [
    "kodate_model_chuko",
    "kodate_model_shin",
    "mansion_model_chuko",
    "mansion_model_shin",
    "tochi",
]

def run_pipeline_on_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    1ファイル分のDFに対して、address_normalize_functions のクレンジング処理を順適用。
    TARGET_COLUMNS が無い場合は None 埋めで列を揃えてから実行。
    """
    df = df.copy()

    # 必要カラムが無ければ None で補完してカラム順を揃える
    cols_available = [c for c in TARGET_COLUMNS if c in df.columns]
    df_selected = df[cols_available].copy()
    for c in TARGET_COLUMNS:
        if c not in df_selected.columns:
            df_selected[c] = None
    df_selected = df_selected[TARGET_COLUMNS]

    # パイプライン
    df_out = (
        df_selected
        .pipe(clean_property_name)
        .pipe(normalize_location_numbers)
        .pipe(add_ward_city_and_city_town)
        .pipe(normalize_exclusive_area)
        .pipe(normalize_station_walk)
        .pipe(normalize_built_year)
        .pipe(normalize_layout)
        .pipe(normalize_price)
    )
    return df_out

all_parts = []

for group in GROUPS:
    in_path = DATA_DIR / f"{group}.csv"
    if not in_path.exists():
        # 無ければスキップ（空データでも処理続行できるようにする）
        continue

    # 読み込み（comma-delimited, UTF-8(BOMあり/なし)両対応）
    try:
        df_in = pd.read_csv(in_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_in = pd.read_csv(in_path, encoding="utf-8")

    # 1ファイル分のクレンジング
    df_norm = run_pipeline_on_df(df_in)

    # ファイル源（フォルダ名）を列として付与
    df_norm["source_group"] = group  # 例: "kodate_model_chuko"

    # （任意）個別に保存したい場合はコメントアウト解除
    # df_norm.to_csv(DATA_DIR / f"normalized_{group}.csv", index=False, encoding="utf-8-sig")

    all_parts.append(df_norm)

# 最後に結合して 1 本化
if all_parts:
    mst_listings = pd.concat(all_parts, ignore_index=True)
else:
    mst_listings = pd.DataFrame(columns=TARGET_COLUMNS + ["ward_city","city_town","LDK","1R","K","S","nando","source_group"])

# built_ym と layout を例外にする
drop_check_cols = [c for c in mst_listings.columns if c not in ["layout", "built_ym"]]

# built_ym / layout を除いた必須カラムに NaN があれば DROP
mst_listings = mst_listings.dropna(subset=drop_check_cols, how="any").reset_index(drop=True)

# ward_city → region_type のマッピング
WARD_CITY_TO_REGION = {
    "文京区": 1, "新宿区": 1, "港区": 1, "渋谷区": 1, "目黒区": 1, "中央区": 1,
    "豊島区": 2, "中野区": 2, "杉並区": 2, "練馬区": 2, "武蔵野市": 2, "三鷹市": 2,
    "府中市": 3, "国分寺市": 3, "立川市": 3, "日野市": 3, "八王子市": 3,
}

# region_type カラムを付与（対応なしは NaN）
mst_listings["region_type"] = mst_listings["ward_city"].map(WARD_CITY_TO_REGION)


from address_normalize_functions import build_geo_bbox_from_coords

# 1) coordinates.csv から bbox を作る（パスは必要に応じて変更）
COORDS_CSV = Path("coordinates.csv")
if COORDS_CSV.exists():
    bbox_df = build_geo_bbox_from_coords(COORDS_CSV)
    # 2) ward_city & city_town で JOIN
    #    ※ city_town は既に address_normalize_functions.add_ward_city_and_city_town() で作成済みを想定
    #    ※ JOIN 後、normalized_listings に x_min,x_max,y_min,y_max を付与
    mst_listings = mst_listings.merge(
        bbox_df,
        on=["ward_city","city_town"],
        how="left",
        validate="m:1"   # 同じ町名に対して bbox は1つに正規化
    )
else:
    # coordinates.csv が未配置なら空列を付与（学習時の欠損扱い）
    for col in ["x_min","x_max","y_min","y_max"]:
        mst_listings[col] = pd.NA


# 最終出力
OUTPUT_FILE = "normalized_listings.csv"
mst_listings.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Wrote {OUTPUT_FILE} with shape={mst_listings.shape}")

#####################################################################

# ===== ベースを1回だけ用意（以後ずっとこれを使う） =====
base = mst_listings.copy()

# --- 重複削除（URLが最強。なければ属性セットで） ---
if "url" in base.columns:
    base = base.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)
else:
    base = base.drop_duplicates(
        subset=["property_name","ward_city","city_town","source_group","price_jpy"],
        keep="last"
    ).reset_index(drop=True)

# 型の安全化（念のため）
base["price_jpy"] = pd.to_numeric(base["price_jpy"], errors="coerce")

# ===== y1: 土地(=tochi) と中古(chuko) の比較 =====
tochi = base[base["source_group"]=="tochi"].copy()

avg_city_town = (
    tochi.groupby(["ward_city","city_town"])["price_jpy"].mean()
         .reset_index().rename(columns={"price_jpy":"tochi_avg_ct"})
)
avg_ward = (
    tochi.groupby(["ward_city"])["price_jpy"].mean()
         .reset_index().rename(columns={"price_jpy":"tochi_avg_w"})
)

# m:1 で左結合（city_town→ward フォールバック）
base = base.merge(avg_city_town, on=["ward_city","city_town"], how="left", validate="m:1")
base = base.merge(avg_ward, on=["ward_city"], how="left", validate="m:1")

base["tochi_ref_price"] = base["tochi_avg_ct"].combine_first(base["tochi_avg_w"])

mask_chuko = base["source_group"].isin(["kodate_model_chuko","mansion_model_chuko"])

base.loc[mask_chuko, "ratio1"] = base.loc[mask_chuko, "tochi_ref_price"] / base.loc[mask_chuko, "price_jpy"]
base.loc[mask_chuko, "y1"] = (base.loc[mask_chuko, "ratio1"] > 1.5).astype("Int64")
# 参照が無い個所は y1=NA
base.loc[mask_chuko & base["tochi_ref_price"].isna(), "y1"] = pd.NA

# ===== y2: 中古 ↔ 新築 の比較 =====
shin_mansion = base[base["source_group"]=="mansion_model_shin"]
shin_kodate  = base[base["source_group"]=="kodate_model_shin"]

# 新築マンションの平均
shin_mansion_city = (
    shin_mansion.groupby(["ward_city","city_town"])["price_jpy"].mean()
               .reset_index().rename(columns={"price_jpy":"shin_mansion_avg_ct"})
)
shin_mansion_ward = (
    shin_mansion.groupby(["ward_city"])["price_jpy"].mean()
               .reset_index().rename(columns={"price_jpy":"shin_mansion_avg_w"})
)

# 新築戸建の平均
shin_kodate_city = (
    shin_kodate.groupby(["ward_city","city_town"])["price_jpy"].mean()
              .reset_index().rename(columns={"price_jpy":"shin_kodate_avg_ct"})
)
shin_kodate_ward = (
    shin_kodate.groupby(["ward_city"])["price_jpy"].mean()
              .reset_index().rename(columns={"price_jpy":"shin_kodate_avg_w"})
)

# m:1 でマップ（平均表はキー一意）
base = base.merge(shin_mansion_city, on=["ward_city","city_town"], how="left", validate="m:1")
base = base.merge(shin_mansion_ward, on=["ward_city"],            how="left", validate="m:1")
base = base.merge(shin_kodate_city,  on=["ward_city","city_town"], how="left", validate="m:1")
base = base.merge(shin_kodate_ward,  on=["ward_city"],            how="left", validate="m:1")

mask_m_chuko = base["source_group"]=="mansion_model_chuko"
mask_k_chuko = base["source_group"]=="kodate_model_chuko"

# 参照価格の決定（中古マンション→新築マンション、 中古戸建→新築戸建）
base.loc[mask_m_chuko, "shin_ref_price"] = (
    base.loc[mask_m_chuko, "shin_mansion_avg_ct"].combine_first(base.loc[mask_m_chuko, "shin_mansion_avg_w"])
)
base.loc[mask_k_chuko, "shin_ref_price"] = (
    base.loc[mask_k_chuko, "shin_kodate_avg_ct"].combine_first(base.loc[mask_k_chuko, "shin_kodate_avg_w"])
)

# y2 判定（2倍以上）
base.loc[mask_chuko, "ratio2"] = base.loc[mask_chuko, "shin_ref_price"] / base.loc[mask_chuko, "price_jpy"]
base.loc[mask_chuko, "y2"] = (base.loc[mask_chuko, "ratio2"] >= 2).astype("Int64")
base.loc[mask_chuko & base["shin_ref_price"].isna(), "y2"] = pd.NA

# （任意）作業カラムの整理：残したくない中間列を消す
cols_drop = [
    "tochi_avg_ct","tochi_avg_w",
    "shin_mansion_avg_ct","shin_mansion_avg_w",
    "shin_kodate_avg_ct","shin_kodate_avg_w",
    "ratio1","ratio2","tochi_ref_price","shin_ref_price"
]
keep_cols = [c for c in base.columns if c not in cols_drop]
out = base[keep_cols].copy()

# === 出力 ===
out.to_csv("normalized_listings_with_y.csv", index=False, encoding="utf-8-sig")
print(f"Wrote normalized_listings_with_y.csv  shape={out.shape}")
