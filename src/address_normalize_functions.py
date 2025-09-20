#address_normalize_functions.py

import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
list_csv_files = sorted(DATA_DIR.glob("*.csv"))

TARGET_COLUMNS = [
    "property_name",
    "location",
    "station_walk",
    "built_ym",
    "exclusive_area_sqm",
    "layout",
    "price_jpy",
]

ZEN2HAN_NUM = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "ー": "-", "－": "-"
})

def load_selected_columns():

    all_dfs = []

    for csv_file in list_csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            # 念のため fallback
            df = pd.read_csv(csv_file, encoding="utf-8")

        # 必要カラムが存在する部分だけ抽出
        cols_available = [c for c in TARGET_COLUMNS if c in df.columns]
        df_selected = df[cols_available].copy()

        # 存在しないカラムはNoneで埋める
        for c in TARGET_COLUMNS:
            if c not in df_selected.columns:
                df_selected[c] = None

        # カラム順を統一
        df_selected = df_selected[TARGET_COLUMNS]

        all_dfs.append(df_selected)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=TARGET_COLUMNS)

def normalize_location_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "location" in df.columns:
        df["location"] = df["location"].astype(str).str.translate(ZEN2HAN_NUM)
    return df

def clean_property_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "property_name" not in df.columns:
        return df

    def clean_name(name: str):
        if not isinstance(name, str):
            return name

        text = name.strip()

        # …があればそれ以降を削除
        text = text.split("…")[0]

        # 営業用記号など削除
        symbols = [
            "◆", "◇", "□", "■", "～", "【", "】", "(", ")", "（", "）", "♪", "万円", "可"
        ]
        for s in symbols:
            text = text.replace(s, "")

        # 括弧内のテキストを削除（全角・半角両方）
        text = re.sub(r"（.*?）", "", text)
        text = re.sub(r"\(.*?\)", "", text)

        # 数字+年（例: 2003年, 1998年）を削除
        text = re.sub(r"\d{4}年", "", text)

        # 余分な空白を整理（全角スペースも対象）
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text if text else None  # 空になったら None 扱い

    df["property_name"] = df["property_name"].map(clean_name)

    return df



def normalize_exclusive_area(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "exclusive_area_sqm" in df.columns:
        df["exclusive_area_sqm"] = (
            df["exclusive_area_sqm"]
            .astype(str)
            .str.replace("㎡", "", regex=False)
            .str.strip()
        )
        df["exclusive_area_sqm"] = pd.to_numeric(
            df["exclusive_area_sqm"], errors="coerce"
        )
    return df



# 範囲/列挙の区切り記号（全角半角いろいろ）
_SEP_SPLIT_RE = re.compile(r"[～〜\-−ー~・／/]")

def normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def to_yen_first(part: str):
        """
        '1億8880万円' → 188800000
        '8480万円'   → 84800000
        マッチしなければ None
        """
        if not isinstance(part, str):
            return None
        t = part.strip().replace(",", "").replace("円", "").replace("万円", "万")
        m = re.match(r"^\s*(?:(\d+)億)?\s*(\d+)万", t)
        if m:
            oku = int(m.group(1)) if m.group(1) else 0
            man = int(m.group(2)) if m.group(2) else 0
            return oku * 100_000_000 + man * 10_000
        # まれに「xxxxx円」だけの表記にフォールバック
        m2 = re.match(r"^\s*(\d+)\s*円", t)
        if m2:
            return int(m2.group(1))
        return None

    def parse_price(val: str):
        if not isinstance(val, str):
            return None
        # 範囲・列挙は「左側（最初）」を採用
        first = _SEP_SPLIT_RE.split(val)[0]
        return to_yen_first(first)

    if "price_jpy" in df.columns:
        df["price_jpy"] = df["price_jpy"].astype(str).map(parse_price).astype("Int64")

        # 0円はDROP（NaNは保持）
        mask_keep = df["price_jpy"].isna() | (df["price_jpy"] != 0)
        df = df[mask_keep].reset_index(drop=True)

    return df





def normalize_station_walk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "station_walk" in df.columns:
        df["station_walk"] = (
            df["station_walk"]
            .astype(str)
            .str.replace("徒歩", "", regex=False)
            .str.replace("分", "", regex=False)
            .str.strip()
        )
        # 数字に変換
        df["station_walk"] = pd.to_numeric(df["station_walk"], errors="coerce").astype("Int64")
    return df

def normalize_built_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "built_ym" in df.columns:
        df["built_ym"] = (
            df["built_ym"]
            .astype(str)
            .str.extract(r"(\d{4})")  # 西暦4桁だけ抽出
        )
        df["built_ym"] = pd.to_numeric(df["built_ym"], errors="coerce").astype("Int64")
    return df

import re
import pandas as pd

def normalize_layout(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "layout" not in df.columns:
        return df

    def parse_layout(val: str):
        out = {"LDK": 0, "1R": 0, "K": 0, "S": 0, "nando": 0}
        if not isinstance(val, str):
            return out
        t = val.strip()

        if "ワンルーム" in t:
            out["1R"] = 1

        m_ldk = re.search(r"(\d+)\s*LDK", t)
        if m_ldk:
            out["LDK"] = int(m_ldk.group(1))

        # LDK 内の K はカウントしない / 純粋な「2K」などのみ拾う
        if "LDK" not in t:
            m_k = re.search(r"(\d+)\s*K", t)
            if m_k:
                out["K"] = int(m_k.group(1))

        # S は「2S」「+S」いずれも対応（数字なしは1）
        for s in re.findall(r"(\d*)\s*S", t):
            out["S"] += int(s) if s.isdigit() else 1

        if "納戸" in t:
            out["nando"] = 1

        return out

    parsed = df["layout"].map(parse_layout)

    # ★ index を維持したまま DataFrame 化
    layout_df = pd.DataFrame(parsed.tolist(), index=df.index).fillna(0).astype("Int64")

    # ★ index を使って安全に結合
    df = df.join(layout_df)

    return df

import re
import pandas as pd

WARD_CITY_RE = re.compile(r"^東京都(?P<ward_city>.+?(?:区|市|村))(?P<rest>.*)")

def add_ward_city_and_city_town(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "location" not in df.columns:
        df["ward_city"] = None
        df["city_town"] = None
        return df

    def extract_parts(loc: str):
        ward_city = None
        city_town = None
        if not isinstance(loc, str):
            return ward_city, city_town

        text = loc.strip()
        m = WARD_CITY_RE.match(text)
        if not m:
            return ward_city, city_town

        ward_city = m.group("ward_city")
        rest = (m.group("rest") or "").strip()

        # --- city_town 抽出 ---
        # 1. 「〇〇町」で始まる場合
        m_town = re.match(r"^([^\d\s\-－（(　]+?町)", rest)
        if m_town:
            name = m_town.group(1)
            if not ("ヶ" in name and name.endswith("ヶ谷")):
                city_town = name
            return ward_city, city_town

        # 2. 「〇〇丁目」など → 「〇〇」を取る
        m_chome = re.match(r"^([^\d\s\-－（(　]+?)丁目", rest)
        if m_chome:
            name = m_chome.group(1)
            if not ("ヶ" in name and name.endswith("ヶ谷")):
                city_town = name
            return ward_city, city_town

        # 3. 数字に続く前の漢字ブロックを取る（八丁堀4, 四谷5 など）
        m_block = re.match(r"^([^\d\s\-－（(　]+)", rest)
        if m_block:
            name = m_block.group(1)
            if not ("ヶ" in name and name.endswith("ヶ谷")):
                city_town = name
            return ward_city, city_town

        return ward_city, city_town

    pairs = df["location"].astype(str).map(extract_parts)
    df[["ward_city", "city_town"]] = pd.DataFrame(list(pairs), index=df.index)

    return df




# --- ここから追記 ------------------------------------------------------------
from typing import Iterable

# data/<subdir>/*.csv を 1 本に結合して data/<subdir>.csv を吐き出す
SUBDIRS = [
    "kodate_model_chuko",
    "kodate_model_shin",
    "mansion_model_chuko",
    "mansion_model_shin",
    "tochi",
]

def _read_csv_safe(p: Path) -> pd.DataFrame:
    """
    すべて comma-delimited 前提。
    文字コードは UTF-8 (BOM 付き/なし) を優先し、失敗時は fallback。
    """
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    # 最後の手段
    return pd.read_csv(p, encoding="cp932", errors="ignore")

def build_folder_level_csvs() -> None:
    """
    data/<subdir>/*.csv を結合して data/<subdir>.csv を保存する。
    - 区切りはカンマ（前提どおり）
    - カラムはそのまま（統合時に union）
    - 同一行の重複は簡易に drop_duplicates で除去
    """
    for sub in SUBDIRS:
        src_dir = DATA_DIR / sub
        if not src_dir.exists():
            continue

        csv_files = sorted(src_dir.glob("*.csv"))
        if not csv_files:
            # 空でも、空の CSV を吐いておくと後工程が楽
            out = DATA_DIR / f"{sub}.csv"
            if not out.exists():
                pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
            continue

        dfs: list[pd.DataFrame] = []
        for f in csv_files:
            try:
                df = _read_csv_safe(f)
                dfs.append(df)
            except Exception:
                # 壊れたファイルはスキップ
                continue

        if not dfs:
            out = DATA_DIR / f"{sub}.csv"
            if not out.exists():
                pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
            continue

        merged = pd.concat(dfs, ignore_index=True)
        # 代表的な重複除去（URL があれば優先。なければ全カラムで）
        if "url" in merged.columns:
            merged = merged.drop_duplicates(subset=["url"], keep="last")
        else:
            merged = merged.drop_duplicates(keep="last")

        out = DATA_DIR / f"{sub}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, index=False, encoding="utf-8-sig")

def _list_folder_level_csvs() -> list[Path]:
    """ build_folder_level_csvs 実行後に出来上がる 5 本だけを読む """
    paths = [(DATA_DIR / f"{sub}.csv") for sub in SUBDIRS]
    return [p for p in paths if p.exists()]
# --- ここまで追記 ------------------------------------------------------------

# === ここから追記: 住所→バウンディングボックス作成ユーティリティ ==================
import pandas as pd
import numpy as np
import re
from pathlib import Path

# 丁目の正規化：末尾の「（漢数字/全角/半角）丁目」を安全に除去
# 例: "麹町六丁目"→"麹町" / "永田町一丁目"→"永田町" / "六番町"→変化なし
_RE_TRAILING_CHOME = re.compile(r"(.*?)([一二三四五六七八九十〇零百０-９0-9]+)?丁目$")

def strip_chome(oaza_chome: str) -> str:
    if not isinstance(oaza_chome, str):
        return oaza_chome
    oaza_chome = oaza_chome.strip()
    m = _RE_TRAILING_CHOME.match(oaza_chome)
    if m:
        base = m.group(1).strip()
        return base if base else oaza_chome
    return oaza_chome

from typing import Union, Optional

def _to_decimal_tail(val: Union[float, str]) -> Optional[float]:
    """
    東京前提で 緯度:35.xxxxx 経度:139.xxxxx の小数部のみを使う。
    例: 139.732787 -> 0.732787 / 35.687614 -> 0.687614
    """
    try:
        f = float(val)
    except Exception:
        return None
    frac = abs(f) - int(abs(f))
    return frac

def _to_decimal_tail(val: Union[float, str]) -> Optional[float]:
    """
    東京前提で 緯度:35.xxxxx 経度:139.xxxxx の小数部のみを使う。
    例: 139.732787 -> 0.732787 / 35.687614 -> 0.687614
    """
    try:
        f = float(val)
    except Exception:
        return None
    frac = abs(f) - int(abs(f))
    return frac

def _round_for_5km(frac: Optional[float]) -> Optional[float]:
    """5km程度で十分 → 小数第2位に丸め（約1.1km）。粗くしたければ round(frac, 1)。"""
    if frac is None:
        return None
    return round(frac, 2)

def build_geo_bbox_from_coords(coordinates_csv: str | Path) -> pd.DataFrame:
    """
    coordinates.csv を読み、（市区町村名, 大字_丁目名<丁目除去>）単位で
    経度・緯度の小数部（139,35を除いた fractional）から
    [x_min,x_max,y_min,y_max] を作る。
    出力: columns = ['ward_city','city_town','x_min','x_max','y_min','y_max']
    """
    p = Path(coordinates_csv)
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            gdf = pd.read_csv(p, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"failed to read {coordinates_csv}")

    # 列名あわせ
    # 市区町村名 → ward_city
    # 大字_丁目名 → city_town（ただし末尾の「…丁目」を落とす）
    if "市区町村名" not in gdf.columns or "大字_丁目名" not in gdf.columns:
        raise ValueError("coordinates.csv に '市区町村名' と '大字_丁目名' が必要です。")

    gdf["ward_city"] = gdf["市区町村名"].astype(str).str.strip()
    gdf["city_town_raw"] = gdf["大字_丁目名"].astype(str).str.strip()
    gdf["city_town"] = gdf["city_town_raw"].map(strip_chome)

    # 緯度/経度の小数部。欠損時は X/Y 座標の fallback も可（今回は不要ならスキップ）
    if "緯度" not in gdf.columns or "経度" not in gdf.columns:
        raise ValueError("coordinates.csv に '緯度' と '経度' が必要です。")

    gdf["x_frac"] = gdf["経度"].map(_to_decimal_tail).map(_round_for_5km)  # 経度→x
    gdf["y_frac"] = gdf["緯度"].map(_to_decimal_tail).map(_round_for_5km)  # 緯度→y

    # BBox（max-min）を作成
    bbox = (
        gdf.groupby(["ward_city", "city_town"], dropna=False)[["x_frac", "y_frac"]]
          .agg(x_min=("x_frac", "min"),
               x_max=("x_frac", "max"),
               y_min=("y_frac", "min"),
               y_max=("y_frac", "max"))
          .reset_index()
    )

    return bbox[["ward_city","city_town","x_min","x_max","y_min","y_max"]]
# === ここまで追記 =============================================================
