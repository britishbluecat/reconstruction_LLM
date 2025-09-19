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

def normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def parse_price(val: str):
        if not isinstance(val, str):
            return None
        val = val.strip().replace(",", "").replace("円", "")

        # 「〇億〇〇〇万」/「〇億」/「〇〇〇万」に対応
        m = re.match(r"(?:(\d+)億)?(?:(\d+)万)?", val)
        if m:
            oku = int(m.group(1)) if m.group(1) else 0
            man = int(m.group(2)) if m.group(2) else 0
            return oku * 100_000_000 + man * 10_000
        return None

    if "price_jpy" in df.columns:
        df["price_jpy"] = df["price_jpy"].astype(str).map(parse_price).astype("Int64")

        # 0円は DROP。ただし NaN は残す（必要に応じてここで落としてもOK）
        mask_keep = df["price_jpy"].isna() | (df["price_jpy"] != 0)
        df = df[mask_keep].reset_index(drop=True)  # ★ここが重要

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





