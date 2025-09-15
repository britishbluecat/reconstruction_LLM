# storage.py — CSVストレージ（prefix付きファイル名 & BOM付きUTF-8）
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv
from typing import Dict, List, Optional

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def _make_csv_path() -> Path:
    """現在時刻を使って prefix付きファイルパスを生成"""
    ts = datetime.now().strftime("%Y%m%d%H%M")
    return DATA_DIR / f"{ts}-suumo_reviews.csv"

# 実行ごとに新規ファイルを使う
CSV_PATH = _make_csv_path()

# 残す列
COLUMNS = [
    "url",
    "fetched_at",
    "status",
    "location",
    "station_walk",
    "built_ym",
    "exclusive_area_sqm",
    "layout",
    "price_jpy",
]

# ---------- 基本I/O ----------

def _ensure_csv():
    """CSVがなければ新規作成"""
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()

def _load_all() -> List[Dict[str, str]]:
    _ensure_csv()
    rows: List[Dict[str, str]] = []
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {col: r.get(col, "") for col in COLUMNS}
            if (row.get("url") or "").strip():  # url空は除外
                rows.append(row)
    return rows

def _write_all(rows: List[Dict[str, str]]):
    tmp = CSV_PATH.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            row = {col: r.get(col, "") for col in COLUMNS}
            if (row.get("url") or "").strip():
                w.writerow(row)
    tmp.replace(CSV_PATH)

def _now_iso_utc() -> str:
    return datetime.utcnow().isoformat()

# ---------- meta 展開 ----------

def _extract_meta(meta: dict) -> Dict[str, str]:
    def pick(*keys, default=""):
        for k in keys:
            if k in meta and meta[k] is not None:
                v = str(meta[k]).strip()
                if v:
                    return v
        return default

    return {
        "location": pick("location", "所在地", "住所"),
        "station_walk": pick("station_walk", "駅徒歩", "沿線・駅"),
        "built_ym": pick("built_ym", "築年月", "築年"),
        "exclusive_area_sqm": pick("exclusive_area_sqm", "専有面積", "面積"),
        "layout": pick("layout", "間取り"),
        "price_jpy": pick("price_jpy", "販売価格", "価格", "price"),
    }

# ---------- 公開API ----------

def get_db():
    _ensure_csv()
    return str(CSV_PATH)

def upsert_review(conn, url: str, status: int, title: str, body: str, meta: dict):
    if not url or not str(url).strip():
        return

    rows = _load_all()
    idx: Optional[int] = None
    for i, r in enumerate(rows):
        if r.get("url") == url:
            idx = i
            break

    meta_ex = _extract_meta(meta or {})

    rec = {
        "url": url.strip(),
        "fetched_at": _now_iso_utc(),
        "status": str(status),
        "location": meta_ex["location"],
        "station_walk": meta_ex["station_walk"],
        "built_ym": meta_ex["built_ym"],
        "exclusive_area_sqm": meta_ex["exclusive_area_sqm"],
        "layout": meta_ex["layout"],
        "price_jpy": meta_ex["price_jpy"],
    }

    if idx is None:
        rows.append(rec)
    else:
        rows[idx] = rec

    _write_all(rows)

def read_all_reviews() -> List[Dict[str, str]]:
    return _load_all()

def get_by_url(query_url: str) -> Optional[Dict[str, str]]:
    for r in _load_all():
        if r.get("url") == query_url:
            return r
    return None
