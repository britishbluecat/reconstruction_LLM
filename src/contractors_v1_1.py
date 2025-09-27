# -*- coding: utf-8 -*-
"""
SUUMO リフォーム業者＆施工事例 スクレイパ（東京 / 12ページ対応 / 口コミは /review 参照）

変更点（全面修正）
- 一覧のベースURLを /remodel/tokyo/ （?pn=2..12）に変更
- 一覧ページの会社カード抽出を堅牢化（会社名・会社URL・会社ID）
- 8つの評価は各社の /review/?ar=030 から取得（一覧で出ないケース対策）
- 施工事例は /jitsurei/?ar=030 を巡回、費用/建物タイプ/築年数/間取り(Before/After) 抽出
- enrich（既存CSVのrating空欄後埋め）と crawl_all（フル取得）を提供

出力:
  data/contractors_tokyo.csv
  data/case_studies_tokyo.csv

注意:
  - 利用規約/robots.txtの遵守、過剰な高頻度アクセスの回避
  - HTML構造変更に備えてフェイルセーフ（テキスト正規表現）を併用
"""

from __future__ import annotations
import csv
import os
import re
import time
import random
from typing import List, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

BASE = "https://suumo.jp"
# 一覧はユーザー指定のこれ（12ページ）
LIST_PATH = "/remodel/tokyo/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; reconstruction-LLM-bot/1.0)",
    "Accept-Language": "ja,en;q=0.8",
}

RATING_KEYS = [
    "仕上がり",
    "価格の納得感",
    "マナー・態度",
    "作業中の配慮",
    "説明のわかりやすさ",
    "対応の速さ",
    "施工の段取り",
    "約束・時間の厳守",
]

OUT_DIR = "data"
CONTRACTORS_CSV = os.path.join(OUT_DIR, "contractors_tokyo.csv")
CASESTUDY_CSV = os.path.join(OUT_DIR, "case_studies_tokyo.csv")


# -------------------- 基本ユーティリティ --------------------

def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s


def _sleep(a: float = 0.8, b: float = 1.8) -> None:
    time.sleep(random.uniform(a, b))


def _extract_contractor_id(url: str) -> str | None:
    # /remodel/ki_0003872/ → 0003872
    m = re.search(r"/remodel/ki_(\d+)/", url)
    return m.group(1) if m else None


def _iter_list_pages(total_pages: int = 12):
    """東京一覧ページの12ページを順に返す。"""
    yield LIST_PATH  # page 1
    for i in range(2, total_pages + 1):
        yield f"{LIST_PATH}?pn={i}"


# -------------------- 一覧ページ（会社抽出） --------------------

def parse_list_page(html: str) -> List[Dict]:
    """/remodel/tokyo/ の1ページから会社カードを抽出。
    会社名リンク a[href*='/remodel/ki_'] を起点に、重複を除いて収集。
    """
    soup = BeautifulSoup(html, "lxml")

    # a要素から会社リンク候補を収集
    anchors = [a for a in soup.select("a[href*='/remodel/ki_']") if a.get_text(strip=True)]

    out: List[Dict] = []
    seen: set = set()
    for a in anchors:
        href = a.get("href") or ""
        if "/remodel/ki_" not in href:
            continue
        url = urljoin(BASE, href)
        cid = _extract_contractor_id(url)
        if not cid:
            continue
        name = a.get_text(strip=True)
        key = (cid, url)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "contractor_id": cid,
            "contractor_name": name,
            "contractor_url": url,
        })

    return out


def fetch_contractors_from_list(total_pages: int = 12) -> List[Dict]:
    sess = _build_session()
    all_rows: List[Dict] = []
    for rel in _iter_list_pages(total_pages):
        url = urljoin(BASE, rel)
        resp = sess.get(url, timeout=25)
        if resp.status_code != 200:
            # ページが尽きた/エラー
            break
        rows = parse_list_page(resp.text)
        # 2ページ目以降でゼロ件ならページ終端と判断
        if not rows and rel.endswith(tuple(f"?pn={i}" for i in range(2, total_pages + 1))):
            break
        all_rows.extend(rows)
        _sleep()
    return dedup_by_id(all_rows)


def dedup_by_id(rows: List[Dict]) -> List[Dict]:
    out, seen = [], set()
    for r in rows:
        k = r.get("contractor_id")
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def save_contractors_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "contractor_id",
        "contractor_name",
        "contractor_url",
        *[f"rating_{k}" for k in RATING_KEYS],
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------- 口コミ（/review）で8指標を取得 --------------------

def fetch_ratings_from_review(session: requests.Session, contractor_url: str) -> Dict:
    parsed = urlparse(contractor_url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if not base.endswith("/"):
        base += "/"
    review_url = urljoin(base, "review/?ar=030")

    resp = session.get(review_url, timeout=25)
    if resp.status_code != 200:
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    text = soup.get_text("\n", strip=True)

    ratings = {}
    for label in RATING_KEYS:
        m = re.search(label + r"\s*：\s*([0-9.]+)", text)
        if m:
            ratings[f"rating_{label}"] = m.group(1)
    return ratings


# -------------------- 施工事例（/jitsurei） --------------------

CASE_FIELDS = ["cost", "building_type", "age_years", "layout_before", "layout_after"]


def parse_case_list(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    cases: List[Dict] = []

    # 事例カードっぽいブロックを総当たりで探索
    for blk in soup.select("article, li, div"):
        t = blk.get_text("\n", strip=True)
        if all(k in t for k in ["費用", "建物タイプ", "築年数", "間取り"]):
            m_cost = re.search(r"費用\s*([0-9,\.]+)\s*万円", t)
            m_bld = re.search(r"建物タイプ\s*([^\n]+)", t)
            m_age = re.search(r"築年数\s*([0-9]+)\s*年", t)
            # 表記ゆれ（コロン/全角スペース/矢印）を吸収
            m_lay = re.search(r"間取り\s*Before[:：]\s*([^\n→>]+)\s*[→>-]\s*After[:：]\s*([^\n]+)", t)

            if any([m_cost, m_bld, m_age, m_lay]):
                cases.append({
                    "cost": (m_cost.group(1) + "万円") if m_cost else None,
                    "building_type": m_bld.group(1).strip() if m_bld else None,
                    "age_years": (m_age.group(1) + "年") if m_age else None,
                    "layout_before": m_lay.group(1).strip() if m_lay else None,
                    "layout_after": m_lay.group(2).strip() if m_lay else None,
                })

    # フォールバック（ページ全体からラベル束を拾う）
    if not cases:
        t_all = soup.get_text("\n", strip=True)
        for seg in re.split(r"(?=費用\s*[0-9,\.]+万円)", t_all):
            if all(k in seg for k in ["建物タイプ", "築年数", "間取り"]):
                m_cost = re.search(r"費用\s*([0-9,\.]+)\s*万円", seg)
                m_bld = re.search(r"建物タイプ\s*([^\n]+)", seg)
                m_age = re.search(r"築年数\s*([0-9]+)\s*年", seg)
                m_lay = re.search(r"間取り\s*Before[:：]\s*([^\n→>]+)\s*[→>-]\s*After[:：]\s*([^\n]+)", seg)
                cases.append({
                    "cost": (m_cost.group(1) + "万円") if m_cost else None,
                    "building_type": m_bld.group(1).strip() if m_bld else None,
                    "age_years": (m_age.group(1) + "年") if m_age else None,
                    "layout_before": m_lay.group(1).strip() if m_lay else None,
                    "layout_after": m_lay.group(2).strip() if m_lay else None,
                })

    return cases


def fetch_case_studies_for_contractor(
    session: requests.Session,
    contractor_url: str,
    contractor_id: str,
    max_pages: int = 12,
) -> List[Dict]:
    parsed = urlparse(contractor_url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if not base.endswith("/"):
        base += "/"
    jitsurei = urljoin(base, "jitsurei/?ar=030")

    out: List[Dict] = []
    for i in range(1, max_pages + 1):
        url = jitsurei if i == 1 else f"{jitsurei}&pn={i}"
        resp = session.get(url, timeout=25)
        if resp.status_code != 200:
            break
        page_cases = parse_case_list(resp.text)
        if not page_cases and i > 1:
            break
        for c in page_cases:
            c["contractor_id"] = contractor_id
            c["case_url_page"] = url
        out.extend(page_cases)
        _sleep()
    return out


def save_case_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["contractor_id", "case_url_page", *CASE_FIELDS]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------- フロー --------------------

def enrich_ratings_from_csv(csv_in: str = CONTRACTORS_CSV, csv_out: str | None = None) -> None:
    """既存 contractors_tokyo.csv の rating_* 空欄を /review で後埋め。"""
    sess = _build_session()
    with open(csv_in, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if all(r.get(f"rating_{k}") for k in RATING_KEYS):
            continue
        url = r.get("contractor_url") or ""
        if not url:
            continue
        try:
            ratings = fetch_ratings_from_review(sess, url)
            r.update(ratings)
        except Exception as e:
            print(f"[WARN] enrich failed for {r.get('contractor_id')}: {e}")
        _sleep(1.0, 2.0)

    save_contractors_csv(csv_out or csv_in, rows)
    print(f"[OK] enriched ratings -> {csv_out or csv_in}")


def crawl_all(save_cases: bool = True) -> None:
    """フルクロール: 東京一覧→会社一覧抽出→/review でratings→/jitsurei で事例。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    sess = _build_session()

    contractors = fetch_contractors_from_list(total_pages=12)

    # 各社の口コミ評価を付与
    for r in contractors:
        try:
            r.update(fetch_ratings_from_review(sess, r["contractor_url"]))
        except Exception as e:
            print(f"[WARN] rating failed for {r['contractor_id']}: {e}")
        _sleep()

    save_contractors_csv(CONTRACTORS_CSV, contractors)
    print(f"[OK] contractors: {len(contractors)} -> {CONTRACTORS_CSV}")

    if save_cases:
        all_cases: List[Dict] = []
        for r in contractors:
            try:
                all_cases += fetch_case_studies_for_contractor(
                    sess, r["contractor_url"], r["contractor_id"], max_pages=12
                )
            except Exception as e:
                print(f"[WARN] cases failed for {r['contractor_id']}: {e}")
            _sleep(1.2, 2.0)

        save_case_csv(CASESTUDY_CSV, all_cases)
        print(f"[OK] case studies: {len(all_cases)} -> {CASESTUDY_CSV}")


if __name__ == "__main__":
    # 1) 既存CSVを後埋めする場合:
    # enrich_ratings_from_csv("data/contractors_tokyo.csv")

    # 2) フルクロールする場合:
    crawl_all(save_cases=True)
