# scraper/suumo_mansion_review_ikkodate.py
import os
import re
import time
import random
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import requests
from bs4 import BeautifulSoup

USER_AGENT = os.getenv("USER_AGENT", "ReconstructionLLM/1.0")
REQUEST_INTERVAL = float(os.getenv("REQUEST_INTERVAL", "1.0"))
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"}

def polite_sleep():
    time.sleep(REQUEST_INTERVAL + random.uniform(0, 0.5))

def fetch(url: str) -> requests.Response:
    return requests.get(url, headers=HEADERS, timeout=20)

# 正規表現（流用/拡張）
RE_PRICE  = re.compile(r"(\d[\d,]*\s*万(?:円)?|\d[\d,]*\s*円)")
RE_SQM_TX = re.compile(r"(\d+(?:\.\d+)?)")
RE_WALK   = re.compile(r"徒歩\s*\d+\s*分")

def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

RE_NUM_FIRST = re.compile(r"(\d+(?:\.\d+)?)")
RE_PAREN = re.compile(r"[（(].*?[）)]")  # 全角/半角の括弧と中身

# 全角→半角（数字・小数点・記号の一部）
ZEN2HAN = str.maketrans({
    "０":"0","１":"1","２":"2","３":"3","４":"4",
    "５":"5","６":"6","７":"7","８":"8","９":"9",
    "．":".","，":",","－":"-","〜":"~","／":"/","　":" "
})



def _normalize_sqm(dd_el) -> str:
    """
    dd内の面積テキストから、最初に現れる数値だけを抽出し、「{数値}㎡」で返す。
    括弧注記（（登記）, (実測), (壁芯), (xx坪) 等）は削除。
    単位の揺れ（m2, m^2, m², 平方メートル）は ㎡ に統一。
    例:
      '102.32m2～102.73m2'         -> '102.32㎡'
      '54.75m2（16.56坪）'         -> '54.75㎡'
      '90.61m2（実測）'            -> '90.61㎡'
      '96.04m2・96.87m2'           -> '96.04㎡'
      '107.07m2（32.38坪）、うち１階車庫10.24m2' -> '107.07㎡'
      '63.06m2（登記）'            -> '63.06㎡'
    """
    if dd_el is None:
        return ""
    # 生テキスト
    t = dd_el.get_text("", strip=True)

    # 全角→半角（先にやる）
    t = t.translate(ZEN2HAN)

    # 括弧注記をすべて除去（例: (登記), (実測), (32.38坪)）
    t = RE_PAREN.sub("", t)

    # 単位ゆれを ㎡ に統一
    t = (t.replace("m²", "㎡")
           .replace("m^2", "㎡")
           .replace("m2", "㎡")
           .replace("平方メートル", "㎡"))

    # 最初に現れる小数含む数値を一つだけ取得
    m = RE_NUM_FIRST.search(t)
    if not m:
        return ""
    num = m.group(1)
    return f"{num}㎡"


def _pick_dd_by_dt(container, label: str):
    # 直下/入れ子問わず、この container の中だけを探索（:scope は使わない）
    for dl in container.find_all("dl"):
        dt = dl.find("dt")
        if dt and dt.get_text("", strip=True) == label:
            dd = dl.find("dd")
            if dd:
                return dd
    # テーブル（dottable-fix）内
    for table in container.find_all("table", class_="dottable-fix"):
        for dl in table.find_all("dl"):
            dt = dl.find("dt")
            if dt and dt.get_text("", strip=True) == label:
                dd = dl.find("dd")
                if dd:
                    return dd
    return None


def _find_detail_link_in(block) -> str:
    # 戸建てでも一覧→詳細のリンクは /bukken/ や /nc_ が混在し得る
    for a in block.select("a[href]"):
        href = a.get("href", "")
        if "/nc_" in href or "/bukken/" in href:
            return href
    return ""

def parse_listing_card_cassette(cassette, base_url: str) -> dict:
    price = ""
    for blk in cassette.select(".dottable-line"):
        dd = _pick_dd_by_dt(blk, "販売価格")
        if dd:
            price = dd.get_text("", strip=True)
            break

    location = walk = layout = ""
    land_sqm = bldg_sqm = ""
    built_ym = ""

    for blk in cassette.select(".dottable-line"):
        if not location:
            dd = _pick_dd_by_dt(blk, "所在地")
            if dd: location = dd.get_text(" ", strip=True)

        if not walk:
            dd = _pick_dd_by_dt(blk, "沿線・駅")
            if dd:
                m = RE_WALK.search(dd.get_text(" ", strip=True))
                walk = m.group(0) if m else ""

        if not land_sqm:
            dd = _pick_dd_by_dt(blk, "土地面積")
            if dd: land_sqm = _normalize_sqm(dd)

        if not bldg_sqm:
            dd = _pick_dd_by_dt(blk, "建物面積")
            if dd: bldg_sqm = _normalize_sqm(dd)

        if not layout:
            dd = _pick_dd_by_dt(blk, "間取り")
            if dd: layout = dd.get_text("", strip=True)

        if not built_ym:
            # ラベルの揺れに対応
            for label in ("築年月", "建築年月", "築年数（築年月）", "築年数"):
                dd = _pick_dd_by_dt(blk, label)
                if dd:
                    # 注記を落としてそのまま保存（後段で年だけ抽出する想定）
                    bym = RE_PAREN.sub("", dd.get_text("", strip=True))
                    built_ym = bym
                    break

    href = _find_detail_link_in(cassette)
    detail_url = urljoin(base_url, href) if href else base_url
    title = location or price or base_url

    meta = {
        "物件名": title,
        "所在地": location,
        "駅徒歩": walk,
        "土地面積": land_sqm,           # 例: '131.93㎡'
        "建物面積": bldg_sqm,           # 例: '113.03㎡' → 下流で exclusive_area_sqm 相当
        "exclusive_area_sqm": bldg_sqm, # 下流処理との整合のために同値も入れておく
        "間取り": layout,
        "販売価格": price,
        "built_ym": built_ym,           # 例: '2003年3月' 等（後段で年抽出）
        "source": base_url,
    }
    return {"url": detail_url, "title": title, "body": "", "meta": meta}

'''
def parse_list_page(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")

    # 一覧コンテナ候補
    cont = soup.select_one("#js-bukkenList") or soup

    items = []

    # まずは property_unit を優先（戸建てでよく使われる）
    cards = cont.select("div.property_unit, li.property_unit")
    if not cards:
        # フォールバック: カセット型
        cards = cont.select("div.dottable.dottable--cassette, div.cassette")

    for card in cards:
        try:
            items.append(parse_listing_card_cassette(card, base_url))
        except Exception:
            continue
    return items
'''

# scraper/suumo_mansion_review_ikkodate.py

def parse_list_page(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")

    # 一覧コンテナ候補
    cont = soup.select_one("#js-bukkenList") or soup

    items = []

    # まずは property_unit を優先（戸建てでよく使われる）
    cards = cont.select("div.property_unit, li.property_unit")
    if not cards:
        # フォールバック: カセット型
        cards = cont.select("div.dottable.dottable--cassette, div.cassette")

    # ★ enumerate でカードごとに一意な idx を付ける
    for idx, card in enumerate(cards, start=1):
        try:
            it = parse_listing_card_cassette(card, base_url)

            # ★ 詳細URLが取れず base_url のまま/空なら、擬似URLで一意化して上書き防止
            if (not it.get("url")) or (it["url"] == base_url):
                it["url"] = f"{base_url}#item={idx}"

            items.append(it)
        except Exception:
            continue

    return items


def parse_detail_page(html: str, base_url: str) -> dict:
    # 一覧で十分拾えるため、念のため据え置き実装
    soup = BeautifulSoup(html, "lxml")
    page_title = _text(soup.select_one("title"))

    location = walk = land_sqm = bldg_sqm = layout = price = ""
    for blk in soup.select(".dottable-line"):
        if not price:
            m = RE_PRICE.search(_text(blk))
            if m: price = m.group(1)
        if not location:
            dd = _pick_dd_by_dt(blk, "所在地")
            if dd: location = dd.get_text(" ", strip=True)
        if not walk:
            dd = _pick_dd_by_dt(blk, "沿線・駅")
            if dd:
                m = RE_WALK.search(dd.get_text(" ", strip=True))
                walk = m.group(0) if m else ""
        if not land_sqm:
            dd = _pick_dd_by_dt(blk, "土地面積")
            if dd: land_sqm = _normalize_sqm(dd)
        if not bldg_sqm:
            dd = _pick_dd_by_dt(blk, "建物面積")
            if dd: bldg_sqm = _normalize_sqm(dd)
        if not layout:
            dd = _pick_dd_by_dt(blk, "間取り")
            if dd: layout = dd.get_text("", strip=True)

    title = location or page_title or base_url
    meta = {
        "物件名": title,
        "所在地": location,
        "駅徒歩": walk,
        "土地面積": land_sqm,
        "建物面積": bldg_sqm,
        "間取り": layout,
        "販売価格": price,
        "source": base_url,
    }
    return {"url": base_url, "title": title, "body": "", "meta": meta}

# ---- ページネーション（戸建ては page= が基本） ----
def _build_url_with_page(url: str, page_no: int) -> str:
    u = urlparse(url)
    qs = parse_qs(u.query)
    qs["page"] = [str(page_no)]
    new_query = urlencode(qs, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))

def _max_page_in_pagination(soup: BeautifulSoup) -> int:
    max_no = 1
    for a in soup.select("a[href*='page=']"):
        txt = _text(a)
        try:
            n = int(txt)
            if n > max_no:
                max_no = n
        except ValueError:
            continue
    return max_no

def crawl_list(url: str) -> list[dict]:
    results = []
    polite_sleep()
    r = fetch(url)
    r.raise_for_status()
    first_html = r.text
    soup = BeautifulSoup(first_html, "lxml")
    max_page = _max_page_in_pagination(soup)

    # 1ページ目
    results.extend(parse_list_page(first_html, url))

    # 2ページ目以降
    for pn in range(2, max_page + 1):
        page_url = _build_url_with_page(url, pn)
        polite_sleep()
        rr = fetch(page_url)
        if rr.status_code != 200:
            break
        results.extend(parse_list_page(rr.text, page_url))

    return results
