# scraper/suumo_mansion_review_tochi.py
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

def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

# ---- 正規化補助 ----
RE_PAREN = re.compile(r"[（(].*?[）)]")
RE_FIRST_NUM = re.compile(r"(\d+(?:\.\d+)?)")
RE_WALK = re.compile(r"徒歩\s*\d+\s*分")

ZEN2HAN = str.maketrans({
    "０":"0","１":"1","２":"2","３":"3","４":"4",
    "５":"5","６":"6","７":"7","８":"8","９":"9",
    "．":".","，":",","－":"-","〜":"~","／":"/","　":" "
})

# scraper/suumo_mansion_review_tochi.py

# 置き換え：parse_detail_page（今の簡易版 → 詳細パース版）
# 置き換え：parse_detail_page
def parse_detail_page(html: str, base_url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # 基本
    location = _pick_value_by_label(soup, "所在地")
    walk_raw = _pick_value_by_label(soup, "沿線・駅")
    m = RE_WALK.search(walk_raw or "")
    station_walk = m.group(0) if m else ""

    # 価格（文字列のまま。下流で normalize）
    price_text = ""
    price_raw = _pick_value_by_label(soup, "販売価格")
    if price_raw:
        price_text = price_raw.translate(ZEN2HAN)
    else:
        strong = soup.select_one(".price, .dottable-value, .cassette_price-accent")
        price_text = _text(strong).translate(ZEN2HAN) if strong else ""

    # 土地面積
    land_raw = _pick_value_by_label(soup, "土地面積")
    land_sqm = _normalize_sqm_text(land_raw)

    # 坪単価
    tsubo_raw = _pick_value_by_label(soup, "坪単価")
    tsubo_price = tsubo_raw.translate(ZEN2HAN) if tsubo_raw else ""

    # 建ぺい率・容積率
    ky_raw = _pick_value_by_label(soup, "建ぺい率・容積率").translate(ZEN2HAN)
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*％", ky_raw)
    kenpei = nums[0] if len(nums) >= 1 else ""
    youseki = nums[1] if len(nums) >= 2 else ""

    title = location or _text(soup.select_one("title")) or base_url

    meta = {
        "物件名": title,
        "所在地": location,
        "駅徒歩": station_walk,
        "土地面積": land_sqm,
        "坪単価": tsubo_price,
        "建ぺい率": kenpei,
        "容積率": youseki,
        "間取り": "",
        "販売価格": price_text,
        # ↓↓↓ 追加（統一キーを両方入れる）
        "exclusive_area_sqm": land_sqm,
        "専有面積": land_sqm,
        "source": base_url,
    }
    return {"url": base_url, "title": title, "body": "", "meta": meta}




def _normalize_sqm_text(t: str) -> str:
    """
    '41.77m2', '102.32m2～102.73m2', '54.75m2（16.56坪）' などから
    最初の数値だけを抜き 'xx.xx㎡' で返す
    """
    if not t:
        return ""
    t = t.translate(ZEN2HAN)
    t = RE_PAREN.sub("", t)
    t = (t.replace("m²", "㎡")
           .replace("m^2", "㎡")
           .replace("m2", "㎡")
           .replace("平方メートル", "㎡"))
    m = RE_FIRST_NUM.search(t)
    return f"{m.group(1)}㎡" if m else ""

def _first_price_text(el) -> str:
    # カード内の販売価格（文字列のまま返す：クレンジングは後段）
    if not el:
        return ""
    return _text(el).translate(ZEN2HAN)

def _pick_dd_by_dt(container, label: str):
    # :scopeを使わずカード内だけ探索
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

def _find_detail_link_in(card) -> str | None:
    # できるだけ「詳細らしい」ものを優先
    for sel in [
        "a.property_unit-title[href]",
        "a.cassette-main_title[href]",
        "a[href*='/bukken/']",
        "a[href^='/']",
        "a[href]"
    ]:
        a = card.select_one(sel)
        if a and a.get("href"):
            return a.get("href")
    return None

def parse_listing_card(card, base_url: str) -> dict:
    # 基本
    location = station_walk = ""
    price_text = land_sqm = tsubo_price = kenpei = youseki = ""

    # 価格
    price_el = None
    # 優先: 明示の価格ラベル
    dd = _pick_dd_by_dt(card, "販売価格")
    if dd:
        price_el = dd.select_one(".dottable-value") or dd
    else:
        # フォールバック：カード内にあるテキストから推定
        price_el = card.select_one(".dottable-value")
    price_text = _first_price_text(price_el)

    # 所在地 / 沿線・駅
    dd = _pick_dd_by_dt(card, "所在地")
    if dd:
        location = dd.get_text(" ", strip=True)
    dd = _pick_dd_by_dt(card, "沿線・駅")
    if dd:
        m = RE_WALK.search(dd.get_text(" ", strip=True))
        station_walk = m.group(0) if m else ""

    # 土地面積 / 坪単価（最初だけ）
    dd = _pick_dd_by_dt(card, "土地面積")
    if dd:
        land_sqm = _normalize_sqm_text(dd.get_text("", strip=True))
    dd = _pick_dd_by_dt(card, "坪単価")
    if dd:
        # 例: '469.4万円／坪' → 原文のまま（後段で解析したければ別途）
        tsubo_price = dd.get_text("", strip=True).translate(ZEN2HAN)

    # 建ぺい率・容積率
    dd = _pick_dd_by_dt(card, "建ぺい率・容積率")
    if dd:
        raw = dd.get_text("", strip=True).translate(ZEN2HAN)
        # 例: '建ペい率：60％、容積率：160％'
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*％", raw)
        if nums:
            if len(nums) >= 1: kenpei = nums[0]
            if len(nums) >= 2: youseki = nums[1]

    # 詳細URL
    href = _find_detail_link_in(card)
    detail_url = urljoin(base_url, href) if href else None

    # タイトル=所在地（要件）
    title = location or base_url

    meta = {
        "物件名": title,
        "所在地": location,
        "駅徒歩": station_walk,
        "土地面積": land_sqm,              # 既存
        "坪単価": tsubo_price,
        "建ぺい率": kenpei,
        "容積率": youseki,
        "間取り": "",
        "販売価格": price_text,
        # ↓↓↓ 追加（統一キーを両方入れる）
        "exclusive_area_sqm": land_sqm,     # CSVの固定列を埋める
        "専有面積": land_sqm,               # storage がこのキーを見る場合の保険
        "source": base_url,
    }
    return {"url": detail_url or base_url, "title": title, "body": "", "meta": meta}

def parse_list_page(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    cont = soup.select_one("#js-bukkenList") or soup

    # カード群を列挙（tochi でも property_unit / cassette 両方に備える）
    cards = cont.select("div.property_unit, li.property_unit")
    if not cards:
        cards = cont.select("div.dottable.dottable--cassette, div.cassette")

    items = []
    for idx, card in enumerate(cards, start=1):
        try:
            it = parse_listing_card(card, base_url)
            # 詳細URLが取れない or base_urlのままなら擬似URLで一意化（上書き防止）
            if (not it.get("url")) or (it["url"] == base_url):
                it["url"] = f"{base_url}#item={idx}"
            items.append(it)
        except Exception:
            continue
    return items

# ---- ページネーション（?page=） ----
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

# scraper/suumo_mansion_review_tochi.py

def _pick_value_by_label(container, label: str):
    """
    カード/ページ内から、'label' に対応する値テキストを返す。
    サポート:
      - <dl><dt>label</dt><dd>value</dd>
      - <table><tr><th>label</th><td>value</td></tr>
    フォールバック:
      - 'label' を含む要素を見つけて、近傍の次要素から値を推定
    """
    norm = lambda s: re.sub(r"\s+", "", s or "")
    target = norm(label)

    # 1) dl/dt/dd パターン
    for dl in container.find_all("dl"):
        dt = dl.find("dt")
        if not dt: 
            continue
        if target in norm(dt.get_text("", strip=True)):
            dd = dl.find("dd")
            if dd:
                return dd.get_text(" ", strip=True)

    # 2) table th/td パターン
    for table in container.find_all("table"):
        for tr in table.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if th and td and (target in norm(th.get_text("", strip=True))):
                return td.get_text(" ", strip=True)

            # th/td でなくても dl が入ってるケースもあるので保険
            dd = tr.find("dd")
            dt = tr.find("dt")
            if dt and dd and (target in norm(dt.get_text("", strip=True))):
                return dd.get_text(" ", strip=True)

    # 3) フォールバック: 'label' を含むノードの直後兄弟を拾う
    cand = container.find(string=lambda x: isinstance(x, str) and target in norm(x))
    if cand and cand.parent:
        # 次の兄弟に dd / td があれば優先
        sib = cand.parent.find_next_sibling()
        if sib:
            dd = sib.find("dd")
            if dd: 
                return dd.get_text(" ", strip=True)
            td = sib.find("td")
            if td:
                return td.get_text(" ", strip=True)
        # なければ親の次の兄弟
        par_sib = cand.parent.find_next_sibling()
        if par_sib:
            td = par_sib.find("td")
            if td:
                return td.get_text(" ", strip=True)

    return ""
