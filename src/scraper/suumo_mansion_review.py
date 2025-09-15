import os
import re
import time
import random
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENT = os.getenv("USER_AGENT", "ReconstructionLLM/1.0")
REQUEST_INTERVAL = float(os.getenv("REQUEST_INTERVAL", "1.0"))
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"}

def polite_sleep():
    time.sleep(REQUEST_INTERVAL + random.uniform(0, 0.5))

def fetch(url: str) -> requests.Response:
    return requests.get(url, headers=HEADERS, timeout=20)

# --------- 正規表現（補助） ----------
RE_PRICE  = re.compile(r"(\d[\d,]*\s*万(?:円)?|\d[\d,]*\s*円)")
RE_SQM_TX = re.compile(r"(\d+(?:\.\d+)?)")
RE_WALK   = re.compile(r"徒歩\s*\d+\s*分")
RE_BUILT  = re.compile(r"(\d{4})年\s*(\d{1,2})月")

def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

def _normalize_sqm(dd_el) -> str:
    """dd要素内の m<sup>2</sup> を ㎡ に。数字+㎡ を返す（例：33.65㎡）。括弧など補記は残さない。"""
    if dd_el is None:
        return ""
    # supを展開して文字列化
    for sup in dd_el.find_all("sup"):
        sup.replace_with("^" + sup.get_text("", strip=True))
    raw = dd_el.get_text("", strip=True)
    # "m^2" → "㎡"
    raw = raw.replace("m^2", "㎡").replace("m2", "㎡")
    # 先頭の数値を抽出して "X㎡" に（見た目優先）
    m = RE_SQM_TX.search(raw)
    return f"{m.group(1)}㎡" if m else raw

def _pick_dd_by_dt_label(container, label: str):
    """
    dottable-line 内の <dl><dt>label</dt><dd>...</dd></dl> から dd を返す。
    複数あれば最初。
    """
    for dl in container.select("dl"):
        dt = dl.find("dt")
        if dt and dt.get_text("", strip=True) == label:
            return dl.find("dd")
    return None

def parse_listing_card(card: BeautifulSoup, base_url: str) -> dict:
    """
    一覧ページの property_unit 1件をパース。
    """
    # タイトル & 詳細リンク
    a = card.select_one(".property_unit-header h2 a")
    title = _text(a)
    href = a["href"].strip() if a and a.has_attr("href") else ""
    detail_url = urljoin(base_url, href) if href else base_url

    # 価格（カード上部の価格領域 or 生テキストから）
    price = _text(card.select_one(".price, .property_unit-price, .dottable-value.price"))
    if not price:
        m = RE_PRICE.search(_text(card))
        price = m.group(1) if m else ""

    # 住所/面積/間取り/築年月/徒歩 を dottable-line の dt/dd から取得
    # 1) 直下の dottable-line 群をスキャン
    #    例1: <dl><dt>専有面積</dt><dd>33.65m<sup>2</sup>（壁芯）</dd></dl>
    #    例2: <dl><dt>所在地</dt><dd>東京都中央区銀座８</dd></dl>
    dot_blocks = card.select(".dottable-line")

    location = ""
    layout = ""
    sqm = ""
    built = ""
    walk = ""

    for blk in dot_blocks:
        # 所在地
        if not location:
            dd = _pick_dd_by_dt_label(blk, "所在地")
            if dd:
                location = dd.get_text(" ", strip=True)

        # 専有面積
        if not sqm:
            dd = _pick_dd_by_dt_label(blk, "専有面積")
            if dd:
                sqm = _normalize_sqm(dd)

        # 間取り
        if not layout:
            dd = _pick_dd_by_dt_label(blk, "間取り")
            if dd:
                layout = dd.get_text("", strip=True)

        # 築年月
        if not built:
            dd = _pick_dd_by_dt_label(blk, "築年月")
            if dd:
                built = dd.get_text("", strip=True)
            else:
                # フォールバック：ブロックテキストから拾う
                m = RE_BUILT.search(_text(blk))
                if m:
                    built = f"{m.group(1)}年{m.group(2)}月"

        # 沿線・駅 → 徒歩
        if not walk:
            dd = _pick_dd_by_dt_label(blk, "沿線・駅")
            if dd:
                m = RE_WALK.search(dd.get_text(" ", strip=True))
                walk = m.group(0) if m else ""

    # 住所が未取得なら、カード全体テキストから都区市町村っぽい断片を拾う（弱フォールバック）
    if not location:
        # 最低限：タイトルやカード内に「東京都」「中央区」などがあれば拾う
        raw = _text(card)
        m = re.search(r"(東京都[^\s　、，]+|.+区|.+市)", raw)
        if m:
            location = m.group(1)

    meta = {
        "所在地": location,
        "駅徒歩": walk,
        "築年月": built,
        "専有面積": sqm,
        "間取り": layout,
        "販売価格": price,
        "source": base_url,
    }

    return {
        "url": detail_url,
        "title": title,
        "body": "",  # 一覧カードは本文なし
        "meta": meta,
    }

def parse_list_page(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    cont = soup.select_one("#js-bukkenList") or soup
    cards = cont.select("div.property_unit")
    items = []
    for card in cards:
        try:
            items.append(parse_listing_card(card, base_url))
        except Exception:
            continue
    return items

def parse_detail_page(html: str, base_url: str) -> dict:
    """
    詳細ページ（/nc_...）の簡易パーサ：dt/dd を優先的に見る
    """
    soup = BeautifulSoup(html, "lxml")
    title = _text(soup.select_one("title"))

    location = ""
    layout = ""
    sqm = ""
    built = ""
    walk = ""
    price = ""

    # dottable-line 優先
    for blk in soup.select(".dottable-line"):
        if not location:
            dd = _pick_dd_by_dt_label(blk, "所在地")
            if dd:
                location = dd.get_text(" ", strip=True)
        if not sqm:
            dd = _pick_dd_by_dt_label(blk, "専有面積")
            if dd:
                sqm = _normalize_sqm(dd)
        if not layout:
            dd = _pick_dd_by_dt_label(blk, "間取り")
            if dd:
                layout = dd.get_text("", strip=True)
        if not built:
            dd = _pick_dd_by_dt_label(blk, "築年月")
            if dd:
                built = dd.get_text("", strip=True)
        if not walk:
            dd = _pick_dd_by_dt_label(blk, "沿線・駅")
            if dd:
                m = RE_WALK.search(dd.get_text(" ", strip=True))
                walk = m.group(0) if m else ""
        if not price:
            # 詳細では価格が別領域にある事が多いので生テキストfallback
            m = RE_PRICE.search(_text(blk))
            if m:
                price = m.group(1)

    if not price:
        m = RE_PRICE.search(_text(soup))
        price = m.group(1) if m else ""

    meta = {
        "所在地": location,
        "駅徒歩": walk,
        "築年月": built,
        "専有面積": sqm,
        "間取り": layout,
        "販売価格": price,
        "source": base_url,
    }
    return {"url": base_url, "title": title, "body": "", "meta": meta}
