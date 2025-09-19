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

# --------- 正規表現（補助） ----------
RE_PRICE  = re.compile(r"(\d[\d,]*\s*万(?:円)?|\d[\d,]*\s*円)")
RE_SQM_TX = re.compile(r"(\d+(?:\.\d+)?)")
RE_WALK   = re.compile(r"徒歩\s*\d+\s*分")
RE_BUILT  = re.compile(r"(\d{4})年\s*(\d{1,2})月")

# 追加：ラベル候補
NAME_LABELS = ["物件名", "マンション名", "建物名", "名称"]

def _pick_dd_by_dt_labels(container, labels):
    """ dt のラベル候補リストから最初に見つかった dd を返す """
    for lab in labels:
        dd = _pick_dd_by_dt_label(container, lab)
        if dd:
            return dd
    return None


def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

def _normalize_sqm(dd_el) -> str:
    """dd要素内の m<sup>2</sup> を ㎡ に。数字+㎡ を返す（例：33.65㎡）。括弧など補記は残さない。"""
    if dd_el is None:
        return ""
    for sup in dd_el.find_all("sup"):
        sup.replace_with("^" + sup.get_text("", strip=True))
    raw = dd_el.get_text("", strip=True)
    raw = raw.replace("m^2", "㎡").replace("m2", "㎡")
    m = RE_SQM_TX.search(raw)
    return f"{m.group(1)}㎡" if m else raw

def _pick_dd_by_dt_label(container, label: str):
    """ dottable-line 内の <dl><dt>label</dt><dd>...</dd></dl> から dd を返す（最初の一致）。 """
    # 直接の dl
    for dl in container.select(":scope dl"):
        dt = dl.find("dt")
        if dt and dt.get_text("", strip=True) == label:
            return dl.find("dd")
    # テーブル入れ子（dottable-fix）にも対応
    for dl in container.select(":scope table.dottable-fix dl"):
        dt = dl.find("dt")
        if dt and dt.get_text("", strip=True) == label:
            return dl.find("dd")
    return None

# ---------------------------
# 1) 旧来カード: .property_unit
# ---------------------------
def parse_listing_card_property_unit(card: BeautifulSoup, base_url: str) -> dict:
    a = card.select_one(".property_unit-header h2 a, h2 a")
    title = _text(a)
    href = a["href"].strip() if a and a.has_attr("href") else ""
    detail_url = urljoin(base_url, href) if href else base_url

    price = _text(card.select_one(".price, .property_unit-price, .dottable-value.price"))
    if not price:
        m = RE_PRICE.search(_text(card))
        price = m.group(1) if m else ""

    location = layout = sqm = built = walk = ""
    name = ""

    for blk in card.select(".dottable-line"):
        if not name:
            dd = _pick_dd_by_dt_labels(blk, NAME_LABELS)
            if dd:
                name = dd.get_text(" ", strip=True)

        if not location:
            dd = _pick_dd_by_dt_label(blk, "所在地")
            if dd: location = dd.get_text(" ", strip=True)
        if not sqm:
            dd = _pick_dd_by_dt_label(blk, "専有面積")
            if dd: sqm = _normalize_sqm(dd)
        if not layout:
            dd = _pick_dd_by_dt_label(blk, "間取り")
            if dd: layout = dd.get_text("", strip=True)
        if not built:
            dd = _pick_dd_by_dt_label(blk, "築年月")
            if dd:
                built = dd.get_text("", strip=True)
            else:
                m = RE_BUILT.search(_text(blk))
                if m: built = f"{m.group(1)}年{m.group(2)}月"
        if not walk:
            dd = _pick_dd_by_dt_label(blk, "沿線・駅")
            if dd:
                m = RE_WALK.search(dd.get_text(" ", strip=True))
                walk = m.group(0) if m else ""

    # フォールバック：見出しテキストを物件名として使う
    if not name:
        name = title

    if not location:
        raw = _text(card)
        m = re.search(r"(東京都[^\s　、，]+|.+区|.+市)", raw)
        if m: location = m.group(1)

    meta = {
        "物件名": name,
        "所在地": location,
        "駅徒歩": walk,
        "築年月": built,
        "専有面積": sqm,
        "間取り": layout,
        "販売価格": price,
        "source": base_url,
    }
    return {"url": detail_url, "title": name or title, "body": "", "meta": meta}


# -------------------------------------
# 2) 新型カード: .dottable.dottable--cassette
#    （あなたが貼ってくれたHTMLに対応）
# -------------------------------------
def _find_detail_link_in(block) -> str:
    # 詳細ページへのリンクは a[href*="/nc_"] や a[href*="/bukken/"] が混在し得るので網羅的に拾う
    for a in block.select("a[href]"):
        href = a.get("href", "")
        if "/nc_" in href or "/bukken/" in href:
            return href
    return ""

def parse_listing_card_cassette(cassette: BeautifulSoup, base_url: str) -> dict:
    name = ""
    for blk in cassette.select(".dottable-line"):
        dd = _pick_dd_by_dt_labels(blk, NAME_LABELS)
        if dd:
            name = dd.get_text(" ", strip=True)
            break

    # 価格
    price = ""
    for blk in cassette.select(".dottable-line"):
        dd = _pick_dd_by_dt_label(blk, "販売価格")
        if dd:
            price = _text(dd.select_one(".dottable-value")) or dd.get_text("", strip=True)
            break
    if not price:
        m = RE_PRICE.search(_text(cassette))
        price = m.group(1) if m else ""

    location = layout = sqm = built = walk = ""
    for blk in cassette.select(".dottable-line"):
        if not location:
            dd = _pick_dd_by_dt_label(blk, "所在地")
            if dd: location = dd.get_text(" ", strip=True)
        if not walk:
            dd = _pick_dd_by_dt_label(blk, "沿線・駅")
            if dd:
                m = RE_WALK.search(dd.get_text(" ", strip=True))
                walk = m.group(0) if m else ""
        if not sqm:
            dd = _pick_dd_by_dt_label(blk, "専有面積")
            if dd: sqm = _normalize_sqm(dd)
        if not layout:
            dd = _pick_dd_by_dt_label(blk, "間取り")
            if dd: layout = dd.get_text("", strip=True)
        if not built:
            dd = _pick_dd_by_dt_label(blk, "築年月")
            if dd: built = dd.get_text("", strip=True)

    href = _find_detail_link_in(cassette)
    detail_url = urljoin(base_url, href) if href else base_url

    # フォールバック：タイトルタグなど
    title = name
    if not title:
        # 一部カードは見出しが別DOMにあるため、念のため全体からの最初のh2も試す
        h2 = cassette.select_one("h2, .property_unit-header h2")
        title = _text(h2)

    meta = {
        "物件名": name or title,
        "所在地": location,
        "駅徒歩": walk,
        "築年月": built,
        "専有面積": sqm,
        "間取り": layout,
        "販売価格": price,
        "source": base_url,
    }
    return {"url": detail_url, "title": name or title, "body": "", "meta": meta}


# ---------------------------
# 一覧ページパース（両型対応）
# ---------------------------
def parse_list_page(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    cont = soup.select_one("#js-bukkenList") or soup

    items = []

    # 旧来カード
    for card in cont.select("div.property_unit"):
        try:
            items.append(parse_listing_card_property_unit(card, base_url))
        except Exception:
            continue

    # 新型カセット
    for cassette in cont.select("div.dottable.dottable--cassette, div.cassette"):
        try:
            items.append(parse_listing_card_cassette(cassette, base_url))
        except Exception:
            continue

    return items

# ---------------------------
# 詳細ページ（念のため据え置き）
# ---------------------------
def parse_detail_page(html: str, base_url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    page_title = _text(soup.select_one("title"))

    location = layout = sqm = built = walk = price = ""
    name = ""

    for blk in soup.select(".dottable-line"):
        if not name:
            dd = _pick_dd_by_dt_labels(blk, NAME_LABELS)
            if dd: name = dd.get_text(" ", strip=True)
        if not location:
            dd = _pick_dd_by_dt_label(blk, "所在地")
            if dd: location = dd.get_text(" ", strip=True)
        if not sqm:
            dd = _pick_dd_by_dt_label(blk, "専有面積")
            if dd: sqm = _normalize_sqm(dd)
        if not layout:
            dd = _pick_dd_by_dt_label(blk, "間取り")
            if dd: layout = dd.get_text("", strip=True)
        if not built:
            dd = _pick_dd_by_dt_label(blk, "築年月")
            if dd: built = dd.get_text("", strip=True)
        if not walk:
            dd = _pick_dd_by_dt_label(blk, "沿線・駅")
            if dd:
                m = RE_WALK.search(dd.get_text(" ", strip=True))
                walk = m.group(0) if m else ""
        if not price:
            m = RE_PRICE.search(_text(blk))
            if m: price = m.group(1)
    if not price:
        m = RE_PRICE.search(_text(soup))
        if m: price = m.group(1)

    title = name or page_title
    meta = {
        "物件名": name or title,
        "所在地": location,
        "駅徒歩": walk,
        "築年月": built,
        "専有面積": sqm,
        "間取り": layout,
        "販売価格": price,
        "source": base_url,
    }
    return {"url": base_url, "title": title, "body": "", "meta": meta}


# ---------------------------
# ページネーション対応
# ---------------------------
def _build_url_with_page(url: str, page_no: int) -> str:
    """ &pn= の値を差し替えてURLを返す（無ければ付与） """
    u = urlparse(url)
    qs = parse_qs(u.query)
    qs["pn"] = [str(page_no)]
    new_query = urlencode(qs, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))

def _max_page_in_pagination(soup: BeautifulSoup) -> int:
    """ ページャがあれば最大ページ番号を推定。無ければ 1。 """
    max_no = 1
    for a in soup.select("a[href*='pn=']"):
        txt = _text(a)
        try:
            n = int(txt)
            if n > max_no:
                max_no = n
        except ValueError:
            continue
    return max_no

def crawl_list(url: str) -> list[dict]:
    """ 一覧URL（pc=100対応）から全ページを巡回して物件カードを回収 """
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

# ---------------------------
# 使い方例
# ---------------------------
if __name__ == "__main__":
    # url_list.txt から読み込んで全件クロール → 件数とサンプルを表示
    url_list_path = os.getenv("URL_LIST", "url_list.txt")
    urls = []
    if os.path.exists(url_list_path):
        with open(url_list_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    all_items = []
    for u in urls:
        try:
            items = crawl_list(u)
            all_items.extend(items)
            print(f"[OK] {u} -> {len(items)} items")
        except Exception as e:
            print(f"[ERR] {u}: {e}")

    print(f"TOTAL: {len(all_items)} items")
    if all_items:
        print(all_items[0])
