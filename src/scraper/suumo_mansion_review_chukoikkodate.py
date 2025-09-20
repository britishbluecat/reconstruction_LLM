# scraper/suumo_mansion_review_chukoikkodate.py
import os, re, time, random
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

# 正規表現・正規化
RE_PRICE  = re.compile(r"(\d[\d,]*\s*万(?:円)?|\d[\d,]*\s*円)")
RE_WALK   = re.compile(r"徒歩\s*\d+\s*分")
RE_NUM_FIRST = re.compile(r"(\d+(?:\.\d+)?)")
RE_PAREN = re.compile(r"[（(].*?[）)]")

ZEN2HAN = str.maketrans({
    "０":"0","１":"1","２":"2","３":"3","４":"4",
    "５":"5","６":"6","７":"7","８":"8","９":"9",
    "．":".","，":",","－":"-","〜":"~","／":"/","　":" "
})

def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

def _pick_dd_by_dt(container, label: str):
    # dl/dt/dd 構造と dottable-fix の両方に対応
    for dl in container.find_all("dl"):
        dt = dl.find("dt")
        if dt and dt.get_text("", strip=True) == label:
            dd = dl.find("dd")
            if dd: return dd
    for table in container.find_all("table", class_="dottable-fix"):
        for dl in table.find_all("dl"):
            dt = dl.find("dt")
            if dt and dt.get_text("", strip=True) == label:
                dd = dl.find("dd")
                if dd: return dd
    return None

def _normalize_sqm(dd_el) -> str:
    if dd_el is None: return ""
    t = dd_el.get_text("", strip=True).translate(ZEN2HAN)
    t = RE_PAREN.sub("", t)
    t = (t.replace("m²","㎡").replace("m^2","㎡").replace("m2","㎡").replace("平方メートル","㎡"))
    m = RE_NUM_FIRST.search(t)
    if not m: return ""
    return f"{m.group(1)}㎡"

def _find_detail_link_in(block) -> str:
    for a in block.select("a[href]"):
        href = a.get("href","")
        # 詳細は /bukken/ or /nc_ が多い
        if "/bukken/" in href or "/nc_" in href:
            return href
    return ""

def parse_listing_card(card, base_url: str) -> dict:
    # 共通フィールド
    price = location = walk = layout = built_ym = ""
    land_sqm = bldg_sqm = ""

    # 価格・所在地・徒歩・面積などは「情報表」ブロックを総なめに
    for blk in card.select(".dottable-line, .dottable, .property_unit-data, .cassette"):
        if not price:
            m = RE_PRICE.search(_text(blk))
            if m: price = m.group(1)

        if not location:
            dd = _pick_dd_by_dt(blk, "所在地")
            if dd: location = dd.get_text(" ", strip=True)

        if not walk:
            # 「沿線・駅」または「交通」
            for lbl in ("沿線・駅","交通"):
                dd = _pick_dd_by_dt(blk, lbl)
                if dd:
                    m = RE_WALK.search(dd.get_text(" ", strip=True))
                    walk = m.group(0) if m else walk

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
            for label in ("築年月", "建築年月", "築年数（築年月）", "築年数"):
                dd = _pick_dd_by_dt(blk, label)
                if dd:
                    built_ym = RE_PAREN.sub("", dd.get_text("", strip=True))
                    break

    href = _find_detail_link_in(card)
    detail_url = urljoin(base_url, href) if href else base_url
    title = location or price or base_url

    meta = {
        "物件名": title,
        "所在地": location,
        "駅徒歩": walk,
        "土地面積": land_sqm,
        "建物面積": bldg_sqm,
        "exclusive_area_sqm": bldg_sqm,  # ★ 戸建ては建物面積をそのまま専有面積相当として出す
        "間取り": layout,
        "販売価格": price,
        "built_ym": built_ym,
        "source": base_url,
    }
    return {"url": detail_url, "title": title, "body": "", "meta": meta}

def parse_list_page(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    cont = soup.select_one("#js-bukkenList") or soup
    items = []

    # 1) 新UIカード
    cards = cont.select("div.property_unit, li.property_unit")
    # 2) フォールバック：カセット型
    if not cards:
        cards = cont.select("div.dottable.dottable--cassette, div.cassette")

    for idx, card in enumerate(cards, start=1):
        try:
            it = parse_listing_card(card, base_url)
            if (not it.get("url")) or (it["url"] == base_url):
                it["url"] = f"{base_url}#item={idx}"
            items.append(it)
        except Exception:
            continue
    return items

def parse_detail_page(html: str, base_url: str) -> dict:
    # 一覧で拾い切れる想定。詳細は保険的に実装
    soup = BeautifulSoup(html, "lxml")
    page_title = _text(soup.select_one("title"))
    location = walk = land_sqm = bldg_sqm = layout = price = built_ym = ""

    for blk in soup.select(".dottable-line, .dottable, .property_unit-data"):
        if not price:
            m = RE_PRICE.search(_text(blk))
            if m: price = m.group(1)
        if not location:
            dd = _pick_dd_by_dt(blk, "所在地")
            if dd: location = dd.get_text(" ", strip=True)
        if not walk:
            for lbl in ("沿線・駅","交通"):
                dd = _pick_dd_by_dt(blk, lbl)
                if dd:
                    m = RE_WALK.search(dd.get_text(" ", strip=True))
                    walk = m.group(0) if m else walk
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
            for label in ("築年月","建築年月","築年数（築年月）","築年数"):
                dd = _pick_dd_by_dt(blk, label)
                if dd:
                    built_ym = RE_PAREN.sub("", dd.get_text("", strip=True))
                    break

    title = location or page_title or base_url
    meta = {
        "物件名": title,
        "所在地": location,
        "駅徒歩": walk,
        "土地面積": land_sqm,
        "建物面積": bldg_sqm,
        "exclusive_area_sqm": bldg_sqm,
        "間取り": layout,
        "販売価格": price,
        "built_ym": built_ym,
        "source": base_url,
    }
    return {"url": base_url, "title": title, "body": "", "meta": meta}

# ---- ページネーション（/chukoikkodate/?page=N）
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
            max_no = max(max_no, n)
        except ValueError:
            continue
    return max_no

def crawl_list(url: str) -> list[dict]:
    results = []
    polite_sleep()
    r = fetch(url); r.raise_for_status()
    first_html = r.text
    soup = BeautifulSoup(first_html, "lxml")
    max_page = _max_page_in_pagination(soup)

    results.extend(parse_list_page(first_html, url))
    for pn in range(2, max_page + 1):
        page_url = _build_url_with_page(url, pn)
        polite_sleep()
        rr = fetch(page_url)
        if rr.status_code != 200:
            break
        results.extend(parse_list_page(rr.text, page_url))
    return results
