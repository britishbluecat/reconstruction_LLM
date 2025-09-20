# scraper/suumo_mansion_review_shinchiku.py
import os, re, time, random
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

# -------- helpers --------
RE_FIRST_NUM = re.compile(r"(\d+(?:\.\d+)?)")
RE_PAREN = re.compile(r"[（(].*?[）)]")
ZEN2HAN = str.maketrans({
    "０":"0","１":"1","２":"2","３":"3","４":"4",
    "５":"5","６":"6","７":"7","８":"8","９":"9",
    "．":".","，":",","－":"-","〜":"~","／":"/","　":" "
})

def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

def _first_area_sqm(text: str) -> str:
    """
    '48.82m2～70.58m2' → '48.82㎡'
    '96.04m2・96.87m2' → '96.04㎡'
    """
    if not text:
        return ""
    t = text.translate(ZEN2HAN)
    t = RE_PAREN.sub("", t)
    t = (t.replace("m²", "㎡")
           .replace("m^2", "㎡")
           .replace("m2", "㎡")
           .replace("平方メートル", "㎡"))
    m = RE_FIRST_NUM.search(t)
    return f"{m.group(1)}㎡" if m else ""

def _first_layout(text: str) -> str:
    """
    '2LDK・3LDK' → '2LDK'
    '1LDK+S（納戸）/ 3LDK' → '1LDK+S（納戸）'（最初優先）
    """
    if not text:
        return ""
    t = text.strip()
    # 最初に現れる代表的間取りパターン
    m = re.search(r"\d+\s*(?:LDK|DK|K)(?:\s*\+\s*\d*S(?:（納戸）)?)?", t)
    if m:
        return m.group(0).replace(" ", "")
    # ワンルーム
    m2 = re.search(r"ワンルーム", t)
    return m2.group(0) if m2 else ""

def _first_price_yen(text: str):
    """
    '1億600万円～1億8000万円／予定' → 106000000
    '8900万円／予定' → 89000000
    最初に出る金額だけを円に変換して返す（int）。取れなければ None
    """
    if not text:
        return None
    t = text.translate(ZEN2HAN)
    t = t.replace(",", "")
    # 先頭マッチだけ対象（範囲の左側）
    m = re.match(r"\s*(?:(\d+)億)?\s*(\d+)万", t)
    if m:
        oku = int(m.group(1)) if m.group(1) else 0
        man = int(m.group(2)) if m.group(2) else 0
        return oku * 100_000_000 + man * 10_000
    # 万が明示されず「円」だけのパターン（稀）
    m2 = re.match(r"\s*(\d+)円", t)
    if m2:
        return int(m2.group(1))
    return None

# ❶ 既存の parse_list_page をこの版に置き換え

def parse_list_page(html: str, base_url: str) -> list[dict]:
    """
    新築: 1ページ複数カードを列挙
    - カード容器: div.cassette-result_detail / li.cassette-result_item / div.property_unit
    - 詳細URL: カード内の a[href] からできる限り特定。なければ擬似URLで一意化。
    """
    soup = BeautifulSoup(html, "lxml")
    items = []

    cont = soup.select_one("#js-cassette") or soup

    # カード候補を列挙
    cards = cont.select("div.cassette-result_detail, li.cassette-result_item")
    if not cards:
        cards = cont.select("div.property_unit")

    for idx, card in enumerate(cards, start=1):
        try:
            # ------- 基本欄 -------
            location = station_walk = delivery = ""

            basic_items = card.select(".cassette_basic .cassette_basic-item")
            if not basic_items:
                basic_items = card.select(".cassette_basic *")

            for it in basic_items:
                title_el = it.select_one(".cassette_basic-title")
                value_el = it.select_one(".cassette_basic-value")
                if not title_el or not value_el:
                    continue
                title = _text(title_el)
                value = _text(value_el)
                if title == "所在地":
                    location = value
                elif title == "交通":
                    m = re.search(r"徒歩\s*\d+\s*分", value)
                    if m and not station_walk:
                        station_walk = m.group(0)
                elif title == "引渡時期":
                    delivery = value

            # ------- 価格/間取り/面積 -------
            price_raw_el = card.select_one(".cassette_price .cassette_price-accent") \
                           or card.select_one(".cassette_price-accent")
            price_raw = _text(price_raw_el)

            desc_el = card.select_one(".cassette_price .cassette_price-description") \
                   or card.select_one(".cassette_price-description")
            desc = _text(desc_el)

            layout_first = ""
            area_first = ""
            if desc and "/" in desc:
                left, right = [s.strip() for s in desc.split("/", 1)]
                layout_first = _first_layout(left)
                area_first = _first_area_sqm(right)
            else:
                layout_first = _first_layout(desc)
                area_first = _first_area_sqm(desc)

            price_yen = _first_price_yen(price_raw)
            price_text = price_raw

            # ------- 詳細URL（強化）-------
            # できるだけ「物件詳細」っぽいリンクを優先順位付きで探索
            detail_href = None
            for sel in [
                "a.cassette-main_title[href]",
                "a.cassette-main[href]",
                "a[href*='/ms/shinchiku/']",
                "a[href*='/bukken/']",
                "a[href^='/']",
                "a[href]"
            ]:
                a = card.select_one(sel)
                if a and a.get("href"):
                    detail_href = a.get("href")
                    break

            detail_url = urljoin(base_url, detail_href) if detail_href else None
            if not detail_url or detail_url == base_url:
                # ★擬似URLで一意化（上書き防止）
                detail_url = f"{base_url}#item={idx}"

            # ------- レコード化 -------
            title = location or price_raw or base_url
            meta = {
                "物件名": title,
                "所在地": location,
                "駅徒歩": station_walk,
                "間取り": layout_first,
                "専有面積": area_first,           # 'xx.xx㎡'
                "exclusive_area_sqm": area_first, # 下流正規化用
                "販売価格_raw": price_text,       # 範囲含む原文
                "販売価格": price_yen,           # 最初だけ円換算
                "引渡時期": delivery,
                "source": base_url,
            }
            items.append({"url": detail_url, "title": title, "body": "", "meta": meta})
        except Exception:
            continue

    return items



def parse_detail_page(html: str, base_url: str) -> dict:
    # 新築は一覧に十分情報があることが多い。必要なら詳細の追加項目をここで抽出。
    soup = BeautifulSoup(html, "lxml")
    page_title = _text(soup.select_one("title"))
    title = page_title or base_url
    return {"url": base_url, "title": title, "body": "", "meta": {"source": base_url}}
