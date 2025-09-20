# scraper/suumo_mansion_review_tochi_test.py
import sys
from pprint import pprint

from suumo_mansion_review_tochi import (
    fetch,
    parse_list_page,
    parse_detail_page,
    _pick_value_by_label,      # デバッグ用
    _normalize_sqm_text,       # デバッグ用
)

LIST_URL = "https://suumo.jp/tochi/tokyo/sc_bunkyo/"

def test_list_page(url: str, show=5):
    print(f"\n[LIST TEST] GET {url}")
    r = fetch(url)
    print("status:", r.status_code)
    items = parse_list_page(r.text, url)
    print("items:", len(items))
    for i, it in enumerate(items[:show], start=1):
        print(f"\n--- LIST ITEM {i} ---")
        print("url:", it["url"])
        print("title:", it["title"])
        print("meta keys:", list(it["meta"].keys()))
        pprint(it["meta"], width=120)

    return items

def test_detail_page(detail_url: str):
    print(f"\n[DETAIL TEST] GET {detail_url}")
    r = fetch(detail_url)
    print("status:", r.status_code)

    # もし “土地面積” が拾えない場合、近傍の生HTMLをダンプして原因を掴む
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "lxml")
    around = _pick_value_by_label(soup, "土地面積")
    if not around:
        # フォールバックで “敷地面積” も試す（サイトの表記ブレに備える）
        around = _pick_value_by_label(soup, "敷地面積")
    print("RAW land area text:", repr(around))
    print("NORMALIZED:", _normalize_sqm_text(around))

    it = parse_detail_page(r.text, detail_url)
    print("\n--- DETAIL META ---")
    pprint(it["meta"], width=120)

if __name__ == "__main__":
    # 1) 一覧ページをパースして meta を直接確認
    items = test_list_page(LIST_URL, show=5)

    # 2) 一覧で得た1件目のURLで詳細ページをパースして meta を確認
    if items:
        # 一覧で詳細URLが base_url のまま（=擬似URL付与 or 取れてない）なら、そのままでも parse_detail_page は動く
        test_detail_page(items[0]["url"])
    else:
        print("No items parsed from list page.")
        sys.exit(1)
