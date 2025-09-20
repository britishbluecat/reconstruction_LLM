# pipeline_scrape_suumo.py
import os
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from storage import get_db, upsert_review

from scraper.suumo_mansion_review import (
    fetch as fetch_ms,
    parse_list_page as parse_list_ms,
    parse_detail_page as parse_detail_ms,
    polite_sleep as polite_sleep_ms,
)  # :contentReference[oaicite:0]{index=0}

from scraper.suumo_mansion_review_tochi import (
    fetch as fetch_tochi,
    parse_list_page as parse_list_tochi,
    parse_detail_page as parse_detail_tochi,
    polite_sleep as polite_sleep_tochi,
)

from scraper.suumo_mansion_review_ikkodate import (
    fetch as fetch_ik,
    parse_list_page as parse_list_ik,
    parse_detail_page as parse_detail_ik,
    polite_sleep as polite_sleep_ik,
)

from scraper.suumo_mansion_review_chukoikkodate import (
    fetch as fetch_chik,
    parse_list_page as parse_list_chik,
    parse_detail_page as parse_detail_chik,
    polite_sleep as polite_sleep_chik,
)

from scraper.suumo_mansion_review_shinchiku import (
    fetch as fetch_new,
    parse_list_page as parse_list_new,
    parse_detail_page as parse_detail_new,
    polite_sleep as polite_sleep_new,
)

# 置き換え：pick_scraper

def pick_scraper(url: str):
    path = (urlparse(url).path or "")
    qs = parse_qs(urlparse(url).query)
    bs = (qs.get("bs", [""])[0] or "").strip()

    # 土地
    if ("/tochi/" in path) or (bs == "030"):
        return ("tochi", fetch_tochi, parse_list_tochi, parse_detail_tochi, polite_sleep_tochi)

    # ★ 中古一戸建て
    if "/chukoikkodate/" in path:
        return ("chukoikkodate", fetch_chik, parse_list_chik, parse_detail_chik, polite_sleep_chik)

    # 新築戸建て
    if "/ikkodate/" in path:
        return ("ikkodate", fetch_ik, parse_list_ik, parse_detail_ik, polite_sleep_ik)

    # 新築マンション
    if "/ms/shinchiku/" in path:
        return ("shinchiku", fetch_new, parse_list_new, parse_detail_new, polite_sleep_new)

    # 既定：中古マンション
    return ("manshon", fetch_ms, parse_list_ms, parse_detail_ms, polite_sleep_ms)



def is_list_page(url: str) -> bool:
    p = urlparse(url)
    path = p.path or ""
    if "/tochi/" in path:
        return True
    # ★ 中古一戸建ての一覧判定（/bukken/ や /nc_ を含まない）
    if "/chukoikkodate/" in path:
        return ("/bukken/" not in path) and ("/nc_" not in path)
    if "/ikkodate/" in path:
        return ("/bukken/" not in path) and ("/nc_" not in path)
    if "/ms/shinchiku/" in path:
        return True
    return ("/ms/chuko/" in path) and ("/nc_" not in path)

def main(list_path: str):
    load_dotenv()
    conn = get_db()

    with open(list_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        try:
            kind, fetch, parse_list_page, parse_detail_page, polite_sleep = pick_scraper(url)

            resp = fetch(url)
            status = resp.status_code
            if status != 200:
                upsert_review(conn, url, status, "", "", {})
                polite_sleep()
                continue

            if is_list_page(url):
                items = parse_list_page(resp.text, url)
                if not items:
                    upsert_review(conn, url, status, "LIST_EMPTY", "", {})
                for it in items:
                    upsert_review(conn, it["url"], status, it["title"], it["body"], it["meta"])
                    polite_sleep()
            else:
                it = parse_detail_page(resp.text, url)
                upsert_review(conn, it["url"], status, it["title"], it["body"], it["meta"])
        except Exception as e:
            upsert_review(conn, url, 0, "", f"ERROR: {e}", {})
        # どのスクレイパでもpolite_sleep
        try:
            polite_sleep()
        except Exception:
            polite_sleep_ms()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline_scrape_suumo url_list.txt")
        sys.exit(1)
    main(sys.argv[1])
