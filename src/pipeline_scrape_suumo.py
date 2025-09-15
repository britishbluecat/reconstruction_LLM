import os
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv

from scraper.suumo_mansion_review import fetch, parse_list_page, parse_detail_page, polite_sleep
from storage import get_db, upsert_review

def is_list_page(url: str) -> bool:
    """
    極めて簡易な判定。
    /ms/chuko/ かつ /nc_ が含まれていなければ「一覧」と見なす。
    """
    p = urlparse(url)
    path = p.path or ""
    return ("/ms/chuko/" in path) and ("/nc_" not in path)

def main(list_path: str):
    load_dotenv()
    conn = get_db()

    with open(list_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        try:
            resp = fetch(url)
            status = resp.status_code
            if status != 200:
                upsert_review(conn, url, status, "", "", {})
                polite_sleep()
                continue

            if is_list_page(url):
                items = parse_list_page(resp.text, url)
                if not items:
                    # 一覧のはずが0件 → タイトルだけ記録しておく
                    upsert_review(conn, url, status, "LIST_EMPTY", "", {})
                for it in items:
                    upsert_review(conn, it["url"], status, it["title"], it["body"], it["meta"])
                    polite_sleep()
            else:
                # 詳細ページ
                it = parse_detail_page(resp.text, url)
                upsert_review(conn, it["url"], status, it["title"], it["body"], it["meta"])
        except Exception as e:
            upsert_review(conn, url, 0, "", f"ERROR: {e}", {})
        polite_sleep()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline_scrape_suumo url_list.txt")
        sys.exit(1)
    main(sys.argv[1])
