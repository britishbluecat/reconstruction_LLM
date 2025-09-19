import pandas as pd
from pathlib import Path

# 関数群をインポート
from address_normalize_functions import (
    clean_property_name,
    load_selected_columns,
    normalize_location_numbers,
    normalize_exclusive_area,
    normalize_price,
    normalize_station_walk,
    normalize_built_year,
    normalize_layout,
    add_ward_city_and_city_town,
)


# パイプライン処理
mst_listings = (
    load_selected_columns()
    .pipe(clean_property_name)
    .pipe(normalize_location_numbers)
    .pipe(add_ward_city_and_city_town)
    .pipe(normalize_exclusive_area)
    .pipe(normalize_station_walk)
    .pipe(normalize_built_year)
    .pipe(normalize_layout)
    .pipe(normalize_price)
)

OUTPUT_FILE = "normalized_listings.csv"
mst_listings.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
