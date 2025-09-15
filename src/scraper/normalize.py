def to_llm_payload(meta_raw: dict) -> dict:
    # 取得した生メタから、社内で定義した物理名に正規化
    return {
        "location": meta_raw.get("location", ""),
        "building_age": meta_raw.get("building_age", ""),
        "floor_area": meta_raw.get("floor_area", ""),
        "structure": meta_raw.get("structure", ""),
        "current_condition": meta_raw.get("condition", ""),
        "desired_use": meta_raw.get("desired_use", ""),
    }
