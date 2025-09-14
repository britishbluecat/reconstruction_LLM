# src/prompt.py
SYSTEM_PROMPT = """あなたは建築業者の営業アシスタントAIです。
不確実な数値は幅で記述し、根拠を短く添えてください。
出力は必ずMarkdownで、指定のフォーマットを守ってください。"""

TEMPLATE = """以下の物件情報を読み取り、出力要件に従って回答してください。

[物件情報]
- 所在地: {location}
- 築年数: {building_age} 年
- 延床面積: {floor_area} ㎡
- 構造: {structure}
- 現況: {condition}
- 施主の希望: {desire}

[出力要件]
1) 建て替え / リノベ / 解体・更地売却 の3案を比較表で提示
2) 各案: 初期費用目安 / 工期感 / メリット / デメリット / 主なリスク / 向いている施主
3) 法令・制度の留意点を箇条書き（断定不可なら“可能性”表現）
4) 施主向け提案文（500字以内）
5) 現地確認チェックリスト
6) 追加入力テンプレ（不足情報）

[出力フォーマット]
## 提案比較表
| 案 | 初期費用目安 | 工期感 | メリット | デメリット | 主なリスク | 向いている施主 |
|---|---|---|---|---|---|---|
| 建て替え |  |  |  |  |  |  |
| リノベ |  |  |  |  |  |  |
| 解体・更地売却 |  |  |  |  |  |  |

## 法令・制度の留意点
- 

## 提案文ドラフト（施主向け）
（500字以内）

## 現地確認チェックリスト
- 

## 追加入力テンプレ
- 敷地面積 / 用途地域 / 建蔽率・容積率 / 接道状況 / 再建築可否 / インフラ（給排水・電気・ガス） / 既存不適格の有無 / 近隣相場感（賃料/売価）
"""

def build_prompt(payload: dict) -> str:
    return TEMPLATE.format(
        location=payload.get("location", ""),
        building_age=payload.get("building_age", ""),
        floor_area=payload.get("floor_area", ""),
        structure=payload.get("structure", ""),
        condition=payload.get("current_condition", ""),
        desire=payload.get("desired_use", ""),
    )

def default_payload():
    return {
        "location": "東京都八王子市",
        "building_age": 35,
        "floor_area": 80,
        "structure": "木造2階建て",
        "current_condition": "老朽化あり、内装傷み、屋根修理必要",
        "desired_use": "賃貸に活用したい",
    }
