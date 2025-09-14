import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ====== 設定 ======
MODEL_NAME = "gpt-4o-mini"  # 速さ/コスト重視の現実解
TEMPERATURE = 0.2
PER_ROW_DELAY_SEC = 0.2      # 行ごとのインターバル（簡易レート制御）

# 日本語(論理名) → 物理名 のマッピング
LOGICAL_TO_PHYSICAL = {
    "住所": "location",
    "築年": "building_age",
    "面積": "floor_area",
    "築造": "structure",
    "状態": "current_condition",
    "希望用": "desired_use",
}

# 物理名（英語） → 日本語（論理名）の逆マップ（後で使う）
PHYSICAL_TO_LOGICAL = {v: k for k, v in LOGICAL_TO_PHYSICAL.items()}

# 追記する出力列（7列）
OUTPUT_COLS = ["案", "初期費用目安", "工期感", "メリット", "デメリット", "主なリスク", "向いている施主"]

# ====== OpenAI 初期化 ======
load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """あなたは建築業者の営業アシスタントAIです。
与えられた1件の物件情報に対して、「建て替え / リノベ / 解体・更地売却」の3案を比較し、各案の
- 案（案名）
- 初期費用目安（幅で/根拠を簡記）
- 工期感（幅で）
- メリット
- デメリット
- 主なリスク
- 向いている施主
をJSON形式で返してください。

厳守事項:
- 出力は**厳密に**JSONオブジェクトで、キーは日本語: 「案」「初期費用目安」「工期感」「メリット」「デメリット」「主なリスク」「向いている施主」。
- 3案を配列として返す（配列長は3、順に 建て替え/リノベ/解体・更地売却）。
- 不確実な数値は幅で。根拠は短く。法令/制度は断定せず可能性表現。
- 文章は簡潔。Excelセルで見やすいよう、過度に長くしない。
"""

USER_TEMPLATE = """次の物件情報に対して3案の比較を出力して下さい。

[物件情報]
- 所在地: {location}
- 築年数: {building_age} 年
- 延床面積: {floor_area} ㎡
- 構造: {structure}
- 現況: {current_condition}
- 施主の希望: {desired_use}

出力はJSONのみ。例:
[
  {{
    "案": "建て替え",
    "初期費用目安": "2,000〜2,600万円（坪単価×延床/耐震等級2〜3想定）",
    "工期感": "5〜8ヶ月",
    "メリット": "耐震性能/配管一新/間取り自由度",
    "デメリット": "費用高/工期長/仮住まい必要",
    "主なリスク": "地盤/近隣調整/資材価格変動",
    "向いている施主": "長期居住/資産価値重視"
  }},
  ...
]"""

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    入力Excelの列見出しが 日本語(論理名) or 物理名(英語) のどちらでも
    内部的に物理名に統一する。
    """
    col_map = {}
    for col in df.columns:
        col_strip = str(col).strip()
        # 1) すでに物理名（英語）か？
        if col_strip in PHYSICAL_TO_LOGICAL:
            col_map[col] = col_strip
            continue
        # 2) 日本語(論理名)なら物理名へ
        if col_strip in LOGICAL_TO_PHYSICAL:
            col_map[col] = LOGICAL_TO_PHYSICAL[col_strip]
            continue
        # 3) どちらでもない → そのまま
        col_map[col] = col_strip
    return df.rename(columns=col_map)

def row_to_payload(row: pd.Series) -> dict:
    """ 行データから物理名キーのペイロードを作成 """
    return {
        "location": row.get("location", ""),
        "building_age": row.get("building_age", ""),
        "floor_area": row.get("floor_area", ""),
        "structure": row.get("structure", ""),
        "current_condition": row.get("current_condition", ""),
        "desired_use": row.get("desired_use", ""),
    }

def call_gpt(payload: dict) -> list:
    """ GPT呼び出し。3案の配列(JSON)をパースして返す。 """
    user_prompt = USER_TEMPLATE.format(
        location=payload.get("location", ""),
        building_age=payload.get("building_age", ""),
        floor_area=payload.get("floor_area", ""),
        structure=payload.get("structure", ""),
        desired_use=payload.get("desired_use", ""),
        current_condition=payload.get("current_condition", ""),
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # max_tokensは状況に応じて
    )
    content = resp.choices[0].message.content

    # JSONだけが返る想定だが、念のため安全にパース
    # 返答にコードフェンスが付くケースなども考慮
    import json
    text = content.strip()
    if text.startswith("```"):
        # ```json ... ``` を剥がす
        text = text.strip("`")
        # 最初のjsonやlang指定行を落とす
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline+1:]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("GPTの出力が配列ではありません")
    # 各要素が必要キーを持っているか念のため整える
    normalized = []
    for item in data:
        normalized.append({
            "案": item.get("案", ""),
            "初期費用目安": item.get("初期費用目安", ""),
            "工期感": item.get("工期感", ""),
            "メリット": item.get("メリット", ""),
            "デメリット": item.get("デメリット", ""),
            "主なリスク": item.get("主なリスク", ""),
            "向いている施主": item.get("向いている施主", ""),
        })
    return normalized

def plans_to_multiline(plans: list, key: str) -> str:
    """
    3案の同じキーを改行で連結して1セルに入れる。
    Excelでは改行で見やすく（Alt+Enter相当）扱える。
    """
    vals = []
    for p in plans:
        vals.append(str(p.get(key, "")).strip())
    return "\n".join(vals)

def process_excel(path: str) -> str:
    # 読み込み（最初のシート）
    df = pd.read_excel(path)
    df = normalize_columns(df)

    # 必要カラムがあるかチェック（最低限6項目のどれかが存在）
    required_phys = ["location", "building_age", "floor_area", "structure", "current_condition", "desired_use"]
    if not any(col in df.columns for col in required_phys):
        raise ValueError("入力Excelに必要な列（住所/築年/面積/築造/状態/希望用 または location/building_age/...）が見つかりません。")

    # 出力列を用意（既にあるなら上書き）
    for col in OUTPUT_COLS:
        df[col] = ""

    # 行ごとに処理
    for idx, row in df.iterrows():
        payload = row_to_payload(row)
        try:
            plans = call_gpt(payload)  # 3案の配列
            # 各列を改行連結で書き込み
            for col in OUTPUT_COLS:
                df.at[idx, col] = plans_to_multiline(plans, col)
        except Exception as e:
            # エラー時はメッセージを入れておく
            msg = f"生成失敗: {e}"
            for col in OUTPUT_COLS:
                df.at[idx, col] = msg
        time.sleep(PER_ROW_DELAY_SEC)

    # 保存先
    base, ext = os.path.splitext(path)
    out_path = f"{base}_処理済み.xlsx"
    df.to_excel(out_path, index=False)
    return out_path

# ====== GUI (tkinter) ======
def run_gui():
    root = tk.Tk()
    root.title("空き物件アドバイザー（Excel処理）")

    frm = tk.Frame(root, padx=16, pady=16)
    frm.pack()

    lbl = tk.Label(frm, text="Excelファイル（.xlsx）を選んで処理します。")
    lbl.grid(row=0, column=0, sticky="w")

    status_var = tk.StringVar(value="待機中")
    status = tk.Label(frm, textvariable=status_var, fg="gray")
    status.grid(row=1, column=0, sticky="w", pady=(8, 16))

    def on_select():
        path = filedialog.askopenfilename(
            title="Excelを選択",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not path:
            return
        try:
            status_var.set("処理中…（行ごとにGPTに問い合わせます）")
            root.update_idletasks()
            out = process_excel(path)
            status_var.set(f"完了: {out}")
            messagebox.showinfo("完了", f"処理が完了しました。\n\n保存先:\n{out}")
        except Exception as e:
            status_var.set("エラー発生")
            messagebox.showerror("エラー", str(e))

    btn = tk.Button(frm, text="Excelを選択して処理", command=on_select)
    btn.grid(row=2, column=0, sticky="w")

    root.mainloop()

if __name__ == "__main__":
    # 事前にAPIキーが設定されているか軽く確認
    if not os.getenv("OPENAI_API_KEY"):
        print("※ OPENAI_API_KEY が環境変数または .env に設定されていません。")
    run_gui()
