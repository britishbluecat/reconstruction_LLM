# src/advisor.py
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv
from prompt import build_prompt, default_payload, SYSTEM_PROMPT

def main():
    load_dotenv()
    client = OpenAI()

    # 入力: JSONファイル指定 or デフォルト
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = default_payload()

    user_prompt = build_prompt(payload)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # 応答を短くしすぎないためmax_tokensは指定しない（必要なら調整）
    )

    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
