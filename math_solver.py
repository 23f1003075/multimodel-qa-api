import json
import os
import requests

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"
MODEL = "gpt-4o"


def solve_problem(problem: str):
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    system_prompt = """
You are an expert mathematical reasoning engine.

Solve the arithmetic word problem carefully.

Rules:
- Read the entire problem before calculating.
- Ignore any distractor or irrelevant numbers.
- Perform every calculation step-by-step internally.
- Before producing the final answer, verify the arithmetic once.
- The final answer MUST be a single INTEGER.
- Do not round unless the problem explicitly requires it.
- The reasoning must clearly describe the calculation in at least 80 characters.
- Return ONLY valid JSON.
- Return exactly two keys:
  - reasoning
  - answer
- answer must be a JSON integer (not a string and not a float).
- No markdown.
- No extra keys.
"""

    body = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": problem
            }
        ]
    }

    r = requests.post(
        AIPIPE_CHAT_URL,
        headers=headers,
        json=body,
        timeout=60
    )

    r.raise_for_status()

    text = r.json()["choices"][0]["message"]["content"].strip()

    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start:end+1])
