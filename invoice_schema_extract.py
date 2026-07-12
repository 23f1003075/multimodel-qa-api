import json
import os
import requests

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"
MODEL = "gpt-4o"


def extract_invoice(text: str, schema: dict):
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN not configured.")

    system_prompt = """
You are an expert invoice extraction engine.

You will receive:
1. Invoice text.
2. A JSON Schema describing the exact output.

Return ONLY valid JSON that exactly matches the provided schema.

Rules:
- No markdown.
- No explanation.
- No extra keys.
- No missing keys.
- Use null when information cannot be determined.
- Respect every type in the schema.
"""

    user_prompt = f"""
Invoice Text:

{text}

JSON Schema:

{json.dumps(schema, indent=2)}

Return ONLY the JSON object.
"""

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    body = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = requests.post(
        AIPIPE_CHAT_URL,
        headers=headers,
        json=body,
        timeout=60,
    )

    response.raise_for_status()

    result = response.json()["choices"][0]["message"]["content"].strip()

try:
    return json.loads(result)
except json.JSONDecodeError:
    start = result.find("{")
    end = result.rfind("}")

    if start != -1 and end != -1:
        return json.loads(result[start:end + 1])

    raise
