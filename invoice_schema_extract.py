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

Return ONLY valid JSON matching the provided schema.

Rules:

- vendor: copy exactly as written.
- currency: convert symbols or names into ISO 4217 code (USD, EUR, GBP, INR, JPY).
- total_amount: integer only, no commas or symbols.
- invoice_date: ALWAYS YYYY-MM-DD.
- due_in_days: integer.
- is_paid: boolean true/false.
- priority: one of low, normal, high, urgent (lowercase only).
- contact_email: lowercase.
- line_items: preserve order exactly.
- item_count: number of line_items.

Return ONLY JSON.
No markdown.
No explanation.
No extra keys.
No missing keys.
Use null if a value cannot be extracted.
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
