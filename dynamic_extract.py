"""
Dynamic schema extraction logic.
Given free-form text and a caller-supplied schema (field_name -> type string),
uses an LLM (via AIPipe) to extract the fields, then strictly validates/coerces
the result so it always matches the requested schema exactly:
  - Exactly the keys in `schema` (no extras, none missing).
  - null for anything that can't be found.
  - Correct JSON types (int, float, bool, str, list) per the requested type.
"""

import json
import os
import re

import requests
from dateutil import parser as dateparser

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"

SUPPORTED_TYPES = {
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "array[string]",
    "array[integer]",
}

SYSTEM_PROMPT = """You are a precise information extraction engine.

You will be given:
1. A block of free-form text.
2. A JSON schema: an object mapping field names to a type name.

Extract the value of each field from the text, following these rules exactly:
- Output ONLY the keys listed in the schema. Do not add any extra keys. Do not omit any key.
- If a field's value cannot be determined from the text, use JSON null for that field.
- Respect the requested type for every field:
  - "string": a plain string.
  - "integer": a whole number (no quotes, no decimal point).
  - "float": a decimal number (no quotes).
  - "boolean": true or false (no quotes).
  - "date": an ISO-8601 date string "YYYY-MM-DD".
  - "array[string]": a JSON array of strings.
  - "array[integer]": a JSON array of integers.
- Output ONLY the raw JSON object as your entire response. No markdown, no code
  fences, no explanation, no preamble — just the JSON object itself.
"""


def _call_llm(text: str, schema: dict) -> str:
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN not configured on server.")

    user_prompt = (
        f"Text:\n{text}\n\n"
        f"Schema (field name -> type):\n{json.dumps(schema, indent=2)}\n\n"
        "Extract the fields per the rules above. Output only the JSON object."
    )

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    r = requests.post(AIPIPE_CHAT_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _parse_json_loosely(raw: str) -> dict:
    """Parse a JSON object out of the model's raw text response, tolerating
    markdown code fences or stray text around the JSON."""
    candidate = raw.strip()

    # Strip common markdown code-fence wrapping, e.g. ```json ... ```
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to extracting the first {...} block in the text.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response: {raw!r}")


def _coerce_string(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_integer(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        m = re.search(r"[-+]?\d+", value.replace(",", ""))
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                return None
    return None


def _coerce_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", value.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _coerce_boolean(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _coerce_date(value):
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        # Already ISO format
        iso_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", candidate)
        if iso_match:
            y, mo, d = (int(x) for x in iso_match.groups())
            try:
                from datetime import date
                return date(y, mo, d).isoformat()
            except ValueError:
                return None
        try:
            dt = dateparser.parse(candidate, fuzzy=True)
            if dt:
                return dt.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            return None
    return None


def _coerce_array_string(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return None


def _coerce_array_integer(value):
    if value is None:
        return None
    if isinstance(value, list):
        result = []
        for v in value:
            coerced = _coerce_integer(v)
            if coerced is not None:
                result.append(coerced)
        return result
    return None


_COERCERS = {
    "string": _coerce_string,
    "integer": _coerce_integer,
    "float": _coerce_float,
    "boolean": _coerce_boolean,
    "date": _coerce_date,
    "array[string]": _coerce_array_string,
    "array[integer]": _coerce_array_integer,
}


def _enforce_schema(raw_result: dict, schema: dict) -> dict:
    """Guarantee the output has exactly the schema's keys, with correctly
    coerced types, regardless of what the model actually returned."""
    output = {}
    for field, field_type in schema.items():
        coercer = _COERCERS.get(field_type, _coerce_string)
        raw_value = raw_result.get(field) if isinstance(raw_result, dict) else None
        try:
            output[field] = coercer(raw_value)
        except Exception:
            output[field] = None
    return output


def dynamic_extract(text: str, schema: dict) -> dict:
    raw_response = _call_llm(text, schema)
    try:
        parsed = _parse_json_loosely(raw_response)
    except ValueError:
        parsed = {}
    return _enforce_schema(parsed, schema)
