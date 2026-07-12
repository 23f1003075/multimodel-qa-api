"""
Multimodal QA API (AIPipe version)
------------------------------------
POST /answer-image
Request:  {"image_base64": "<base64 string>", "question": "What is the total?"}
Response: {"answer": "4089.35"}

Uses your course-provided AIPipe token (OpenAI-compatible proxy) with a
vision-capable model to read the image and answer the question.

Requires the environment variable AIPIPE_TOKEN to be set on the host
(Render / Fly.io / HuggingFace Spaces all let you set env vars in their dashboard).
"""

import base64
import binascii
import os

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from invoice_extract import extract_invoice_fields
from dynamic_extract import dynamic_extract as run_dynamic_extract

app = FastAPI(title="Multimodal QA API (AIPipe)")

# CORS: allow the grader (running from a Cloudflare Worker / any origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"
MODEL = "gpt-4o"  # vision-capable model available through AIPipe


class AnswerImageRequest(BaseModel):
    image_base64: str
    question: str


class AnswerImageResponse(BaseModel):
    answer: str


def detect_media_type(raw_bytes: bytes) -> str:
    """Best-effort detection of image type from magic bytes, defaulting to PNG."""
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


@app.post("/answer-image", response_model=AnswerImageResponse)
def answer_image(payload: AnswerImageRequest):
    if not AIPIPE_TOKEN:
        raise HTTPException(status_code=500, detail="AIPIPE_TOKEN not configured on server.")

    # Strip data URL prefix if the client sent one, e.g. "data:image/png;base64,...."
    image_data = payload.image_base64
    if "," in image_data and image_data.strip().lower().startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    media_type = detect_media_type(raw_bytes)
    data_url = f"data:{media_type};base64,{image_data}"

    system_prompt = (
        "You are a precise document data extractor. You will be shown an image "
        "(a chart, receipt, invoice, table, or similar) and a question about it. "
        "Answer with ONLY the requested value, nothing else. "
        "If the answer is a number, respond with just the number "
        "(no currency symbols, no units, no commas, no extra words). "
        "If the answer is text, respond with just that text."
    )

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": payload.question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }

    try:
        r = requests.post(AIPIPE_CHAT_URL, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        answer_text = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

    return AnswerImageResponse(answer=answer_text)


class ExtractRequest(BaseModel):
    invoice_text: str


class ExtractResponse(BaseModel):
    invoice_no: str | None
    date: str | None
    vendor: str | None
    amount: float | None
    tax: float | None
    currency: str | None


@app.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest):
    fields = extract_invoice_fields(payload.invoice_text)
    return ExtractResponse(**fields)


from typing import Any
from pydantic import Field, ConfigDict


class DynamicExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    schema_: dict[str, str] = Field(alias="schema")


@app.post("/dynamic-extract")
def dynamic_extract_endpoint(payload: DynamicExtractRequest):
    try:
        result = run_dynamic_extract(payload.text, payload.schema_)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Extraction failed: {e}")
    return result


@app.get("/")
def health_check():
    return {"status": "ok"}
