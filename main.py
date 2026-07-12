"""
Multimodal QA API
------------------
POST /answer-image
Request:  {"image_base64": "<base64 string>", "question": "What is the total?"}
Response: {"answer": "4089.35"}

Uses Anthropic's Claude vision model to read the image and answer the question.
Requires the environment variable ANTHROPIC_API_KEY to be set on the host
(Render / Fly.io / HuggingFace Spaces / etc. all let you set env vars in their dashboard).
"""

import base64
import binascii
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

app = FastAPI(title="Multimodal QA API")

# CORS: allow the grader (running from a Cloudflare Worker / any origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


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
    # Strip data URL prefix if the client sent one, e.g. "data:image/png;base64,...."
    image_data = payload.image_base64
    if "," in image_data and image_data.strip().lower().startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    media_type = detect_media_type(raw_bytes)

    system_prompt = (
        "You are a precise document data extractor. You will be shown an image "
        "(a chart, receipt, invoice, table, or similar) and a question about it. "
        "Answer with ONLY the requested value, nothing else. "
        "If the answer is a number, respond with just the number "
        "(no currency symbols, no units, no commas, no extra words). "
        "If the answer is text, respond with just that text."
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": payload.question},
                    ],
                }
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

    answer_text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()

    return AnswerImageResponse(answer=answer_text)


@app.get("/")
def health_check():
    return {"status": "ok"}
