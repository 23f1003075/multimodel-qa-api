import os
import requests
import math

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_BASE = "https://aipipe.org/openai/v1"
MODEL = "text-embedding-3-small"


def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    body = {
        "model": MODEL,
        "input": text
    }

    r = requests.post(
        f"{AIPIPE_BASE}/embeddings",
        headers=headers,
        json=body,
        timeout=60,
    )

    r.raise_for_status()

    return r.json()["data"][0]["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def rank(query, candidates):
    q = get_embedding(query)

    scores = []

    for i, text in enumerate(candidates):
        emb = get_embedding(text)
        scores.append((cosine(q, emb), i))

    scores.sort(reverse=True)

    return [i for _, i in scores[:3]]
