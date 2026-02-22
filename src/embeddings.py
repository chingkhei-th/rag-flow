"""
Custom Ollama Embeddings wrapper.

The standard `langchain-ollama` OllamaEmbeddings sometimes triggers a NaN
error on the server when embedding longer paragraphs in one shot. This wrapper
works around the issue by:
  1. Calling the /api/embed endpoint directly via `requests` (bypasses the
     langchain-ollama SDK's internal routing).
  2. Splitting each document into smaller sentences and averaging the per-
     sentence vectors, which keeps individual payloads small and avoids the
     numerical instability in the model.

Usage::

    from src.embeddings import OllamaRobustEmbeddings
    embeddings = OllamaRobustEmbeddings(base_url=..., model=...)
"""

import re
import requests
from langchain_core.embeddings import Embeddings
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def _mean_vector(vecs: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a list of vectors."""
    if not vecs:
        return []
    size = len(vecs[0])
    result = [0.0] * size
    for v in vecs:
        for i, val in enumerate(v):
            result[i] += val
    return [x / len(vecs) for x in result]


def _embed_texts(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    """
    Call the Ollama /api/embed endpoint with full document chunks.
    This version provides maximum accuracy by preserving context.
    """
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/embed",
            json={"model": model, "input": texts},
            timeout=120,
        )

        if response.status_code != 200:
            print(f"Error from Ollama ({response.status_code}): {response.text}")
            response.raise_for_status()

        data = response.json()
        if "embeddings" not in data or not data["embeddings"]:
            raise ValueError(f"No embeddings returned from Ollama. Response: {data}")

        return data["embeddings"]
    except Exception as e:
        print(f"\nCRITICAL ERROR during embedding batch (Size: {len(texts)})")
        print(f"Model: {model} | URL: {base_url}")
        raise e


class OllamaRobustEmbeddings(Embeddings):
    """
    A High-Accuracy LangChain-compatible Embeddings class.
    Sends full chunks to Ollama to preserve narrative flow and meaning.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents in one batch for speed and accuracy."""
        return _embed_texts(texts, self.base_url, self.model)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return _embed_texts([text], self.base_url, self.model)[0]
