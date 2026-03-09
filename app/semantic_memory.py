import hashlib
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

VECTOR_SIZE = 128
COLLECTION_NAME = "life_optimizer_semantic_memory"
_client: chromadb.ClientAPI | None = None
_collection = None


class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for text in input:
            vector = [0.0] * VECTOR_SIZE
            tokens = (text or "").lower().split()
            if not tokens:
                embeddings.append(vector)
                continue

            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                index = int(digest[:8], 16) % VECTOR_SIZE
                vector[index] += 1.0

            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0:
                vector = [value / norm for value in vector]
            embeddings.append(vector)
        return embeddings


def _primitive_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}

    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        elif isinstance(value, (list, tuple, set)):
            cleaned[key] = ", ".join(str(item) for item in value)
        else:
            cleaned[key] = str(value)
    return cleaned


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    if os.getenv("CHROMA_DIR", "").strip():
        base_dir = Path(os.getenv("CHROMA_DIR", ""))
    elif os.getenv("VERCEL", "").strip() == "1":
        base_dir = Path("/tmp/chroma")
    else:
        base_dir = Path(__file__).resolve().parents[1] / "data" / "chroma"

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        base_dir = Path(tempfile.gettempdir()) / "life_optimizer_chroma"
        base_dir.mkdir(parents=True, exist_ok=True)

    _client = chromadb.PersistentClient(
        path=str(base_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=HashEmbeddingFunction(),
        metadata={"description": "Long-term user behavior memory"},
    )
    return _collection


def add_semantic_memory(user_id: str, text: str, metadata: dict[str, Any] | None = None) -> bool:
    if not user_id or user_id == "anonymous":
        return False
    if not text.strip():
        return False

    try:
        collection = _get_collection()
        full_metadata = {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **_primitive_metadata(metadata),
        }
        collection.add(
            ids=[f"{user_id}-{uuid.uuid4()}"],
            documents=[text],
            metadatas=[full_metadata],
        )
        return True
    except Exception:
        return False


def query_semantic_memory(user_id: str, query: str, n_results: int = 4) -> list[dict[str, Any]]:
    if not user_id or user_id == "anonymous" or not query.strip():
        return []

    try:
        collection = _get_collection()
        result = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"user_id": user_id},
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for index, text in enumerate(documents):
            score = None
            if index < len(distances) and distances[index] is not None:
                score = max(0.0, 1.0 - float(distances[index]))
            hits.append(
                {
                    "text": text,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "score": score,
                }
            )
        return hits
    except Exception:
        return []


def list_semantic_memories(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return []

    try:
        collection = _get_collection()
        result = collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
            limit=limit,
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        items: list[dict[str, Any]] = []
        for index, text in enumerate(documents):
            items.append(
                {
                    "text": text,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                }
            )
        return items
    except Exception:
        return []
