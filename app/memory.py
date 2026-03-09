import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MEMORY_FILE = DATA_DIR / "memory_store.json"


def _load_store() -> dict[str, list[dict[str, Any]]]:
    if not MEMORY_FILE.exists():
        return {}

    try:
        content = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return content if isinstance(content, dict) else {}
    except Exception:
        return {}


def _save_store(store: dict[str, list[dict[str, Any]]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def load_memory(user_id: str) -> dict[str, Any]:
    memory = _load_store().get(user_id, [])
    history: list[str] = []

    for item in memory:
        user_text = item.get("user_input")
        recommendation = item.get("recommendation")
        if user_text:
            history.append(f"User: {user_text}")
        if recommendation:
            history.append(f"Agent: {recommendation}")

    return {"user_id": user_id, "memory": memory, "history": history}


def save_memory(
    user_id: str,
    user_input: str,
    recommendation: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    store = _load_store()
    entry = {
        "user_input": user_input,
        "recommendation": recommendation,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    store.setdefault(user_id, []).append(entry)
    _save_store(store)


def get_recent_history(user_id: str, limit: int = 10) -> list[str]:
    history = load_memory(user_id)["history"]
    if limit <= 0:
        return []
    return history[-limit:]
