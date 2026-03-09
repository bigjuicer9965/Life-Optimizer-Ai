import json
import os
import tempfile
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


def _memory_file_path() -> Path:
    env_path = os.getenv("MEMORY_FILE_PATH", "").strip()
    if env_path:
        return Path(env_path)
    if os.getenv("VERCEL", "").strip() == "1":
        return Path(tempfile.gettempdir()) / "life_optimizer_memory_store.json"
    return Path(__file__).resolve().parents[1] / "data" / "memory_store.json"


MEMORY_FILE = _memory_file_path()


def _load_store() -> dict[str, list[dict[str, Any]]]:
    if not MEMORY_FILE.exists():
        return {}

    try:
        content = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return content if isinstance(content, dict) else {}
    except Exception:
        return {}


def _save_store(store: dict[str, list[dict[str, Any]]]) -> None:
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")
    except OSError:
        # In read-only runtimes, silently skip local-file persistence.
        return


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
