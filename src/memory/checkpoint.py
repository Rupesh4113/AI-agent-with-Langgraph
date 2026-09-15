"""
Memory and Checkpointing Layer for LangGraph Research Agent.
Provides durable thread persistence, session resumption, and history management.
"""
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger("memory")

_MEMORY_SAVER_INSTANCE: Optional[MemorySaver] = None

def get_checkpointer(use_sqlite: bool = True) -> BaseCheckpointSaver:
    """
    Returns a configured LangGraph checkpointer.
    Prefers SqliteSaver for persistent storage on disk, with fallback to MemorySaver.
    """
    global _MEMORY_SAVER_INSTANCE

    if use_sqlite:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = Path(settings.CHECKPOINT_DB_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            # Create connection and initialize saver
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            saver = SqliteSaver(conn)
            saver.setup()
            return saver
        except Exception as exc:
            logger.warning(f"Could not initialize SqliteSaver ({exc}). Using MemorySaver.")

    if _MEMORY_SAVER_INSTANCE is None:
        _MEMORY_SAVER_INSTANCE = MemorySaver()
    return _MEMORY_SAVER_INSTANCE


def reset_memory() -> None:
    """Clears the global in-memory checkpointer."""
    global _MEMORY_SAVER_INSTANCE
    _MEMORY_SAVER_INSTANCE = MemorySaver()


def clear_thread_history(conversation_id: str) -> bool:
    """
    Clears checkpoint history for a specific conversation ID from SQLite if available.
    """
    try:
        db_path = Path(settings.CHECKPOINT_DB_PATH)
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            # Attempt deleting checkpoints matching thread_id if table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';"
            )
            if cursor.fetchone():
                cursor.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?;", (conversation_id,)
                )
                conn.commit()
            conn.close()
        reset_memory()
        return True
    except Exception as exc:
        logger.error(f"Failed to clear thread history for '{conversation_id}': {exc}")
        return False
