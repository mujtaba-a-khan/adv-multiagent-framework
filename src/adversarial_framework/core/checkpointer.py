"""LangGraph checkpointer backed by PostgreSQL (Neon).

Provides both async Postgres checkpointer for production and in-memory
checkpointer for CLI / testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def get_memory_checkpointer() -> MemorySaver:
    """Return an in-memory checkpointer (CLI, tests)."""
    return MemorySaver()


async def get_postgres_checkpointer(
    connection_string: str,
) -> BaseCheckpointSaver:
    """Return an async PostgreSQL checkpointer for production.

    Requires `langgraph-checkpoint-postgres` and a valid Neon connection
    string (`postgresql+psycopg://...` or `postgresql://...`).

    The caller is responsible for calling ``await saver.setup()`` after
    creation and ensuring the connection is closed on shutdown.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    """LangGraph's Postgres saver expects a psycopg connection string
    (not asyncpg), so convert if needed."""
    
    conn_str = connection_string
    if "+asyncpg" in conn_str:
        conn_str = conn_str.replace("+asyncpg", "")

    saver = AsyncPostgresSaver.from_conn_string(conn_str)
    await saver.setup()
    return saver
