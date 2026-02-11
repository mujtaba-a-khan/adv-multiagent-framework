"""WebSocket endpoint for live session streaming.

Clients connect to ``/api/v1/ws/{session_id}`` and receive real-time
updates as the LangGraph adversarial loop executes.
"""

from __future__ import annotations

import contextlib

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)
router = APIRouter()

# In-memory connection registry (Phase 5 will use Redis pub/sub)
_connections: dict[str, list[WebSocket]] = {}


async def broadcast(session_id: str, message: dict) -> None:
    """Send a message to all WebSocket clients watching a session."""
    clients = _connections.get(session_id, [])
    disconnected: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        clients.remove(ws)


# Fine-tuning connections (separate registry)

_ft_connections: dict[str, list[WebSocket]] = {}


async def broadcast_finetuning(job_id: str, message: dict) -> None:
    """Send a message to all WebSocket clients watching a fine-tuning job."""
    clients = _ft_connections.get(job_id, [])
    disconnected: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        clients.remove(ws)


@router.websocket("/ws/finetuning/{job_id}")
async def ws_finetuning(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for live fine-tuning progress updates."""
    await websocket.accept()

    if job_id not in _ft_connections:
        _ft_connections[job_id] = []
    _ft_connections[job_id].append(websocket)

    logger.info("ft_ws_connected", job_id=job_id)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("ft_ws_disconnected", job_id=job_id)
    finally:
        if job_id in _ft_connections:
            with contextlib.suppress(ValueError):
                _ft_connections[job_id].remove(websocket)
            if not _ft_connections[job_id]:
                del _ft_connections[job_id]


# Playground connections (separate registry)

_pg_connections: dict[str, list[WebSocket]] = {}


async def broadcast_playground(conversation_id: str, message: dict) -> None:
    """Send a message to all WebSocket clients watching a playground."""
    clients = _pg_connections.get(conversation_id, [])
    disconnected: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        clients.remove(ws)


@router.websocket("/ws/playground/{conversation_id}")
async def ws_playground(websocket: WebSocket, conversation_id: str) -> None:
    """WebSocket endpoint for live playground updates."""
    await websocket.accept()

    if conversation_id not in _pg_connections:
        _pg_connections[conversation_id] = []
    _pg_connections[conversation_id].append(websocket)

    logger.info("pg_ws_connected", conversation_id=conversation_id)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(
            "pg_ws_disconnected",
            conversation_id=conversation_id,
        )
    finally:
        if conversation_id in _pg_connections:
            with contextlib.suppress(ValueError):
                _pg_connections[conversation_id].remove(websocket)
            if not _pg_connections[conversation_id]:
                del _pg_connections[conversation_id]


# Session connections


@router.websocket("/ws/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for live session updates.

    Messages sent to clients follow this schema::

        {
            "type": "turn_start" | "turn_complete" | "session_complete" | "error",
            "session_id": "...",
            "turn_number": 1,
            "data": { ... }
        }
    """
    await websocket.accept()

    # Register
    if session_id not in _connections:
        _connections[session_id] = []
    _connections[session_id].append(websocket)

    logger.info("ws_connected", session_id=session_id)

    try:
        # Keep connection alive until client disconnects
        while True:
            # We don't expect client messages, but await to detect disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("ws_disconnected", session_id=session_id)
    finally:
        if session_id in _connections:
            with contextlib.suppress(ValueError):
                _connections[session_id].remove(websocket)
            if not _connections[session_id]:
                del _connections[session_id]
