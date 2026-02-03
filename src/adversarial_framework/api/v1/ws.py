"""WebSocket endpoint for live session streaming.

Clients connect to ``/api/v1/ws/{session_id}`` and receive real-time
updates as the LangGraph adversarial loop executes.
"""

from __future__ import annotations

import uuid

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
            try:
                _connections[session_id].remove(websocket)
            except ValueError:
                pass
            if not _connections[session_id]:
                del _connections[session_id]
