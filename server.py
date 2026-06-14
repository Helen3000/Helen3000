"""FastAPI server exposing the flight/hotel agent via SSE streaming."""

import os
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from agent import FlightHotelAgent

load_dotenv()

# ── Session store ──────────────────────────────────────────────────────────────
# In-memory store keyed by session_id. Each entry holds the agent and a timestamp.
# For multi-instance deployments, swap this for Redis.
_sessions: dict[str, dict] = {}
SESSION_TTL = 3600  # 1 hour


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, FlightHotelAgent]:
    now = time.time()

    # Evict expired sessions
    expired = [k for k, v in _sessions.items() if now - v["ts"] > SESSION_TTL]
    for k in expired:
        del _sessions[k]

    if session_id and session_id in _sessions:
        _sessions[session_id]["ts"] = now
        return session_id, _sessions[session_id]["agent"]

    sid = session_id or str(uuid.uuid4())
    agent = FlightHotelAgent(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    _sessions[sid] = {"agent": agent, "ts": now}
    return sid, agent


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    yield


app = FastAPI(
    title="Flight & Hotel Price Agent",
    description="Search for the cheapest flights and hotels across major platforms.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
async def search(req: SearchRequest):
    """
    Stream a flight/hotel search as Server-Sent Events.

    Event types:
      session   – {"session_id": "..."}  (first event, always)
      tool      – {"name": "web_search" | "web_fetch"}
      text      – {"chunk": "..."}
      done      – {"session_id": "..."}
      error     – {"message": "..."}
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    session_id, agent = _get_or_create_session(req.session_id)

    async def event_generator():
        # Always send session_id first so clients can resume
        yield {"event": "session", "data": json.dumps({"session_id": session_id})}

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_text(chunk: str):
            queue.put_nowait(("text", chunk))

        def on_tool_use(tool_name: str):
            queue.put_nowait(("tool", tool_name))

        def run_agent():
            try:
                agent.search(req.query, on_text=on_text, on_tool_use=on_tool_use)
            except Exception as exc:
                queue.put_nowait(("error", str(exc)))
            finally:
                queue.put_nowait(("done", None))

        # Run the blocking agent call in a thread so we don't block the event loop
        future = loop.run_in_executor(None, run_agent)

        while True:
            kind, value = await queue.get()
            if kind == "text":
                yield {"event": "text", "data": json.dumps({"chunk": value})}
            elif kind == "tool":
                yield {"event": "tool", "data": json.dumps({"name": value})}
            elif kind == "error":
                yield {"event": "error", "data": json.dumps({"message": value})}
                break
            elif kind == "done":
                yield {"event": "done", "data": json.dumps({"session_id": session_id})}
                break

        await future  # ensure thread is cleaned up

    return EventSourceResponse(event_generator())


@app.post("/session/reset")
def reset_session(req: ResetRequest):
    """Clear a session's conversation history."""
    if req.session_id in _sessions:
        _sessions[req.session_id]["agent"].reset()
        return {"reset": True}
    raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session entirely."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Session not found")
