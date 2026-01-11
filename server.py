"""
Epsilon API Server - Minimal Working Version
Simple FastAPI server with working WebSocket for live progress.
"""

import os
import sys
import asyncio
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


# ============================================================
# WebSocket Manager
# ============================================================

class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(self.connections)}")
    
    async def broadcast(self, message: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()


# ============================================================
# App Setup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Epsilon API starting...")
    yield
    print("👋 Epsilon API shutting down...")

app = FastAPI(title="Epsilon API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Models
# ============================================================

class ResearchRequest(BaseModel):
    goal: str
    max_iterations: Optional[int] = 5

class ResearchResponse(BaseModel):
    run_id: str
    status: str
    message: str


# ============================================================
# State
# ============================================================

active_runs: Dict[str, Dict[str, Any]] = {}
event_queues: Dict[str, asyncio.Queue] = {}


# ============================================================
# Research Pipeline
# ============================================================

async def broadcast_events(run_id: str):
    """Pull events from queue and broadcast to WebSocket."""
    queue = event_queues.get(run_id)
    if not queue:
        return
    
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=600)
            if event is None:
                break
            await manager.broadcast(event)
        except asyncio.TimeoutError:
            break
        except Exception as e:
            print(f"[WS] Broadcast error: {e}")
            break


def make_callback(run_id: str, loop: asyncio.AbstractEventLoop):
    """Create a callback for the controller to emit events."""
    def callback(event_type: str, payload: dict):
        queue = event_queues.get(run_id)
        if queue and loop.is_running():
            loop.call_soon_threadsafe(queue.put_nowait, payload)
    return callback


async def run_pipeline(run_id: str, goal: str, max_iterations: int):
    """Run the research pipeline with event emission."""
    from controller import ResearchController
    
    loop = asyncio.get_running_loop()
    event_queues[run_id] = asyncio.Queue()
    
    # Start broadcaster
    broadcaster = asyncio.create_task(broadcast_events(run_id))
    
    try:
        callback = make_callback(run_id, loop)
        controller = ResearchController(max_iterations=max_iterations, event_callback=callback)
        controller.run_id = uuid.UUID(run_id)
        
        # Run in thread (blocking)
        await asyncio.to_thread(controller.run, goal)
        
        active_runs[run_id]["status"] = "completed"
        
    except Exception as e:
        await manager.broadcast({"type": "error", "run_id": run_id, "error": str(e)})
        active_runs[run_id]["status"] = "failed"
        active_runs[run_id]["error"] = str(e)
    
    finally:
        if run_id in event_queues:
            await event_queues[run_id].put(None)
        await broadcaster
        event_queues.pop(run_id, None)


# ============================================================
# Routes
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/research/start", response_model=ResearchResponse)
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    
    active_runs[run_id] = {
        "status": "running",
        "goal": req.goal,
        "max_iterations": req.max_iterations
    }
    
    background_tasks.add_task(run_pipeline, run_id, req.goal, req.max_iterations)
    
    return ResearchResponse(
        run_id=run_id,
        status="started",
        message=f"Research started: {req.goal}"
    )


@app.get("/api/research/status/{run_id}")
async def get_status(run_id: str):
    if run_id not in active_runs:
        return {"status": "not_found"}
    return active_runs[run_id]


@app.get("/api/memory/evidence")
async def get_evidence(query: str = "", limit: int = 10):
    from memory.memory_service import MemoryService
    svc = MemoryService()
    results = svc.get_evidence(goal=query, limit=limit)
    return {"evidence": results, "count": len(results)}


@app.get("/api/memory/knowledge")
async def get_knowledge(query: str = "", limit: int = 10):
    from memory.memory_service import MemoryService
    svc = MemoryService()
    results = svc.get_knowledge(goal=query, limit=limit)
    return {"knowledge": results, "count": len(results)}


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send connection confirmation
    await websocket.send_json({"type": "connected"})
    
    try:
        while True:
            data = await websocket.receive_text()
            # Just acknowledge
            await websocket.send_json({"type": "ack"})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
