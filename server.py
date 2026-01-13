"""
Epsilon API Server - Multi-User Version
Simple FastAPI server with working WebSocket for live progress and User Authentication.
"""

import os
import sys
import asyncio
import uuid
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from memory.supabase_client import SupabaseManager

# ============================================================
# WebSocket Manager
# ============================================================

class ConnectionManager:
    def __init__(self):
        # Map: run_id -> List[WebSocket]
        # We broadcast events for a specific run to listeners of that run
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, run_id: str, websocket: WebSocket):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
        print(f"[WS] Client connected to run {run_id}. Total listeners: {len(self.active_connections[run_id])}")
    
    def disconnect(self, run_id: str, websocket: WebSocket):
        if run_id in self.active_connections:
            if websocket in self.active_connections[run_id]:
                self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]
        print(f"[WS] Client disconnected from run {run_id}.")
    
    async def broadcast(self, run_id: str, message: dict):
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"[WS] Error sending to client: {e}")

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
# Auth & Models
# ============================================================

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str

class ResearchRequest(BaseModel):
    goal: str
    max_iterations: Optional[int] = 5
    dataset_link: Optional[str] = None
    dataset_config: Optional[Dict[str, Any]] = None

class ResearchResponse(BaseModel):
    run_id: str
    status: str
    message: str

class CodeRequest(BaseModel):
    code: str

# In-memory Helper for simpler Auth (Production should use proper hashing/salting lib like bcrypt)
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-ID header")
    return x_user_id

# ============================================================
# State
# ============================================================

# run_id -> { status, goal, user_id, ... }
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
            await manager.broadcast(run_id, event)
        except asyncio.TimeoutError:
            break
        except Exception as e:
            print(f"[WS] Broadcast error: {e}")
            break


def make_callback(run_id: str, loop: asyncio.AbstractEventLoop):
    """Create a callback for the controller to emit events."""
    def callback(event_type: str, payload: dict):
        queue = event_queues.get(run_id)
        print(f"[CALLBACK] Event {event_type} for run {run_id}, queue exists: {queue is not None}, loop running: {loop.is_running()}")
        if queue and loop.is_running():
            loop.call_soon_threadsafe(queue.put_nowait, payload)
            print(f"[CALLBACK] Queued event {event_type}")
    return callback


async def run_pipeline(run_id: str, user_id: str, goal: str, max_iterations: int, dataset_config: Optional[Dict] = None):
    """Run the research pipeline with event emission."""
    from controller import ResearchController
    
    loop = asyncio.get_running_loop()
    event_queues[run_id] = asyncio.Queue()
    
    # Start broadcaster
    broadcaster = asyncio.create_task(broadcast_events(run_id))
    
    try:
        callback = make_callback(run_id, loop)
        controller = ResearchController(user_id=user_id, max_iterations=max_iterations, event_callback=callback)
        controller.run_id = uuid.UUID(run_id)
        
        full_goal = goal
        if dataset_config:
             full_goal += f"\n\n[USER PROVIDED DATASET CONFIG]: {json.dumps(dataset_config)}"
        
        # Run in thread (blocking)
        await asyncio.to_thread(controller.run, full_goal)
        
        active_runs[run_id]["status"] = "completed"
        
    except Exception as e:
        await manager.broadcast(run_id, {"type": "error", "run_id": run_id, "error": str(e)})
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

# --- Auth ---

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    sb = SupabaseManager()
    
    # Check if user exists
    existing = sb.client.table("users").select("id").eq("email", req.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="User already exists")
        
    pwd_hash = hash_password(req.password)
    res = sb.client.table("users").insert({
        "email": req.email, 
        "password_hash": pwd_hash
    }).execute()
    
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create user")
        
    return {"token": res.data[0]["id"], "email": req.email} # Use ID as token for simplicity


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    sb = SupabaseManager()
    
    pwd_hash = hash_password(req.password)
    res = sb.client.table("users").select("*").eq("email", req.email).eq("password_hash", pwd_hash).execute()
    
    if not res.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    return {"token": res.data[0]["id"], "email": req.email} # Use ID as token for simplicity


# --- Research ---

@app.post("/api/research/start", response_model=ResearchResponse)
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks, user_id: str = Depends(get_user_id)):
    run_id = str(uuid.uuid4())
    
    ds_config = req.dataset_config
    if req.dataset_link and not ds_config:
        ds_config = {
            "type": "external",
            "dataset_id": req.dataset_link,
            "description": "User provided link/ID"
        }

    active_runs[run_id] = {
        "status": "running",
        "goal": req.goal,
        "max_iterations": req.max_iterations,
        "user_id": user_id
    }
    
    background_tasks.add_task(run_pipeline, run_id, user_id, req.goal, req.max_iterations, ds_config)
    
    return ResearchResponse(
        run_id=run_id,
        status="started",
        message=f"Research started: {req.goal}"
    )


@app.get("/api/research/status/{run_id}")
async def get_status(run_id: str, user_id: str = Depends(get_user_id)):
    if run_id not in active_runs:
        # Check DB if not active? (Not implemented for status, assuming volatile for now)
        return {"status": "not_found"}
        
    run = active_runs[run_id]
    if run["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized access to this run")
        
    return run


@app.get("/api/research/list")
async def list_runs(user_id: str = Depends(get_user_id)):
    """List active runs for user."""
    return [
        {"run_id": k, **v} 
        for k, v in active_runs.items() 
        if v.get("user_id") == user_id
    ]


def get_run_dir(run_id: str) -> Path:
    return Path(__file__).parent / "experiments" / run_id

@app.get("/api/research/{run_id}/code")
async def get_experiment_code(run_id: str, user_id: str = Depends(get_user_id)):
    # Validate ownership via active runs (weak check, better to check DB if persistent)
    if run_id in active_runs and active_runs[run_id]["user_id"] != user_id:
         raise HTTPException(status_code=403, detail="Unauthorized")

    path = get_run_dir(run_id) / "run_experiment.py"
    if not path.exists():
        return {"code": "# No code generated yet."}
    return {"code": path.read_text()}


@app.post("/api/research/{run_id}/code")
async def update_experiment_code(run_id: str, req: CodeRequest, user_id: str = Depends(get_user_id)):
    if run_id in active_runs and active_runs[run_id]["user_id"] != user_id:
         raise HTTPException(status_code=403, detail="Unauthorized")
         
    path = get_run_dir(run_id) / "run_experiment.py"
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
    path.write_text(req.code)
    return {"status": "updated", "message": "Code updated successfully"}


@app.get("/api/research/{run_id}/report")
async def get_report(run_id: str, user_id: str = Depends(get_user_id)):
    # Basic check
    if run_id in active_runs and active_runs[run_id]["user_id"] != user_id:
         raise HTTPException(status_code=403, detail="Unauthorized")
    
    report = {
        "summary": {},
        "raw_results": {},
        "execution_log": "",
        "code": ""
    }
    
    base = get_run_dir(run_id)
    
    if (base / "experiment_summary.json").exists():
        report["summary"] = json.loads((base / "experiment_summary.json").read_text())
        
    if (base / "raw_results.json").exists():
        try:
             report["raw_results"] = json.loads((base / "raw_results.json").read_text())
        except:
             report["raw_results"] = {"error": "Could not parse raw_results.json"}
             
    if (base / "execution.log").exists():
        report["execution_log"] = (base / "execution.log").read_text()
        
    if (base / "run_experiment.py").exists():
        report["code"] = (base / "run_experiment.py").read_text()

    if (base / "dataset_used.json").exists():
        try:
            report["dataset"] = json.loads((base / "dataset_used.json").read_text())
        except:
            pass

    return report


@app.get("/api/memory/evidence")
async def get_evidence(query: str = "", limit: int = 10, user_id: str = Depends(get_user_id)):
    from memory.memory_service import MemoryService
    svc = MemoryService()
    results = svc.get_evidence(user_id=user_id, goal=query, limit=limit)
    return {"evidence": results, "count": len(results)}


@app.get("/api/memory/knowledge")
async def get_knowledge(query: str = "", limit: int = 10, user_id: str = Depends(get_user_id)):
    from memory.memory_service import MemoryService
    svc = MemoryService()
    results = svc.get_knowledge(user_id=user_id, goal=query, limit=limit)
    return {"knowledge": results, "count": len(results)}


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    # In a real app we'd validate token here using query param ?token=...
    await manager.connect(run_id, websocket)
    
    # Send connection confirmation
    await websocket.send_json({"type": "connected", "run_id": run_id})
    
    try:
        while True:
            data = await websocket.receive_text()
            # Just acknowledge
            # await websocket.send_json({"type": "ack"})
            pass
    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
