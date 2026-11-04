import asyncio
import hmac
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import create_run, run_agent
from app.config import get_settings
from app.db import SessionLocal, engine, get_db, migrate
from app.execution import cancel_plan, execute_plan, manual_write, propose
from app.models import (
    Conversation,
    Event,
    Message,
    Note,
    Preference,
    Run,
    Task,
    Timer,
    Workflow,
    now,
)
from app.schemas import (
    ApprovalInput,
    ChatInput,
    NoteInput,
    PreferencesInput,
    RunRequest,
    TaskInput,
    TaskPatch,
    TimerInput,
    WorkflowInput,
)
from app.tools import serialize, validate_step
from app.trajectory import recover

settings = get_settings()
log = logging.getLogger("jarvis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
active_tasks: dict[str, asyncio.Task] = {}
rate_buckets: dict[str, deque] = defaultdict(deque)
local_whisper = None
transcription_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_app):
    migrate()
    recover()
    with SessionLocal() as db:
        if db.get(Preference, "default") is None:
            db.add(Preference(id="default", display_name="", instructions=""))
        db.commit()
    yield
    tasks = list(active_tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    engine.dispose()


app = FastAPI(
    title="Jarvis Workspace API",
    version="0.4.0",
    lifespan=lifespan,
    description="Single-user workspace with confirmed, auditable tool execution.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


def allowed_origin(origin: str | None) -> bool:
    return origin is None or origin in settings.origins


def rate_limit(key: str, limit: int = 90) -> bool:
    stamp = time.monotonic()
    # Bound memory use from arbitrary client addresses.
    if len(rate_buckets) > 1000:
        for name in list(rate_buckets):
            if not rate_buckets[name] or stamp - rate_buckets[name][-1] > 60:
                del rate_buckets[name]
    if key not in rate_buckets and len(rate_buckets) >= 1000:
        return False
    bucket = rate_buckets[key]
    while bucket and stamp - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(stamp)
    return True


@app.middleware("http")
async def boundary(request: Request, call_next):
    started = time.monotonic()
    request_id = str(uuid4())
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        if not allowed_origin(request.headers.get("origin")):
            return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
        if settings.app_token:
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(supplied.encode(), settings.app_token.encode()):
                return JSONResponse({"detail": "Workspace token required"}, status_code=401)
        if not rate_limit(request.client.host if request.client else "local"):
            return JSONResponse(
                {"detail": "Too many requests. Try again shortly."}, status_code=429
            )
        length = request.headers.get("content-length")
        if length and (
            not length.isdigit() or int(length) > settings.max_audio_bytes + 1024 * 1024
        ):
            return JSONResponse({"detail": "Request too large"}, status_code=413)
    response = await call_next(request)
    response.headers.update(
        {
            "X-Request-ID": request_id,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "microphone=(self)",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' ws: wss:; media-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
        }
    )
    # FastAPI's documentation uses a CDN and an inline initializer; keep the app policy strict.
    if request.url.path in {"/docs", "/redoc"}:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "img-src 'self' data: https://fastapi.tiangolo.com; connect-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
    log.info(
        "request id=%s method=%s path=%s status=%s ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.monotonic() - started) * 1000,
    )
    return response


def require(db: Session, model, item_id: str):
    item = db.get(model, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    return item


@app.get("/health")
def health():
    with SessionLocal() as db:
        db.execute(select(1))
    return {"status": "ok", "version": "0.4.0"}


@app.get("/api/status")
def status():
    model = settings.openai_model
    if settings.model_api == "chat_completions":
        model = settings.chat_model
    return {
        "ai_enabled": settings.ai_enabled,
        "model": model if settings.ai_enabled else None,
        "model_api": settings.model_api,
        "image_enabled": settings.ai_enabled,
        "web_search_enabled": settings.ai_enabled and settings.web_search_enabled,
        "transcription_provider": settings.transcription_provider,
        "voice_enabled": settings.transcription_provider == "local"
        or bool(settings.openai_api_key),
        "auth_required": bool(settings.app_token),
        "version": "0.4.0",
    }


@app.get("/api/workspace")
def workspace(db: Session = Depends(get_db)):
    result = {}
    for name, model, column in [
        ("notes", Note, Note.updated_at),
        ("tasks", Task, Task.created_at),
        ("timers", Timer, Timer.created_at),
        ("workflows", Workflow, Workflow.created_at),
        ("runs", Run, Run.created_at),
        ("conversations", Conversation, Conversation.created_at),
    ]:
        result[name] = [
            serialize(row) for row in db.scalars(select(model).order_by(column.desc()).limit(200))
        ]
    result["preferences"] = serialize(require(db, Preference, "default"))
    return result


@app.post("/api/conversations", status_code=201)
def new_conversation(db: Session = Depends(get_db)):
    item = Conversation()
    db.add(item)
    db.commit()
    return serialize(item)


@app.get("/api/conversations/{item_id}/messages")
def messages(item_id: UUID, db: Session = Depends(get_db)):
    require(db, Conversation, str(item_id))
    return [
        serialize(m)
        for m in db.scalars(
            select(Message)
            .where(Message.conversation_id == str(item_id))
            .order_by(Message.created_at)
        )
    ]


@app.get("/api/conversations/{item_id}/export")
def export_conversation(item_id: UUID, db: Session = Depends(get_db)):
    conv = require(db, Conversation, str(item_id))
    rows = messages(item_id, db)
    return {"title": conv.title, "exported_at": now(), "messages": rows}


def submit_manual(operation, arguments, item_id=None):
    run, result = manual_write(operation, arguments, item_id)
    if run["status"] != "completed":
        raise HTTPException(404 if run["result"].endswith("not found") else 409, run["result"])
    return result


@app.post("/api/notes", status_code=201)
def create_note(value: NoteInput):
    return submit_manual("create_note", value.model_dump())


@app.put("/api/notes/{item_id}")
def edit_note(item_id: UUID, value: NoteInput):
    return submit_manual("edit_note", value.model_dump(), str(item_id))


@app.delete("/api/notes/{item_id}", status_code=204)
def delete_note(item_id: UUID):
    submit_manual("delete_note", {}, str(item_id))


@app.post("/api/tasks", status_code=201)
def create_task(value: TaskInput):
    return submit_manual("add_task", value.model_dump())


@app.patch("/api/tasks/{item_id}")
def edit_task(item_id: UUID, value: TaskPatch):
    return submit_manual("edit_task", value.model_dump(), str(item_id))


@app.delete("/api/tasks/{item_id}", status_code=204)
def delete_task(item_id: UUID):
    submit_manual("delete_task", {}, str(item_id))


@app.post("/api/timers", status_code=201)
def create_timer(value: TimerInput):
    return submit_manual("start_timer", value.model_dump())


@app.delete("/api/timers/{item_id}", status_code=204)
def cancel_timer(item_id: UUID):
    submit_manual("cancel_timer", {}, str(item_id))


def workflow_values(value):
    values = value.model_dump()
    try:
        values["steps"] = [validate_step(step) for step in values["steps"]]
    except ValueError:
        raise HTTPException(422, "Invalid workflow step arguments") from None
    return values


@app.post("/api/workflows", status_code=201)
def create_workflow(value: WorkflowInput):
    return submit_manual("create_workflow", workflow_values(value))


@app.put("/api/workflows/{item_id}")
def edit_workflow(item_id: UUID, value: WorkflowInput):
    return submit_manual("edit_workflow", workflow_values(value), str(item_id))


@app.delete("/api/workflows/{item_id}", status_code=204)
def delete_workflow(item_id: UUID):
    submit_manual("delete_workflow", {}, str(item_id))


@app.post("/api/workflows/{item_id}/run", status_code=201)
def prepare_workflow(item_id: UUID, value: RunRequest, db: Session = Depends(get_db)):
    workflow = require(db, Workflow, str(item_id))
    steps = workflow.steps
    run, created = create_run(
        str(value.request_id),
        None,
        "/run " + workflow.name,
        kind="workflow",
        source={"workflow_id": str(item_id), "steps": steps},
    )
    if created:
        return propose(run["id"], steps)
    return run


@app.get("/api/runs/{item_id}")
def get_run(item_id: UUID, db: Session = Depends(get_db)):
    return serialize(require(db, Run, str(item_id)))


@app.get("/api/runs/{item_id}/events")
def events(item_id: UUID, db: Session = Depends(get_db)):
    require(db, Run, str(item_id))
    return [
        serialize(e)
        for e in db.scalars(
            select(Event).where(Event.run_id == str(item_id)).order_by(Event.sequence)
        )
    ]


@app.post("/api/runs/{item_id}/confirm")
def confirm(item_id: UUID, value: ApprovalInput):
    try:
        result = execute_plan(str(item_id), value.plan_hash)
        if result["status"] == "failed":
            raise HTTPException(409, result["result"])
        return result
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


@app.get("/api/runs/{item_id}/trajectory")
def trajectory(item_id: UUID, db: Session = Depends(get_db)):
    return {"run": serialize(require(db, Run, str(item_id))), "events": events(item_id, db)}


@app.post("/api/runs/{item_id}/retry", status_code=201)
def retry_run(item_id: UUID, value: RunRequest, db: Session = Depends(get_db)):
    source = require(db, Run, str(item_id))
    if (
        source.status in {"running", "waiting_confirmation"}
        or not isinstance(source.plan, dict)
        or "steps" not in source.plan
    ):
        raise HTTPException(409, "Only finished versioned plans can be proposed again")
    run, created = create_run(
        str(value.request_id), None, source.label, kind=source.kind, parent_run_id=source.id
    )
    return propose(run["id"], source.plan["steps"]) if created else run


@app.get("/api/export")
def export_workspace(db: Session = Depends(get_db)):
    result = {"version": "0.4.0", "exported_at": now()}
    for name, model in [
        ("notes", Note),
        ("tasks", Task),
        ("timers", Timer),
        ("workflows", Workflow),
        ("conversations", Conversation),
        ("messages", Message),
        ("runs", Run),
        ("events", Event),
        ("preferences", Preference),
    ]:
        result[name] = [serialize(row) for row in db.scalars(select(model))]
    return result


@app.post("/api/runs/{item_id}/cancel")
async def cancel(item_id: UUID, db: Session = Depends(get_db)):
    require(db, Run, str(item_id))
    task = active_tasks.get(str(item_id))
    if task:
        task.cancel()
        return {"status": "cancelling"}
    db.rollback()
    try:
        return cancel_plan(str(item_id))
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


@app.put("/api/preferences")
def preferences(value: PreferencesInput):
    return submit_manual("preferences", value.model_dump())


@app.post("/api/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    if not rate_limit("audio:" + (request.client.host if request.client else "local"), 10):
        raise HTTPException(429, "Too many transcription requests")
    if file.content_type not in {
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/wav",
        "audio/mpeg",
        "video/webm",
    }:
        raise HTTPException(415, "Unsupported recording format")
    data = bytearray()
    while chunk := await file.read(65536):
        data.extend(chunk)
        if len(data) > settings.max_audio_bytes:
            raise HTTPException(413, "Recording exceeds the 20 MB limit")
    if not data:
        raise HTTPException(422, "Recording is empty")
    suffix = {"audio/mp4": "m4a", "audio/ogg": "ogg", "audio/wav": "wav", "audio/mpeg": "mp3"}.get(
        file.content_type, "webm"
    )
    try:
        if settings.transcription_provider == "openai":
            if not settings.openai_api_key:
                raise HTTPException(
                    503, "Transcription needs a server-side API key or the optional local provider"
                )
            async with AsyncOpenAI(
                api_key=settings.openai_api_key, timeout=45, max_retries=0
            ) as client:
                result = await client.audio.transcriptions.create(
                    model=settings.transcription_model,
                    file=(f"recording.{suffix}", bytes(data), file.content_type),
                )
                text = result.text
        else:
            async with transcription_lock:
                text = await asyncio.to_thread(local_transcribe, bytes(data))
        return {"text": text.strip()}
    except ImportError:
        raise HTTPException(503, "Install the local-voice dependency group on the server") from None
    except (OpenAIError, ValueError, RuntimeError):
        raise HTTPException(
            502, "Transcription failed. Check provider availability and recording format."
        ) from None


def local_transcribe(data: bytes) -> str:
    import io

    from faster_whisper import WhisperModel

    global local_whisper
    if local_whisper is None:
        local_whisper = WhisperModel(
            settings.local_whisper_model, device="cpu", compute_type="int8"
        )
    segments, _ = local_whisper.transcribe(io.BytesIO(data), beam_size=1, vad_filter=True)
    return " ".join(segment.text for segment in segments)


@app.websocket("/api/chat")
async def chat(socket: WebSocket):
    if not allowed_origin(socket.headers.get("origin")):
        await socket.close(code=1008)
        return
    await socket.accept()
    current: asyncio.Task | None = None
    current_id: str | None = None
    send_lock = asyncio.Lock()

    async def emit(value):
        try:
            async with send_lock:
                await socket.send_json(value)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass  # Durable run status remains recoverable over HTTP after disconnect.

    try:
        auth = await asyncio.wait_for(socket.receive_json(), timeout=8)
        if (
            not isinstance(auth, dict)
            or auth.get("type") != "auth"
            or (
                settings.app_token
                and not hmac.compare_digest(
                    str(auth.get("token", "")).encode(), settings.app_token.encode()
                )
            )
        ):
            await socket.close(code=1008)
            return
        await emit({"type": "ready"})
        while True:
            raw = await socket.receive_text()
            if len(raw) > 64000:
                await emit({"type": "error", "text": "Message too large"})
                continue
            try:
                data = json.loads(raw)
                if data.get("type") == "cancel":
                    if current and not current.done():
                        current.cancel()
                    continue
                value = ChatInput.model_validate(data)
                if not rate_limit("chat:" + (socket.client.host if socket.client else "local"), 20):
                    await emit({"type": "error", "text": "Too many requests. Wait a minute."})
                    continue
                if current and not current.done():
                    await emit(
                        {
                            "type": "error",
                            "text": "Stop the current request before sending another.",
                        }
                    )
                    continue
                run_id, conversation_id = str(value.request_id), str(value.conversation_id)
                try:
                    saved, created = create_run(
                        run_id, conversation_id, value.text, images=value.images
                    )
                except ValueError as error:
                    await emit({"type": "error", "text": str(error)})
                    continue
                if not created:
                    await emit({"type": "replay", "run": saved})
                    continue
                await emit({"type": "started", "run_id": run_id})
                kwargs = {"images": value.images} if value.images else {}
                current = asyncio.create_task(
                    run_agent(run_id, value.text, conversation_id, emit, **kwargs)
                )
                current_id = run_id
                active_tasks[run_id] = current
                current.add_done_callback(lambda _task, rid=run_id: active_tasks.pop(rid, None))
            except (ValidationError, json.JSONDecodeError, AttributeError):
                await emit({"type": "error", "text": "Invalid chat request"})
    except (WebSocketDisconnect, TimeoutError, RuntimeError, ValueError):
        pass
    finally:
        if current and not current.done():
            current.cancel()
            await asyncio.gather(current, return_exceptions=True)
        if current_id:
            active_tasks.pop(current_id, None)


static = Path(settings.static_dir).resolve()
if static.is_dir():
    app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "API endpoint not found")
        candidate = (static / path).resolve()
        if candidate.is_relative_to(static) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(static / "index.html")


def workspace_openapi():
    """Expose the middleware bearer boundary in the interactive API contract."""
    from fastapi.openapi.utils import get_openapi

    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title, version=app.version, description=app.description, routes=app.routes
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["WorkspaceToken"] = {
        "type": "http",
        "scheme": "bearer",
        "description": "APP_TOKEN configured on this server. Optional only in empty-token development mode.",
    }
    for path, methods in schema["paths"].items():
        if path.startswith("/api/"):
            for operation in methods.values():
                operation["security"] = [{"WorkspaceToken": []}]
    app.openapi_schema = schema
    return schema


app.openapi = workspace_openapi
