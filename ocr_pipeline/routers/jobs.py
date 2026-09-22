"""Compatibility wrapper. Prefer /api/runs going forward — these endpoints proxy to it."""
import json

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import StreamingResponse

from ocr_pipeline.models.schemas import JobRequest, JobStatusResponse
from ocr_pipeline.services.db import get_db
from ocr_pipeline.services.run_orchestrator import get_orchestrator
from ocr_pipeline.services.startup import model_readiness


router = APIRouter()


@router.post("/api/jobs")
async def create_job(request: JobRequest):
    if not model_readiness.ready:
        raise HTTPException(status_code=503, detail=f"Model server not ready: {model_readiness.status}")

    orchestrator = get_orchestrator()
    try:
        result = await orchestrator.create_run(
            request.file_paths,
            name=request.name,
            strip_refs=request.strip_refs,
            export_parquet=request.export_parquet,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    orchestrator.start(
        result,
        strip_refs=request.strip_refs,
        export_parquet=request.export_parquet,
    )
    return {"job_id": result.run_id, "status": "queued"}


@router.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    db = get_db()
    run = await db.get_run(job_id)
    if not run:
        raise HTTPException(status_code=404, detail="Job not found")

    documents = await db.list_run_documents(job_id)
    in_flight = next(
        (d for d in documents if d["status"] == "processing"),
        None,
    )
    current_doc = in_flight["document_filename"] if in_flight else None

    return JobStatusResponse(
        job_id=job_id,
        status=run["status"],
        stage=run.get("stage"),
        progress=run.get("progress") or 0.0,
        documents_completed=run.get("documents_completed") or 0,
        documents_total=run.get("documents_total") or 0,
        current_document=current_doc,
        current_page=None,
        current_total_pages=in_flight.get("total_pages") if in_flight else None,
        pages_completed=run.get("pages_completed") or 0,
        pages_total=run.get("pages_total") or 0,
    )


@router.get("/api/jobs/{job_id}/stream")
async def stream_progress(job_id: str, after_event_id: int = 0):
    db = get_db()
    if not await db.get_run(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    orchestrator = get_orchestrator()

    async def event_generator():
        async for event in orchestrator.subscribe(job_id, after_event_id=after_event_id):
            event = {**event, "job_id": job_id}
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.websocket("/ws/jobs/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    db = get_db()
    if not await db.get_run(job_id):
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    orchestrator = get_orchestrator()
    try:
        async for event in orchestrator.subscribe(job_id):
            await websocket.send_json({**event, "job_id": job_id})
            if event.get("type") in ("run_complete", "run_failed"):
                break
    finally:
        await websocket.close()
