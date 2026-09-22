"""Single-PDF extraction. Thin wrapper around the run orchestrator."""
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ocr_pipeline.models.schemas import ExtractRequest, ExtractResponse
from ocr_pipeline.services.db import get_db
from ocr_pipeline.services.document_catalog import SUPPORTED_SOURCE_SUFFIXES
from ocr_pipeline.services.run_orchestrator import get_orchestrator
from ocr_pipeline.services.startup import model_readiness


router = APIRouter()


@router.post("/api/extract", response_model=ExtractResponse)
async def extract_pdf(request: ExtractRequest):
    if not model_readiness.ready:
        raise HTTPException(status_code=503, detail=f"Model server not ready: {model_readiness.status}")

    pdf_path = Path(request.file_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    if pdf_path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
        raise HTTPException(status_code=400, detail="File must be a PDF or EPUB")

    orchestrator = get_orchestrator()
    result = await orchestrator.create_run(
        [str(pdf_path)],
        strip_refs=request.strip_refs,
        export_parquet=request.export_parquet,
    )
    task = orchestrator.start(
        result,
        strip_refs=request.strip_refs,
        export_parquet=request.export_parquet,
    )
    await task

    db = get_db()
    run = await db.get_run(result.run_id)
    if not run:
        raise HTTPException(status_code=500, detail="Run vanished after creation")
    if run["status"] == "failed":
        raise HTTPException(status_code=500, detail=run.get("error") or "Run failed")

    staged = result.documents[0]
    rd = await db.get_run_document(result.run_id, staged.document_id)
    if not rd:
        raise HTTPException(status_code=500, detail="Document not registered")

    return ExtractResponse(
        run_id=result.run_id,
        document_id=staged.document_id,
        filename=staged.filename,
        total_pages=rd.get("total_pages") or 0,
        pages_pass=rd.get("pages_pass") or 0,
        pages_warn=rd.get("pages_warn") or 0,
        pages_fail=rd.get("pages_fail") or 0,
        pages_empty=rd.get("pages_empty") or 0,
        total_processing_time_ms=rd.get("total_processing_time_ms") or 0.0,
        output_raw_txt=rd.get("artifact_raw_txt") or "",
        output_txt=rd.get("artifact_clean_txt") or "",
        output_md=rd.get("artifact_markdown") or "",
        output_meta=rd.get("artifact_meta_json") or "",
        output_dataset_bundle=run.get("dataset_bundle"),
    )
