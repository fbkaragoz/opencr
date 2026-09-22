"""Static-friendly endpoints for input file management. Output/dataset listing
moved to /api/runs."""

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from ocr_pipeline.config import settings
from ocr_pipeline.models.schemas import FileInfo
from ocr_pipeline.services.db import get_db
from ocr_pipeline.services.document_catalog import (
    SUPPORTED_SOURCE_SUFFIXES,
    catalog_document,
    is_supported_source,
)


router = APIRouter()


@router.post("/api/upload")
async def upload_pdf(file: UploadFile):
    """Accept a multipart PDF/EPUB upload and save to input directory."""
    if not file.filename or Path(file.filename).suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only PDF and EPUB files are accepted")

    safe_name = Path(file.filename).name
    if ".." in safe_name or "/" in safe_name or "\\" in safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    settings.input_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    digest = hashlib.sha256(content).hexdigest()
    dest = settings.input_dir / f"{digest[:16]}__{safe_name}"
    dest.write_bytes(content)
    await catalog_document(get_db(), dest, filename=safe_name)

    return {
        "filename": safe_name,
        "stored_filename": dest.name,
        "size": len(content),
        "path": str(dest),
    }


@router.get("/api/files/input", response_model=list[FileInfo])
async def list_input_files():
    """List supported source files in the input directory."""
    input_dir = settings.input_dir
    if not input_dir.exists():
        return []

    documents_by_path = {
        doc["source_path"]: doc for doc in await get_db().list_documents(limit=1000)
    }
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and is_supported_source(p):
            stat = p.stat()
            document = documents_by_path.get(str(p))
            files.append(
                FileInfo(
                    name=document["filename"] if document else p.name,
                    size=stat.st_size,
                    modified=stat.st_mtime,
                    path=str(p),
                )
            )
    return files
