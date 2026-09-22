from pydantic import BaseModel, Field
from typing import Optional


class ExtractRequest(BaseModel):
    """Single document extraction request."""

    file_path: str = Field(description="Path to the PDF or EPUB file")
    output_dir: Optional[str] = Field(
        None, description="Output directory override (deprecated)"
    )
    strip_refs: bool = Field(
        False, description="Strip model reference blocks from output"
    )
    export_parquet: bool = Field(
        False, description="Export trainable Parquet artifacts"
    )


class ExtractResponse(BaseModel):
    """Single PDF extraction response."""

    run_id: str
    document_id: str
    filename: str
    total_pages: int
    pages_pass: int
    pages_warn: int
    pages_fail: int
    pages_empty: int
    total_processing_time_ms: float
    output_raw_txt: str
    output_txt: str
    output_md: str
    output_meta: str
    output_dataset_bundle: Optional[str] = None


class JobRequest(BaseModel):
    """Batch extraction job request (compatibility wrapper around runs)."""

    file_paths: list[str] = Field(description="List of PDF/EPUB file paths to process")
    output_dir: Optional[str] = Field(
        None, description="Output directory override (deprecated)"
    )
    strip_refs: bool = Field(
        False, description="Strip model reference blocks (bounding boxes) from output"
    )
    export_parquet: bool = Field(
        False, description="Export a trainable Parquet bundle for the job"
    )
    name: Optional[str] = Field(None, description="Optional human-friendly name")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    progress: float
    documents_completed: int
    documents_total: int
    current_document: Optional[str] = None
    current_page: Optional[int] = None
    current_total_pages: Optional[int] = None
    pages_completed: int = 0
    pages_total: int = 0


class HealthResponse(BaseModel):
    status: str
    pipeline_version: str
    model_server_url: str
    model_name: str
    model_status: str
    input_dir: str = ""
    output_dir: str = ""


class FileInfo(BaseModel):
    name: str
    size: int
    modified: float
    path: str


class DocumentUpdate(BaseModel):
    display_title: Optional[str] = None
    group_path: Optional[str] = None
    author: Optional[str] = None
    work: Optional[str] = None
    book: Optional[str] = None
    document_date_label: Optional[str] = None
    document_date_precision: Optional[str] = None
    language: Optional[str] = None
    script: Optional[str] = None
    license: Optional[str] = None
    source_citation: Optional[str] = None
    notes: Optional[str] = None
    tags_json: Optional[str] = None


class BulkDocumentUpdate(BaseModel):
    document_ids: list[str]
    group_path: Optional[str] = None


class DocumentSummary(BaseModel):
    id: str
    filename: str
    display_title: str
    group_path: Optional[str] = None
    source_path: str
    file_sha256: str
    file_size_bytes: int
    total_pages: Optional[int] = None
    pdf_title: Optional[str] = None
    pdf_author: Optional[str] = None
    author: Optional[str] = None
    work: Optional[str] = None
    book: Optional[str] = None
    document_date_label: Optional[str] = None
    document_date_precision: Optional[str] = None
    language: Optional[str] = None
    script: Optional[str] = None
    license: Optional[str] = None
    source_citation: Optional[str] = None
    notes: Optional[str] = None
    tags_json: Optional[str] = None
    metadata_complete: bool = False
    latest_run_id: Optional[str] = None
    latest_run_status: Optional[str] = None


class StagedDocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_sha256: str
    deduped: bool
    estimated_pages: int


class RunCreateRequest(BaseModel):
    file_paths: list[str] = Field(description="PDF/EPUB file paths to enqueue")
    name: Optional[str] = None
    strip_refs: bool = False
    export_parquet: bool = True


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    documents_total: int
    pages_total_estimate: int
    documents: list[StagedDocumentInfo]


class RunSummary(BaseModel):
    id: str
    name: Optional[str] = None
    status: str
    stage: Optional[str] = None
    progress: float
    documents_total: int
    documents_completed: int
    pages_total: int
    pages_completed: int
    strip_refs: bool
    export_parquet: bool
    pipeline_version: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None
    dataset_bundle: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class RunDocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_sha256: str
    file_size_bytes: int
    status: str
    total_pages: Optional[int] = None
    pages_pass: int = 0
    pages_warn: int = 0
    pages_fail: int = 0
    pages_empty: int = 0
    total_processing_time_ms: float = 0.0
    total_tokens_cl100k: int = 0
    dominant_script: Optional[str] = None
    dominant_direction: Optional[str] = None
    languages_detected: list[str] = Field(default_factory=list)
    artifact_raw_txt: Optional[str] = None
    artifact_clean_txt: Optional[str] = None
    artifact_markdown: Optional[str] = None
    artifact_meta_json: Optional[str] = None
    artifact_source_pdf: Optional[str] = None
    source_run_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class RunDetail(RunSummary):
    documents: list[RunDocumentSummary] = Field(default_factory=list)


class PageSummary(BaseModel):
    page_num: int
    status: str
    validation_issues: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    script_direction: Optional[str] = None
    primary_script: Optional[str] = None
    detected_languages: list[str] = Field(default_factory=list)
    token_count_cl100k: Optional[int] = None
    text_length_chars: Optional[int] = None
    text_length_words: Optional[int] = None
    processing_time_ms: Optional[float] = None
    extraction_mode: Optional[str] = None
    extraction_attempt: Optional[int] = None
    dpi_used: Optional[int] = None
    has_embedded_text: Optional[bool] = None
    is_image_only: Optional[bool] = None


class RunDocumentDetail(RunDocumentSummary):
    pages: list[PageSummary] = Field(default_factory=list)


class HFPublishRequest(BaseModel):
    repo_id: str = Field(
        description="HuggingFace dataset repo (e.g. user/my-ocr-dataset)"
    )
    private: bool = False
    token: Optional[str] = Field(
        None, description="HF token; if absent, uses HF_TOKEN env"
    )
    commit_message: Optional[str] = None


class HFPublishResponse(BaseModel):
    repo_id: str
    repo_url: str
    commit_url: Optional[str] = None
    files_uploaded: list[str]
