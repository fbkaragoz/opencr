import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from ocr_pipeline.config import settings
from ocr_pipeline.models.metadata import DocumentMetadata, PageMetadata
from ocr_pipeline.services.db import Database
from ocr_pipeline.services.epub_parser import EpubParser
from ocr_pipeline.services.metadata_collector import MetadataCollector
from ocr_pipeline.services.ocr_engine import OCREngine
from ocr_pipeline.services.output_validator import OutputValidator, ValidationStatus
from ocr_pipeline.services.output_writer import OutputWriter
from ocr_pipeline.services.page_analyzer import PageAnalyzer, PageProfile
from ocr_pipeline.services.pdf_renderer import PDFRenderer
from ocr_pipeline.services.run_storage import ArtifactPaths
from ocr_pipeline.services.script_detector import ScriptAnalysis, ScriptDetector, ScriptDirection
from ocr_pipeline.services.text_cleaner import TextCleaner


EventCallback = Callable[[dict], Awaitable[None]]

# OCR retry strategies, ordered: structured first, then progressively
# simpler so a looping page can still produce text.
RETRY_STRATEGIES: tuple[dict, ...] = (
    {"mode": "markdown", "ngram_size": 30, "window_size": 90},
    {"mode": "free_ocr", "ngram_size": 30, "window_size": 90},
    {"mode": "free_ocr", "ngram_size": 8, "window_size": 256},
)

PAGE_DB_FIELDS = (
    "validation_issues", "script_direction", "primary_script", "detected_languages",
    "token_count_cl100k", "text_length_chars", "text_length_words",
    "extraction_mode", "extraction_attempt", "dpi_used", "quality_flags",
    "has_embedded_text", "is_image_only",
)


class BatchProcessor:
    """Processes a single PDF page-by-page with retry, validation, script
    detection and metadata collection. Pages run concurrently, bounded by
    settings.batch_concurrency."""

    def __init__(
        self,
        db: Database,
        *,
        event_callback: EventCallback | None = None,
        strip_refs: bool = False,
        page_concurrency: int | None = None,
    ):
        self.db = db
        self.ocr_engine = OCREngine()
        self.renderer = PDFRenderer()
        self.analyzer = PageAnalyzer()
        self.validator = OutputValidator()
        self.cleaner = TextCleaner()
        self.script_detector = ScriptDetector()
        self.metadata_collector = MetadataCollector(model_name=settings.model_name)
        self.writer = OutputWriter()
        self.event_callback = event_callback
        self.strip_refs = strip_refs
        self.page_concurrency = max(1, page_concurrency or settings.batch_concurrency)

    async def _emit(self, event: dict) -> None:
        if self.event_callback:
            await self.event_callback(event)

    async def extract_with_retry(
        self,
        image,
        page_num: int,
        max_retries: int = 2,
    ) -> tuple[str, str, int, str]:
        clean_text = raw_text = ""
        attempt = 0
        strategies = RETRY_STRATEGIES[: max_retries + 1]

        for attempt, strategy in enumerate(strategies):
            raw_text = await self.ocr_engine.extract_page(
                image,
                mode=strategy["mode"],
                ngram_size=strategy["ngram_size"],
                window_size=strategy["window_size"],
            )
            clean_text = self.cleaner.clean(raw_text, strip_refs=self.strip_refs)
            result = self.validator.validate(clean_text, page_num)

            if result.status in (ValidationStatus.PASS, ValidationStatus.WARN):
                if attempt > 0:
                    result.issues.append(
                        f"Succeeded on attempt {attempt + 1} with strategy {strategy['mode']}"
                    )
                break

            if attempt < len(strategies) - 1:
                await self._emit({
                    "type": "page_retry",
                    "page": page_num,
                    "attempt": attempt + 2,
                    "reason": "; ".join(result.issues),
                    "new_strategy": strategies[attempt + 1]["mode"],
                })
        else:
            attempt = len(strategies) - 1

        return (
            self.cleaner.clean_fidelity(raw_text, strip_refs=self.strip_refs),
            clean_text,
            attempt + 1,
            strategies[attempt]["mode"],
        )

    async def _process_page(
        self,
        run_id: str,
        document_id: str,
        filename: str,
        pdf_path: Path,
        profile: PageProfile,
        total_pages: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, str, PageMetadata, ScriptAnalysis, float]:
        async with semaphore:
            page_num = profile.page_num
            await self._emit({
                "type": "page_start",
                "document": filename,
                "document_id": document_id,
                "page": page_num,
                "total_pages": total_pages,
                "dpi": profile.recommended_dpi,
                "mode": profile.recommended_mode,
            })

            image = await asyncio.to_thread(
                self.renderer.render_page, pdf_path, page_num, profile.recommended_dpi
            )
            t0 = time.perf_counter()
            raw_text, clean_text, attempt, mode_used = await self.extract_with_retry(
                image, page_num, max_retries=settings.max_retries
            )
            processing_ms = (time.perf_counter() - t0) * 1000

            validation = self.validator.validate(clean_text, page_num)
            script_analysis = self.script_detector.analyze_text(clean_text)
            page_meta = self.metadata_collector.build_page_metadata(
                page_num=page_num,
                text=clean_text,
                processing_time_ms=processing_ms,
                mode=mode_used,
                attempt=attempt,
                dpi=profile.recommended_dpi,
                script_analysis=script_analysis,
                validation_result=validation,
                page_profile=profile,
            )

            db_fields = {f: getattr(page_meta, f) for f in PAGE_DB_FIELDS}
            await self.db.upsert_page(
                run_id, document_id, page_num,
                status=page_meta.validation_status,
                processing_time_ms=processing_ms,
                **db_fields,
            )

            await self._emit({
                "type": "page_complete",
                "document": filename,
                "document_id": document_id,
                "page": page_num,
                "total_pages": total_pages,
                "processing_time_ms": round(processing_ms, 1),
                "validation_status": validation.status.value,
                "script_direction": script_analysis.direction.value,
                "primary_script": script_analysis.primary_script.value,
                "text_length": len(clean_text),
                "token_count": page_meta.token_count_cl100k,
            })
            return raw_text, clean_text, page_meta, script_analysis, processing_ms

    async def process_document(
        self,
        pdf_path: Path,
        *,
        run_id: str,
        document_id: str,
        file_sha256: str,
        filename: str | None = None,
        artifact_paths: ArtifactPaths,
    ) -> DocumentMetadata:
        if pdf_path.suffix.lower() == ".epub":
            return await self.process_epub_document(
                pdf_path,
                run_id=run_id,
                document_id=document_id,
                file_sha256=file_sha256,
                filename=filename,
                artifact_paths=artifact_paths,
            )

        filename = filename or pdf_path.name
        file_size = (await asyncio.to_thread(pdf_path.stat)).st_size
        started_at = datetime.now(timezone.utc).isoformat()

        page_profiles = await asyncio.to_thread(self.analyzer.analyze_document, pdf_path)
        total_pages = len(page_profiles)
        pdf_meta = await asyncio.to_thread(self.metadata_collector.extract_pdf_metadata, pdf_path)

        await self.db.update_run_document(
            run_id, document_id,
            status="processing", total_pages=total_pages, started_at=started_at,
        )

        semaphore = asyncio.Semaphore(self.page_concurrency)
        results = await asyncio.gather(*(
            self._process_page(run_id, document_id, filename, pdf_path, p, total_pages, semaphore)
            for p in page_profiles
        ))

        raw_pages, clean_pages, pages_meta, pages_script, page_times = (list(c) for c in zip(*results)) if results else ([], [], [], [], [])
        total_processing_ms = sum(page_times)
        completed_at = datetime.now(timezone.utc).isoformat()

        status_counts = Counter(m.validation_status for m in pages_meta)
        direction_counts = Counter(m.script_direction for m in pages_meta)
        script_counts = Counter(m.primary_script for m in pages_meta)
        all_langs = sorted({lang for m in pages_meta for lang in m.detected_languages})

        doc_metadata = DocumentMetadata(
            filename=filename,
            file_path=str(pdf_path),
            file_size_bytes=file_size,
            file_sha256=file_sha256,
            total_pages=total_pages,
            pdf_title=pdf_meta.get("title"),
            pdf_author=pdf_meta.get("author"),
            pdf_creation_date=pdf_meta.get("creation_date"),
            pdf_producer=pdf_meta.get("producer"),
            total_chars=sum(m.text_length_chars for m in pages_meta),
            total_words=sum(m.text_length_words for m in pages_meta),
            total_tokens_cl100k=sum(m.token_count_cl100k for m in pages_meta),
            total_processing_time_ms=round(total_processing_ms, 1),
            avg_time_per_page_ms=round(total_processing_ms / total_pages, 1) if total_pages else 0,
            dominant_direction=(direction_counts.most_common(1) or [(ScriptDirection.LTR.value, 0)])[0][0],
            dominant_script=(script_counts.most_common(1) or [("unknown", 0)])[0][0],
            pages_ltr=direction_counts.get(ScriptDirection.LTR.value, 0),
            pages_rtl=direction_counts.get(ScriptDirection.RTL.value, 0),
            pages_mixed=direction_counts.get(ScriptDirection.MIXED.value, 0),
            languages_detected=all_langs,
            pages_pass=status_counts.get(ValidationStatus.PASS.value, 0),
            pages_warn=status_counts.get(ValidationStatus.WARN.value, 0),
            pages_fail=status_counts.get(ValidationStatus.FAIL.value, 0),
            pages_empty=status_counts.get(ValidationStatus.EMPTY.value, 0),
            started_at=started_at,
            completed_at=completed_at,
            pipeline_version=settings.pipeline_version,
            model_used=settings.model_name,
            pages=pages_meta,
        )

        await asyncio.to_thread(
            self.writer.write_all,
            artifact_paths, raw_pages, clean_pages, pages_meta, pages_script, doc_metadata,
        )

        await self.db.update_run_document(
            run_id, document_id,
            status="completed",
            pages_pass=doc_metadata.pages_pass,
            pages_warn=doc_metadata.pages_warn,
            pages_fail=doc_metadata.pages_fail,
            pages_empty=doc_metadata.pages_empty,
            total_processing_time_ms=doc_metadata.total_processing_time_ms,
            total_tokens_cl100k=doc_metadata.total_tokens_cl100k,
            dominant_script=doc_metadata.dominant_script,
            dominant_direction=doc_metadata.dominant_direction,
            languages_detected=json.dumps(all_langs, ensure_ascii=False),
            artifact_raw_txt=str(artifact_paths.raw_txt),
            artifact_clean_txt=str(artifact_paths.clean_txt),
            artifact_markdown=str(artifact_paths.markdown),
            artifact_meta_json=str(artifact_paths.meta_json),
            artifact_source_pdf=str(artifact_paths.source_pdf),
            completed_at=completed_at,
        )

        await self._emit({
            "type": "document_complete",
            "document": filename,
            "document_id": document_id,
            "total_pages": total_pages,
            "pages_pass": doc_metadata.pages_pass,
            "pages_warn": doc_metadata.pages_warn,
            "pages_fail": doc_metadata.pages_fail,
            "total_time_ms": doc_metadata.total_processing_time_ms,
            "output_path": str(artifact_paths.markdown),
        })
        return doc_metadata

    async def process_epub_document(
        self,
        epub_path: Path,
        *,
        run_id: str,
        document_id: str,
        file_sha256: str,
        filename: str | None = None,
        artifact_paths: ArtifactPaths,
    ) -> DocumentMetadata:
        filename = filename or epub_path.name
        file_size = (await asyncio.to_thread(epub_path.stat)).st_size
        started_at = datetime.now(timezone.utc).isoformat()
        epub_doc = await asyncio.to_thread(EpubParser().parse, epub_path)
        total_pages = len(epub_doc.sections)

        await self.db.update_run_document(
            run_id, document_id,
            status="processing", total_pages=total_pages, started_at=started_at,
        )

        raw_pages: list[str] = []
        clean_pages: list[str] = []
        pages_meta: list[PageMetadata] = []
        pages_script: list[ScriptAnalysis] = []
        page_times: list[float] = []

        for section in epub_doc.sections:
            await self._emit({
                "type": "page_start",
                "document": filename,
                "document_id": document_id,
                "page": section.index,
                "total_pages": total_pages,
                "dpi": 0,
                "mode": "epub_text",
            })
            t0 = time.perf_counter()
            raw_text = section.text
            clean_text = self.cleaner.clean(raw_text, strip_refs=self.strip_refs)
            processing_ms = (time.perf_counter() - t0) * 1000
            validation = self.validator.validate(clean_text, section.index)
            script_analysis = self.script_detector.analyze_text(clean_text)
            profile = PageProfile(
                page_num=section.index,
                has_embedded_text=True,
                embedded_text_length=len(raw_text),
                has_images=False,
                image_count=0,
                width=0,
                height=0,
                is_landscape=False,
                estimated_complexity="epub",
                recommended_dpi=0,
                recommended_mode="epub_text",
            )
            page_meta = self.metadata_collector.build_page_metadata(
                page_num=section.index,
                text=clean_text,
                processing_time_ms=processing_ms,
                mode="epub_text",
                attempt=1,
                dpi=0,
                script_analysis=script_analysis,
                validation_result=validation,
                page_profile=profile,
            )

            db_fields = {f: getattr(page_meta, f) for f in PAGE_DB_FIELDS}
            await self.db.upsert_page(
                run_id, document_id, section.index,
                status=page_meta.validation_status,
                processing_time_ms=processing_ms,
                **db_fields,
            )

            await self._emit({
                "type": "page_complete",
                "document": filename,
                "document_id": document_id,
                "page": section.index,
                "total_pages": total_pages,
                "processing_time_ms": round(processing_ms, 1),
                "validation_status": validation.status.value,
                "script_direction": script_analysis.direction.value,
                "primary_script": script_analysis.primary_script.value,
                "text_length": len(clean_text),
                "token_count": page_meta.token_count_cl100k,
            })
            raw_pages.append(raw_text)
            clean_pages.append(clean_text)
            pages_meta.append(page_meta)
            pages_script.append(script_analysis)
            page_times.append(processing_ms)

        total_processing_ms = sum(page_times)
        completed_at = datetime.now(timezone.utc).isoformat()
        status_counts = Counter(m.validation_status for m in pages_meta)
        direction_counts = Counter(m.script_direction for m in pages_meta)
        script_counts = Counter(m.primary_script for m in pages_meta)
        all_langs = sorted({lang for m in pages_meta for lang in m.detected_languages})

        doc_metadata = DocumentMetadata(
            filename=filename,
            file_path=str(epub_path),
            file_size_bytes=file_size,
            file_sha256=file_sha256,
            total_pages=total_pages,
            pdf_title=epub_doc.title,
            pdf_author=", ".join(epub_doc.creators) if epub_doc.creators else None,
            pdf_creation_date=None,
            pdf_producer="EPUB direct text extraction",
            total_chars=sum(m.text_length_chars for m in pages_meta),
            total_words=sum(m.text_length_words for m in pages_meta),
            total_tokens_cl100k=sum(m.token_count_cl100k for m in pages_meta),
            total_processing_time_ms=round(total_processing_ms, 1),
            avg_time_per_page_ms=round(total_processing_ms / total_pages, 1) if total_pages else 0,
            dominant_direction=(direction_counts.most_common(1) or [(ScriptDirection.LTR.value, 0)])[0][0],
            dominant_script=(script_counts.most_common(1) or [("unknown", 0)])[0][0],
            pages_ltr=direction_counts.get(ScriptDirection.LTR.value, 0),
            pages_rtl=direction_counts.get(ScriptDirection.RTL.value, 0),
            pages_mixed=direction_counts.get(ScriptDirection.MIXED.value, 0),
            languages_detected=all_langs,
            pages_pass=status_counts.get(ValidationStatus.PASS.value, 0),
            pages_warn=status_counts.get(ValidationStatus.WARN.value, 0),
            pages_fail=status_counts.get(ValidationStatus.FAIL.value, 0),
            pages_empty=status_counts.get(ValidationStatus.EMPTY.value, 0),
            started_at=started_at,
            completed_at=completed_at,
            pipeline_version=settings.pipeline_version,
            model_used="epub-direct-text",
            pages=pages_meta,
        )

        await asyncio.to_thread(
            self.writer.write_all,
            artifact_paths, raw_pages, clean_pages, pages_meta, pages_script, doc_metadata,
        )

        await self.db.update_run_document(
            run_id, document_id,
            status="completed",
            pages_pass=doc_metadata.pages_pass,
            pages_warn=doc_metadata.pages_warn,
            pages_fail=doc_metadata.pages_fail,
            pages_empty=doc_metadata.pages_empty,
            total_processing_time_ms=doc_metadata.total_processing_time_ms,
            total_tokens_cl100k=doc_metadata.total_tokens_cl100k,
            dominant_script=doc_metadata.dominant_script,
            dominant_direction=doc_metadata.dominant_direction,
            languages_detected=json.dumps(all_langs, ensure_ascii=False),
            artifact_raw_txt=str(artifact_paths.raw_txt),
            artifact_clean_txt=str(artifact_paths.clean_txt),
            artifact_markdown=str(artifact_paths.markdown),
            artifact_meta_json=str(artifact_paths.meta_json),
            artifact_source_pdf=str(artifact_paths.source_pdf),
            completed_at=completed_at,
        )

        await self._emit({
            "type": "document_complete",
            "document": filename,
            "document_id": document_id,
            "total_pages": total_pages,
            "pages_pass": doc_metadata.pages_pass,
            "pages_warn": doc_metadata.pages_warn,
            "pages_fail": doc_metadata.pages_fail,
            "total_time_ms": doc_metadata.total_processing_time_ms,
            "output_path": str(artifact_paths.markdown),
        })
        return doc_metadata
