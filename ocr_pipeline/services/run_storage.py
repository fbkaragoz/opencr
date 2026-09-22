from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    raw_txt: Path
    clean_txt: Path
    markdown: Path
    meta_json: Path
    source_pdf: Path


class RunStorage:
    """Resolves filesystem paths for run-scoped artifacts.

    Layout:
        {output_root}/sources/{document_id}.{pdf,epub} (shared, content-addressed)
        {output_root}/runs/{run_id}/
            artifacts/{document_id}__{stem}.{raw.txt,txt,md,meta.json}
            dataset/{pages.parquet,documents.parquet,manifest.json,bundle.zip}
    """

    def __init__(self, output_root: Path, runs_root: Path):
        self.output_root = output_root
        self.runs_root = runs_root

    def sources_dir(self) -> Path:
        return self.output_root / "sources"

    def source_pdf_path(self, document_id: str) -> Path:
        return self.sources_dir() / f"{document_id}.pdf"

    def source_document_path(self, document_id: str, suffix: str) -> Path:
        clean_suffix = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        if clean_suffix not in {".pdf", ".epub"}:
            clean_suffix = ".pdf"
        return self.sources_dir() / f"{document_id}{clean_suffix}"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def artifacts_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts"

    def dataset_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "dataset"

    def ensure_run_dirs(self, run_id: str) -> None:
        self.sources_dir().mkdir(parents=True, exist_ok=True)
        self.artifacts_dir(run_id).mkdir(parents=True, exist_ok=True)
        self.dataset_dir(run_id).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_stem(filename: str) -> str:
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(filename).stem)
        return stem or "document"

    def artifact_paths(self, run_id: str, document_id: str, filename: str) -> ArtifactPaths:
        base = f"{document_id}__{self._safe_stem(filename)}"
        artifacts = self.artifacts_dir(run_id)
        source_suffix = Path(filename).suffix.lower() or ".pdf"
        return ArtifactPaths(
            raw_txt=artifacts / f"{base}.raw.txt",
            clean_txt=artifacts / f"{base}.txt",
            markdown=artifacts / f"{base}.md",
            meta_json=artifacts / f"{base}.meta.json",
            source_pdf=self.source_document_path(document_id, source_suffix),
        )
