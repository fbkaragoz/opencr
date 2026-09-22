from fastapi.testclient import TestClient
from pathlib import Path

import ocr_pipeline.main as main_module
from ocr_pipeline.config import settings
from ocr_pipeline.services.startup import model_readiness
from tests.test_epub_parser import write_minimal_epub


def test_upload_and_list_input(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    runs_dir = output_dir / "runs"
    db_path = output_dir / "opencr.sqlite"
    input_dir.mkdir()
    output_dir.mkdir()

    async def fake_wait_for_model_server():
        model_readiness.ready = True
        model_readiness.error = None
        return True

    monkeypatch.setattr(settings, "input_dir", input_dir)
    monkeypatch.setattr(settings, "output_dir", output_dir)
    monkeypatch.setattr(settings, "runs_dir", runs_dir)
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(
        main_module, "wait_for_model_server", fake_wait_for_model_server
    )
    model_readiness.ready = True
    model_readiness.error = None

    with TestClient(main_module.app) as client:
        files = {"file": ("sample.pdf", b"%PDF-1.4\n", "application/pdf")}
        resp = client.post("/api/upload", files=files)
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["filename"] == "sample.pdf"
        assert payload["size"] > 0

        listing = client.get("/api/files/input")
        assert listing.status_code == 200
        items = listing.json()
        assert any(item["name"] == "sample.pdf" for item in items)

        documents = client.get("/api/documents")
        assert documents.status_code == 200
        docs = documents.json()
        assert docs[0]["filename"] == "sample.pdf"

        patch = client.patch(
            f"/api/documents/{docs[0]['id']}",
            json={
                "author": "Evliyâ Çelebi",
                "group_path": "Ottoman/Seyahatname",
                "work": "Seyahatnâme",
                "book": "1",
                "document_date_label": "1900s",
                "document_date_precision": "century",
                "language": "ota-Latn,tr",
                "script": "latin_extended",
                "license": "cc-by-4.0",
            },
        )
        assert patch.status_code == 200
        assert patch.json()["metadata_complete"] is True
        assert patch.json()["group_path"] == "Ottoman/Seyahatname"

        bulk = client.patch(
            "/api/documents/bulk",
            json={"document_ids": [docs[0]["id"]], "group_path": "Grouped/Batch"},
        )
        assert bulk.status_code == 200
        assert bulk.json()[0]["group_path"] == "Grouped/Batch"


def test_duplicate_upload_filenames_keep_distinct_source_paths(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    runs_dir = output_dir / "runs"
    db_path = output_dir / "opencr.sqlite"
    input_dir.mkdir()
    output_dir.mkdir()

    async def fake_wait_for_model_server():
        model_readiness.ready = True
        return True

    monkeypatch.setattr(settings, "input_dir", input_dir)
    monkeypatch.setattr(settings, "output_dir", output_dir)
    monkeypatch.setattr(settings, "runs_dir", runs_dir)
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(
        main_module, "wait_for_model_server", fake_wait_for_model_server
    )
    model_readiness.ready = True

    with TestClient(main_module.app) as client:
        first = client.post(
            "/api/upload",
            files={"file": ("sample.pdf", b"%PDF-1.4 first\n", "application/pdf")},
        )
        second = client.post(
            "/api/upload",
            files={"file": ("sample.pdf", b"%PDF-1.4 second\n", "application/pdf")},
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["path"] != second.json()["path"]
        assert Path(first.json()["path"]).exists()
        assert Path(second.json()["path"]).exists()

        docs = client.get("/api/documents").json()
        assert len({doc["source_path"] for doc in docs}) == 2


def test_upload_and_list_epub_input(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    runs_dir = output_dir / "runs"
    db_path = output_dir / "opencr.sqlite"
    input_dir.mkdir()
    output_dir.mkdir()
    epub_path = tmp_path / "sample.epub"
    write_minimal_epub(epub_path)

    async def fake_wait_for_model_server():
        model_readiness.ready = True
        return True

    monkeypatch.setattr(settings, "input_dir", input_dir)
    monkeypatch.setattr(settings, "output_dir", output_dir)
    monkeypatch.setattr(settings, "runs_dir", runs_dir)
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(
        main_module, "wait_for_model_server", fake_wait_for_model_server
    )
    model_readiness.ready = True

    with TestClient(main_module.app) as client:
        resp = client.post(
            "/api/upload",
            files={
                "file": (
                    "sample.epub",
                    epub_path.read_bytes(),
                    "application/epub+zip",
                )
            },
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "sample.epub"

        listing = client.get("/api/files/input")
        assert listing.status_code == 200
        assert any(item["name"] == "sample.epub" for item in listing.json())

        docs = client.get("/api/documents").json()
        epub_doc = next(doc for doc in docs if doc["filename"] == "sample.epub")
        assert epub_doc["total_pages"] == 2


def test_runs_list_empty_when_no_runs(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    async def fake_wait_for_model_server():
        model_readiness.ready = True
        return True

    monkeypatch.setattr(settings, "input_dir", tmp_path / "input")
    monkeypatch.setattr(settings, "output_dir", output_dir)
    monkeypatch.setattr(settings, "runs_dir", output_dir / "runs")
    monkeypatch.setattr(settings, "db_path", output_dir / "opencr.sqlite")
    monkeypatch.setattr(
        main_module, "wait_for_model_server", fake_wait_for_model_server
    )
    model_readiness.ready = True

    with TestClient(main_module.app) as client:
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []


def test_run_detail_uses_minimal_terminal_progress():
    repo_root = Path(__file__).parents[1]
    html = (repo_root / "ocr_pipeline/static/index.html").read_text(encoding="utf-8")
    app_js = (repo_root / "ocr_pipeline/static/js/app.js").read_text(encoding="utf-8")

    assert 'class="run-terminal"' in html
    assert 'x-text="runDocumentProgressLabel()"' in html
    assert 'class="run-spinner"' in html
    assert 'class="summary-card"' not in html
    assert 'class="heatmap"' not in html
    assert "currentRunDocument()" in app_js
    assert "runDocumentProgressLabel()" in app_js


def test_home_uses_document_workbench():
    repo_root = Path(__file__).parents[1]
    html = (repo_root / "ocr_pipeline/static/index.html").read_text(encoding="utf-8")
    app_js = (repo_root / "ocr_pipeline/static/js/app.js").read_text(encoding="utf-8")

    assert 'class="document-workbench"' in html
    assert "document_date_label" in html
    assert "document_date_precision" in html
    assert "group_path" in html
    assert "OCR snapshot" in html
    assert "OCR pairs" in html
    assert "Text bundle" in html
    assert "selectedDocumentIds" in app_js
    assert "availableDocumentGroups()" in app_js
    assert "filteredDocuments()" in app_js
    assert "groupedDocuments()" in app_js
    assert "applyBulkGroup()" in app_js
    assert "downloadOCRPairs()" in app_js
    assert "downloadTextBundle()" in app_js
    assert "selectedRunDocumentIds" in app_js
    assert "documentProcessLabel" in app_js
    assert "retryRun()" in app_js
    assert "Retry incomplete" in html
    assert "selectedPageText()" in app_js
    assert "currentPageQualityFlags()" in app_js
    assert "Quality flags" in html
    assert "saveSelectedDocument()" in app_js


def test_home_exposes_profile_and_benchmark_views():
    repo_root = Path(__file__).parents[1]
    html = (repo_root / "ocr_pipeline/static/index.html").read_text(encoding="utf-8")
    app_js = (repo_root / "ocr_pipeline/static/js/app.js").read_text(encoding="utf-8")

    assert 'class="view-nav"' in html
    assert "Profiles" in html
    assert "Benchmarks" in html
    assert "DeepSeek-OCR-2" in html
    assert "activeView" in app_js
    assert "extractionProfiles" in app_js
    assert "benchmarkRows" in app_js
