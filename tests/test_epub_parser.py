import asyncio
import hashlib
from pathlib import Path
from zipfile import ZipFile

from ocr_pipeline.services.batch_processor import BatchProcessor
from ocr_pipeline.services.epub_parser import EpubParser
from ocr_pipeline.services.run_storage import RunStorage


def write_minimal_epub(path: Path) -> None:
    with ZipFile(path, "w") as zf:
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>
            """,
        )
        zf.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="utf-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>Deneme Kitabi</dc:title>
                <dc:creator>Acme Yazar</dc:creator>
                <dc:language>tr</dc:language>
              </metadata>
              <manifest>
                <item id="c1" href="chapters/chapter1.xhtml" media-type="application/xhtml+xml"/>
                <item id="c2" href="chapters/chapter2.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="c1"/>
                <itemref idref="c2"/>
              </spine>
            </package>
            """,
        )
        zf.writestr(
            "OEBPS/chapters/chapter1.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml">
              <head><title>Ignored</title></head>
              <body><h1>Birinci Bolum</h1><p>Merhaba dunya.</p></body>
            </html>""",
        )
        zf.writestr(
            "OEBPS/chapters/chapter2.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml">
              <body><h2>Ikinci Bolum</h2><p>Turkce metin cikiyor.</p></body>
            </html>""",
        )


def test_epub_parser_extracts_spine_sections(tmp_path):
    epub_path = tmp_path / "sample.epub"
    write_minimal_epub(epub_path)

    parsed = EpubParser().parse(epub_path)

    assert parsed.title == "Deneme Kitabi"
    assert parsed.creators == ["Acme Yazar"]
    assert parsed.language == "tr"
    assert len(parsed.sections) == 2
    assert "Birinci Bolum" in parsed.sections[0].text
    assert "Turkce metin cikiyor." in parsed.sections[1].text


def test_batch_processor_extracts_epub_without_ocr(tmp_path):
    epub_path = tmp_path / "sample.epub"
    write_minimal_epub(epub_path)
    output = tmp_path / "output"
    storage = RunStorage(output_root=output, runs_root=output / "runs")
    storage.ensure_run_dirs("run1")
    paths = storage.artifact_paths("run1", "doc1", "sample.epub")

    class FakeDB:
        def __init__(self):
            self.pages = []
            self.document_updates = []

        async def update_run_document(self, *args, **kwargs):
            self.document_updates.append((args, kwargs))

        async def upsert_page(self, *args, **kwargs):
            self.pages.append((args, kwargs))

    db = FakeDB()
    sha = hashlib.sha256(epub_path.read_bytes()).hexdigest()

    meta = asyncio.run(
        BatchProcessor(db).process_document(
            epub_path,
            run_id="run1",
            document_id="doc1",
            file_sha256=sha,
            filename="sample.epub",
            artifact_paths=paths,
        )
    )

    assert meta.model_used == "epub-direct-text"
    assert meta.total_pages == 2
    assert len(db.pages) == 2
    assert paths.markdown.exists()
    assert "Turkce metin cikiyor." in paths.clean_txt.read_text(encoding="utf-8")
