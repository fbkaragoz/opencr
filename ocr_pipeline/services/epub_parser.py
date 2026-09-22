import posixpath
import re
import zipfile
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET


class EpubParseError(ValueError):
    """Raised when an EPUB cannot be parsed enough for text extraction."""


@dataclass(frozen=True)
class EpubSection:
    index: int
    href: str
    title: str | None
    text: str


@dataclass(frozen=True)
class EpubDocument:
    title: str | None
    creators: list[str]
    language: str | None
    sections: list[EpubSection]


class _HTMLToText(HTMLParser):
    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
    SKIP_TAGS = {"head", "script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)

    def text(self) -> str:
        text = unescape("".join(self._parts))
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t\r\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class EpubParser:
    """Extracts readable text from EPUB spine documents using stdlib parsers."""

    CONTAINER_PATH = "META-INF/container.xml"
    HTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html"}

    def parse(self, path: Path) -> EpubDocument:
        try:
            with zipfile.ZipFile(path) as zf:
                opf_path = self._opf_path(zf)
                opf_root = ET.fromstring(self._read_text(zf, opf_path))
                metadata = self._metadata(opf_root)
                sections = self._sections(zf, opf_path, opf_root)
        except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
            raise EpubParseError(f"Invalid EPUB: {path}") from exc

        if not sections:
            raise EpubParseError(f"No readable text sections found in EPUB: {path}")
        return EpubDocument(
            title=metadata.get("title"),
            creators=metadata.get("creators", []),
            language=metadata.get("language"),
            sections=sections,
        )

    def count_sections(self, path: Path) -> int:
        return len(self.parse(path).sections)

    def _opf_path(self, zf: zipfile.ZipFile) -> str:
        if self.CONTAINER_PATH not in zf.namelist():
            raise EpubParseError("EPUB container.xml not found")
        root = ET.fromstring(self._read_text(zf, self.CONTAINER_PATH))
        rootfile = root.find(".//{*}rootfile")
        full_path = rootfile.get("full-path") if rootfile is not None else None
        if not full_path:
            raise EpubParseError("EPUB rootfile missing")
        return full_path

    def _metadata(self, opf_root: ET.Element) -> dict:
        title = self._first_text(opf_root, "title")
        creators = [text for text in self._all_text(opf_root, "creator") if text]
        language = self._first_text(opf_root, "language")
        return {"title": title, "creators": creators, "language": language}

    def _sections(
        self, zf: zipfile.ZipFile, opf_path: str, opf_root: ET.Element
    ) -> list[EpubSection]:
        manifest = {}
        for item in opf_root.findall(".//{*}manifest/{*}item"):
            item_id = item.get("id")
            href = item.get("href")
            if item_id and href:
                manifest[item_id] = {
                    "href": href,
                    "media_type": item.get("media-type", ""),
                }

        ordered_items = []
        for itemref in opf_root.findall(".//{*}spine/{*}itemref"):
            idref = itemref.get("idref")
            if idref and idref in manifest:
                ordered_items.append(manifest[idref])
        if not ordered_items:
            ordered_items = list(manifest.values())

        base = posixpath.dirname(opf_path)
        sections: list[EpubSection] = []
        for item in ordered_items:
            href = str(item["href"]).split("#", 1)[0]
            media_type = str(item["media_type"])
            if not self._is_html_item(href, media_type):
                continue
            zip_path = posixpath.normpath(posixpath.join(base, unquote(href)))
            if zip_path not in zf.namelist():
                continue
            text = self._html_to_text(self._read_text(zf, zip_path))
            if not text:
                continue
            sections.append(
                EpubSection(
                    index=len(sections) + 1,
                    href=zip_path,
                    title=self._section_title(text),
                    text=text,
                )
            )
        return sections

    def _is_html_item(self, href: str, media_type: str) -> bool:
        suffix = Path(href).suffix.lower()
        return media_type in self.HTML_MEDIA_TYPES or suffix in {".xhtml", ".html", ".htm"}

    def _read_text(self, zf: zipfile.ZipFile, path: str) -> str:
        data = zf.read(path)
        for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def _html_to_text(self, html: str) -> str:
        parser = _HTMLToText()
        parser.feed(html)
        parser.close()
        return parser.text()

    def _section_title(self, text: str) -> str | None:
        for line in text.splitlines():
            candidate = line.strip()
            if candidate:
                return candidate[:120]
        return None

    def _first_text(self, root: ET.Element, local_name: str) -> str | None:
        values = self._all_text(root, local_name)
        return values[0] if values else None

    def _all_text(self, root: ET.Element, local_name: str) -> list[str]:
        values = []
        for elem in root.iter():
            if elem.tag.rsplit("}", 1)[-1] == local_name and elem.text:
                text = elem.text.strip()
                if text:
                    values.append(text)
        return values
