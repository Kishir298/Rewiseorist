"""Resource import, text extraction, chunking, grouping and retrieval."""

from __future__ import annotations

import hashlib
import html
import io
import os
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

import storage

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FOLDER_FILES = 50
MAX_FOLDER_BYTES = 200 * 1024 * 1024

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".vtt", ".srt"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | MEDIA_EXTENSIONS

CHUNK_TARGET = 700
CHUNK_OVERLAP = 120


class ResourceError(ValueError):
    """A user-facing import error."""


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def validate_single_file(name: str, size: int) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ResourceError(f"Unsupported file type '{suffix or '(none)'}'. Supported: {supported}")
    if size > MAX_FILE_BYTES:
        raise ResourceError(
            f"'{name}' is {size / (1024 * 1024):.1f} MB. The individual file limit is 20 MB."
        )
    return suffix


def validate_folder(files: Sequence[tuple[str, int]]) -> None:
    if len(files) > MAX_FOLDER_FILES:
        raise ResourceError(
            f"This folder has {len(files)} files. The folder limit is {MAX_FOLDER_FILES} files."
        )
    total = sum(size for _, size in files)
    if total > MAX_FOLDER_BYTES:
        raise ResourceError(
            f"The folder is {total / (1024 * 1024):.1f} MB. The combined limit is about 200 MB."
        )
    for name, size in files:
        validate_single_file(name, size)


def safe_relative_path(name: str) -> str:
    raw = name.replace("\\", "/")
    parts = [secure_filename(part) for part in raw.split("/") if part not in ("", ".", "..")]
    return "/".join(parts)


def store_upload(profile_id: int, filename: str, data: bytes,
                 data_dir: str | Path = storage.DATA_DIR,
                 relative_path: str = "", topic_name: str = "General") -> dict:
    """Validate and save one uploaded file, returning its stored metadata."""
    suffix = validate_single_file(filename, len(data))
    data_dir = Path(data_dir)
    safe_name = secure_filename(Path(filename).name) or f"upload{suffix}"
    topic_slug = secure_filename(topic_name) or "General"
    folder = data_dir / "uploads" / str(profile_id) / topic_slug
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    stored = folder / f"{digest[:10]}_{safe_name}"
    stored.write_bytes(data)
    return {
        "original_name": Path(filename).name,
        "relative_path": safe_relative_path(relative_path or filename),
        "stored_path": str(stored),
        "file_type": suffix.lstrip("."),
        "size_bytes": len(data),
        "sha256": digest,
    }


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _paragraph_chunks(text: str, page_ref: str = "", location_prefix: str = "") -> list[dict]:
    text = _clean_text(text)
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: list[dict] = []
    buffer = ""
    location = 1
    for paragraph in paragraphs:
        candidate = (buffer + "\n\n" + paragraph).strip() if buffer else paragraph
        if len(candidate) > CHUNK_TARGET and buffer:
            chunks.append({
                "text": buffer,
                "page_ref": page_ref,
                "location_ref": f"{location_prefix}{location}",
                "source_ref": f"{page_ref or location_prefix}{location}",
            })
            location += 1
            buffer = paragraph[-CHUNK_OVERLAP:] if len(paragraph) > CHUNK_OVERLAP else paragraph
        else:
            buffer = candidate
    if buffer:
        chunks.append({
            "text": buffer,
            "page_ref": page_ref,
            "location_ref": f"{location_prefix}{location}",
            "source_ref": f"{page_ref or location_prefix}{location}",
        })
    return chunks


def extract_text(path: str | Path, original_name: str = "") -> list[dict]:
    """Extract source-referenced chunks from a supported file."""
    path = Path(path)
    suffix = path.suffix.lower()
    name = original_name or path.name
    if suffix in TEXT_EXTENSIONS:
        text = _decode_text(path.read_bytes())
        return _paragraph_chunks(text, page_ref=name, location_prefix="")
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        chunks: list[dict] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            chunks.extend(_paragraph_chunks(text, page_ref=f"page {index}", location_prefix="p"))
        return chunks
    if suffix == ".docx":
        from docx import Document
        document = Document(str(path))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return _paragraph_chunks("\n\n".join(parts), page_ref="document", location_prefix="para ")
    if suffix == ".pptx":
        from pptx import Presentation
        presentation = Presentation(str(path))
        chunks = []
        for index, slide in enumerate(presentation.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text.strip())
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
                parts.append("Notes: " + slide.notes_slide.notes_text_frame.text.strip())
            chunks.extend(_paragraph_chunks("\n\n".join(parts), page_ref=f"slide {index}",
                                            location_prefix="shape "))
        return chunks
    if suffix in IMAGE_EXTENSIONS:
        if not ocr_available():
            return []
        try:
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(path))
        except Exception:
            return []
        return _paragraph_chunks(text, page_ref=name, location_prefix="ocr ")
    if suffix in MEDIA_EXTENSIONS:
        # A sibling transcript is handled by the caller; no audio transcription is claimed.
        return []
    return []


def transcribe_or_note(path: str | Path) -> tuple[str, str]:
    """Return (status, message) for media/image files that need a companion transcript."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        if ocr_available():
            return "ready", "OCR extracted"
        return "needs_ocr", (
            "OCR is not installed on this laptop. Upload a text-based PDF/DOCX/TXT version, "
            "or install Tesseract and the optional requirements."
        )
    if suffix in MEDIA_EXTENSIONS:
        return "needs_transcript", (
            "Audio/video transcription is not configured. Upload a matching .vtt, .srt or .txt "
            "transcript in the same folder; it will be indexed instead."
        )
    return "ready", ""


def companion_transcript(path: Path) -> Path | None:
    for suffix in (".vtt", ".srt", ".txt"):
        candidate = path.with_suffix(suffix)
        if candidate.exists() and candidate != path:
            return candidate
    return None


def import_files(profile_id: int, topic_id: int | None, topic_name: str,
                 files: Iterable[tuple[str, bytes]], data_dir: str | Path = storage.DATA_DIR,
                 db_path: str | Path = storage.DEFAULT_DB,
                 source_kind: str = "file", plugin: str = "") -> list[dict]:
    """Validate and import a batch of files, preserving relative folder paths."""
    file_list = list(files)
    validate_folder([(name, len(data)) for name, data in file_list])
    results: list[dict] = []
    for name, data in file_list:
        meta = store_upload(profile_id, name, data, data_dir,
                            relative_path=name, topic_name=topic_name)
        stored = Path(meta["stored_path"])
        chunks = extract_text(stored, meta["original_name"])
        status, message = transcribe_or_note(stored)
        if not chunks and status == "ready":
            status = "empty"
            message = "No readable text was found. Try a text-based file or a clearer scan."
        processed_text = "\n\n".join(c["text"] for c in chunks)
        resource = storage.add_resource(
            profile_id, meta["original_name"], meta["stored_path"], meta["file_type"],
            meta["size_bytes"], meta["sha256"], topic_id=topic_id,
            relative_path=meta["relative_path"], source_kind=source_kind, plugin=plugin,
            status=status, error=message, processed_text=processed_text, db_path=db_path,
        )
        if chunks:
            storage.add_chunks(resource["id"], chunks, db_path)
        results.append(resource)
    group_processed_files(profile_id, topic_id, topic_name, data_dir, db_path)
    return results


def group_processed_files(profile_id: int, topic_id: int | None, topic_name: str,
                          data_dir: str | Path = storage.DATA_DIR,
                          db_path: str | Path = storage.DEFAULT_DB) -> list[Path]:
    """Combine processed text from each group of five same-topic files into one .md file."""
    resources = [
        r for r in storage.list_resources(profile_id, db_path)
        if (topic_id is None or r["topic_id"] == topic_id) and r["processed_text"].strip()
    ]
    if not resources:
        return []
    safe_topic = secure_filename(topic_name) or "General"
    out_dir = Path(data_dir) / "processed" / safe_topic
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for batch_index in range(0, len(resources), 5):
        batch = resources[batch_index:batch_index + 5]
        lines = [
            f"# {topic_name} — processed resource batch {batch_index // 5 + 1}",
            "",
            "This file combines processed text from up to five same-topic resources.",
            "",
        ]
        for resource in batch:
            lines.extend([
                f"## {resource['original_name']}",
                f"- Source ID: resource-{resource['id']}",
                f"- Relative path: {resource['relative_path']}",
                f"- Type: {resource['file_type']}",
                "",
                resource["processed_text"].strip(),
                "",
            ])
        path = out_dir / f"batch-{batch_index // 5 + 1:03d}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        outputs.append(path)
    # Include a small partial batch even when there are fewer than five files.
    return outputs


def group_chat_logs(profile_id: int, topic_id: int | None, topic_name: str,
                    data_dir: str | Path = storage.DATA_DIR,
                    db_path: str | Path = storage.DEFAULT_DB) -> list[Path]:
    sessions = [
        s for s in storage.list_sessions(profile_id, limit=200, db_path=db_path)
        if topic_id is None or s["topic_id"] == topic_id
    ]
    if not sessions:
        return []
    safe_topic = secure_filename(topic_name) or "General"
    out_dir = Path(data_dir) / "chatlogs" / safe_topic
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for batch_index in range(0, len(sessions), 5):
        batch = sessions[batch_index:batch_index + 5]
        lines = [f"# {topic_name} — saved chat batch {batch_index // 5 + 1}", ""]
        for session in batch:
            lines.extend([f"## {session['title']} (session {session['id']})", ""])
            for message in storage.list_messages(session["id"], db_path):
                sources = ", ".join(
                    s.get("source_ref", s.get("file", "")) for s in message.get("sources", [])
                )
                lines.append(f"**{message['role'].title()}:** {message['content']}")
                if sources:
                    lines.append(f"_Sources: {sources}_")
                lines.append("")
        path = out_dir / f"batch-{batch_index // 5 + 1:03d}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        outputs.append(path)
    return outputs


def retrieve(query: str, chunks: Sequence[dict], top_k: int = 5) -> list[dict]:
    """TF-IDF cosine retrieval over source-referenced chunks (local, no cloud)."""
    if not query.strip() or not chunks:
        return []
    texts = [c["text"] for c in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                     sublinear_tf=True, max_features=20000)
        matrix = vectorizer.fit_transform(texts + [query])
    except ValueError:
        return []
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    results = []
    for index in ranked[:top_k]:
        if scores[index] <= 0:
            continue
        item = dict(chunks[index])
        item["score"] = float(scores[index])
        item["source_ref"] = item.get("source_ref") or (
            f"{item.get('original_name', 'source')} | {item.get('page_ref', '')}"
        )
        results.append(item)
    return results


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or self._skip:
            return
        if self._in_title:
            self.title += text + " "
        else:
            self.parts.append(text)


def fetch_readable(url: str, timeout: int = 12) -> dict:
    """Fetch a permitted public page and return title/text. No key or file is sent."""
    if not url.startswith(("https://", "http://")):
        raise ResourceError("Only http(s) URLs can be used for web research.")
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Rewiseorist/1.0 (local coursework app; permission-based research)"},
    )
    response.raise_for_status()
    parser = _HTMLText()
    parser.feed(response.text)
    text = _clean_text("\n\n".join(parser.parts))
    return {
        "url": url,
        "title": html.unescape(parser.title.strip()) or url,
        "text": text[:200_000],
        "fetched_at": storage.utc_now(),
    }


def resource_status(resource: dict) -> str:
    if resource.get("status") == "ready":
        return "Ready"
    if resource.get("status") == "needs_ocr":
        return "Needs OCR"
    if resource.get("status") == "needs_transcript":
        return "Needs transcript"
    return resource.get("status", "Unknown").replace("_", " ").title()
