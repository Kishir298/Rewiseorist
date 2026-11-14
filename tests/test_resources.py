from pathlib import Path

import pytest
from docx import Document

import resources
import storage


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    storage.init_db(path)
    return path


def test_validate_limits(tmp_path):
    resources.validate_single_file("notes.txt", 1024)
    with pytest.raises(resources.ResourceError, match="20 MB"):
        resources.validate_single_file("big.txt", resources.MAX_FILE_BYTES + 1)
    with pytest.raises(resources.ResourceError, match="Unsupported"):
        resources.validate_single_file("virus.exe", 10)
    files = [(f"f{i}.txt", 1) for i in range(resources.MAX_FOLDER_FILES + 1)]
    with pytest.raises(resources.ResourceError, match="50"):
        resources.validate_folder(files)


def test_import_text_and_retrieve(tmp_path, db_path):
    profile = storage.create_profile("Test", db_path)
    topic = storage.add_topic(profile["id"], "Forces", db_path=db_path)
    results = resources.import_files(
        profile["id"], topic["id"], "Forces",
        [("forces.md", b"# Forces\n\nA force is a push or pull.\n\nFriction opposes motion.")],
        data_dir=tmp_path, db_path=db_path,
    )
    assert len(results) == 1
    assert results[0]["status"] == "ready"
    chunks = storage.chunks_for_topic(profile["id"], topic["id"], db_path)
    assert len(chunks) >= 1
    hits = resources.retrieve("What opposes motion?", chunks, top_k=3)
    assert hits
    assert "Friction" in hits[0]["text"]


def test_docx_extraction(tmp_path, db_path):
    docx_path = tmp_path / "notes.docx"
    doc = Document()
    doc.add_paragraph("Photosynthesis converts light energy into chemical energy.")
    doc.add_paragraph("Chlorophyll absorbs light in the chloroplast.")
    doc.save(docx_path)
    chunks = resources.extract_text(docx_path)
    assert chunks
    assert any("Chlorophyll" in c["text"] for c in chunks)
    assert chunks[0]["page_ref"] == "document"


def test_grouping_five_files(tmp_path, db_path):
    profile = storage.create_profile("Group", db_path)
    topic = storage.add_topic(profile["id"], "Algebra", db_path=db_path)
    files = [(f"note{i}.md", f"Algebra fact number {i}.".encode()) for i in range(6)]
    resources.import_files(profile["id"], topic["id"], "Algebra", files,
                           data_dir=tmp_path, db_path=db_path)
    outputs = sorted((tmp_path / "processed" / "Algebra").glob("*.md"))
    assert len(outputs) == 2
    text = outputs[0].read_text(encoding="utf-8")
    assert "Source ID" in text
    assert text.count("## note") == 5


def test_unsupported_media_status(tmp_path, db_path):
    profile = storage.create_profile("Media", db_path)
    topic = storage.add_topic(profile["id"], "History", db_path=db_path)
    result = resources.import_files(
        profile["id"], topic["id"], "History",
        [("lecture.mp4", b"not a real video")],
        data_dir=tmp_path, db_path=db_path,
    )[0]
    assert result["status"] == "needs_transcript"
    assert "transcript" in result["error"].lower()
