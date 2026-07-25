from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.documents import load_documents, split_documents
from app.core.exceptions import UnsupportedDocumentError


def test_text_document_is_loaded_and_split(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.txt"
    document_path.write_text("Politique d'équipement. " * 100, encoding="utf-8")
    settings = Settings(chunk_size=150, chunk_overlap=20)

    documents = load_documents([document_path])
    chunks = split_documents(documents, settings)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "guide.txt" for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert "équipement" in chunks[0].page_content


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    document_path = tmp_path / "archive.zip"
    document_path.write_bytes(b"not-a-document")

    with pytest.raises(UnsupportedDocumentError):
        load_documents([document_path])


def test_markdown_document_keeps_section_headers(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.md"
    document_path.write_text(
        "# Politique interne\n\n## Avantages\n\nL'allocation est de 250 euros par an.\n",
        encoding="utf-8",
    )
    settings = Settings(chunk_size=200, chunk_overlap=20)

    chunks = split_documents(load_documents([document_path]), settings)

    assert any("Avantages" in chunk.page_content for chunk in chunks)
    assert any("250 euros" in chunk.page_content for chunk in chunks)
