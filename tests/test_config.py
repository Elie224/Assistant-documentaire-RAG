from pathlib import Path

import pytest

from app.core.config import Settings


def test_relative_index_path_is_resolved_inside_project() -> None:
    settings = Settings(
        vector_store="chroma",
        rag_engine="langchain",
        chroma_dir=Path("data/test-index"),
    )

    assert settings.index_dir.is_absolute()
    assert settings.index_dir.parts[-2:] == ("test-index", "langchain")


def test_index_is_separated_by_engine(tmp_path: Path) -> None:
    langchain = Settings(
        vector_store="faiss",
        rag_engine="langchain",
        faiss_dir=tmp_path,
    )
    llamaindex = Settings(
        vector_store="faiss",
        rag_engine="llamaindex",
        faiss_dir=tmp_path,
    )

    assert langchain.index_dir != llamaindex.index_dir


def test_openai_key_is_required() -> None:
    settings = Settings(openai_api_key=None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.openai_key()


def test_anthropic_key_is_required() -> None:
    settings = Settings(anthropic_api_key=None)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        settings.anthropic_key()


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=200, chunk_overlap=200)
