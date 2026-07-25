from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.config import Settings
from app.core.rag import LangChainBackend, _bm25_search


class ConstantEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for index in range(len(texts)):
            vector = [0.0] * 4
            vector[index % 4] = 1.0
            vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def populated_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LangChainBackend:
    source = tmp_path / "guide.txt"
    source.write_text(
        "L'allocation annuelle est de 250 euros.\n"
        "Le support est ouvert de 8h30 à 18h en semaine.\n"
        "Le télétravail est autorisé trois jours par semaine.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.core.providers.get_langchain_embeddings", lambda settings: ConstantEmbeddings()
    )
    monkeypatch.setattr(
        "app.core.providers.get_langchain_llm",
        lambda settings: FakeListChatModel(responses=["OK"]),
    )
    settings = Settings(
        rag_engine="langchain",
        vector_store="chroma",
        llm_provider="ollama",
        embed_provider="ollama",
        chroma_dir=tmp_path / "chroma",
        faiss_dir=tmp_path / "faiss",
        chunk_size=200,
        chunk_overlap=20,
        hybrid_search=True,
        score_threshold=0.0,
    )
    backend = LangChainBackend(settings)
    backend.ingest([source])
    return backend


def test_bm25_returns_relevant_document_first() -> None:
    from langchain_core.documents import Document

    documents = [
        Document(page_content="Le chat dort sur le canapé."),
        Document(page_content="Le support technique ouvre à 8h30."),
        Document(page_content="Recette de cuisine à base de chocolat."),
    ]

    scored = _bm25_search("support technique", documents, top_k=2)
    assert scored
    assert "support" in scored[0][0].page_content.lower()


def test_hybrid_search_returns_answer(populated_index: LangChainBackend) -> None:
    response = populated_index.ask("Quels sont les horaires du support ?", [])

    assert response.answer
    assert response.sources
