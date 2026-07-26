from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.config import Settings
from app.core.rag import LangChainBackend, _bm25_search, _tokenize


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


class SearchStore:
    def __init__(self, results: list[tuple]) -> None:
        self.results = results
        self.requested_k: int | None = None

    def similarity_search_with_relevance_scores(self, question: str, k: int):
        self.requested_k = k
        return self.results


def test_hybrid_search_uses_a_broad_pool_for_non_local_embeddings() -> None:
    from langchain_core.documents import Document

    documents = [
        Document(page_content="Un passage sémantiquement proche."),
        Document(page_content="Un autre passage général."),
        Document(page_content="Le support technique ouvre à 8h30."),
    ]
    store = SearchStore([(document, 0.8 - index * 0.1) for index, document in enumerate(documents)])
    settings = Settings(
        llm_provider="ollama",
        embed_provider="ollama",
        top_k=2,
        hybrid_search=True,
        bm25_weight=0.8,
    )
    backend = LangChainBackend(settings)

    results = backend._search("support technique", store)

    assert store.requested_k == 20
    assert any("support technique" in document.page_content for document, _ in results)
    assert all(score is not None for _, score in results)


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        from langchain_core.messages import AIMessage

        self.prompts.append(prompt)
        return AIMessage(content="Réponse correcte")


def test_filtered_sources_stay_aligned_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.documents import Document

    recording_llm = RecordingLLM()
    monkeypatch.setattr(
        "app.core.providers.get_langchain_llm", lambda settings: recording_llm
    )
    settings = Settings(
        llm_provider="ollama",
        embed_provider="ollama",
        score_threshold=0.5,
    )
    backend = LangChainBackend(settings)
    first = Document(page_content="Passage conservé A", metadata={"source": "a.txt"})
    discarded = Document(page_content="Passage supprimé", metadata={"source": "b.txt"})
    third = Document(page_content="Passage conservé C", metadata={"source": "c.txt"})
    backend._store = lambda: object()
    backend._search = lambda question, store: [(first, 0.9), (discarded, 0.1), (third, 0.8)]

    response = backend.ask("Quelle information ?", [])

    assert [source.source for source in response.sources] == ["a.txt", "c.txt"]
    assert "Passage conservé A" in recording_llm.prompts[0]
    assert "Passage conservé C" in recording_llm.prompts[0]
    assert "Passage supprimé" not in recording_llm.prompts[0]


def test_bm25_tokenizer_handles_french_punctuation() -> None:
    assert _tokenize("Le support, technique est ouvert ?") == [
        "le",
        "support",
        "technique",
        "est",
        "ouvert",
    ]