from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from app.core.config import Settings
from app.core.rag import (
    LangChainBackend,
    LlamaIndexBackend,
    _file_hash,
    _load_registry,
    _record_files,
    _split_new_paths,
)


class ConstantEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


def backend_settings(tmp_path: Path, engine: str, store: str) -> Settings:
    return Settings(
        rag_engine=engine,
        vector_store=store,
        llm_provider="ollama",
        embed_provider="ollama",
        chroma_dir=tmp_path / "chroma",
        faiss_dir=tmp_path / "faiss",
        chunk_size=150,
        chunk_overlap=20,
        score_threshold=0,
    )


@pytest.mark.parametrize("store", ["chroma", "faiss"])
def test_langchain_backend_ingests_and_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: str
) -> None:
    source = tmp_path / "policy.txt"
    source.write_text("L'allocation annuelle est de 250 euros.", encoding="utf-8")
    monkeypatch.setattr(
        "app.core.providers.get_langchain_embeddings", lambda settings: ConstantEmbeddings()
    )
    monkeypatch.setattr(
        "app.core.providers.get_langchain_llm",
        lambda settings: FakeListChatModel(responses=["250 euros [1]"]),
    )
    backend = LangChainBackend(backend_settings(tmp_path, "langchain", store))

    ingestion = backend.ingest([source])
    response = backend.ask("Quel est le montant de l'allocation ?", [])

    assert ingestion.chunks == 1
    assert response.answer == "250 euros"
    assert response.sources[0].source == "policy.txt"


@pytest.mark.parametrize("store", ["chroma", "faiss"])
def test_llamaindex_backend_ingests_and_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: str
) -> None:
    source = tmp_path / "support.txt"
    source.write_text("Le support est ouvert de 8 h 30 à 18 h.", encoding="utf-8")
    monkeypatch.setattr(
        "app.core.providers.get_llamaindex_embeddings",
        lambda settings: MockEmbedding(embed_dim=8),
    )
    monkeypatch.setattr(
        "app.core.providers.get_llamaindex_llm",
        lambda settings: MockLLM(max_tokens=64),
    )
    backend = LlamaIndexBackend(backend_settings(tmp_path, "llamaindex", store))

    ingestion = backend.ingest([source])
    response = backend.ask("Quels sont les horaires du support ?", [])

    assert ingestion.chunks == 1
    assert response.answer
    assert response.sources[0].source == "support.txt"


def test_duplicate_files_are_skipped(tmp_path: Path) -> None:
    source = tmp_path / "policy.txt"
    source.write_text("allocation", encoding="utf-8")
    registry = _load_registry(tmp_path)
    kept, pairs = _split_new_paths([source], registry)
    assert kept and pairs
    _record_files(tmp_path, pairs)
    again, _ = _split_new_paths([source], _load_registry(tmp_path))
    assert not again
    assert _file_hash(source)


def test_llamaindex_generation_uses_filtered_nodes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Node:
        metadata = {"source": "policy.md"}

        def get_content(self) -> str:
            return "Le télétravail est possible trois jours par semaine."

    class Retriever:
        def retrieve(self, question: str):
            return [SimpleNamespace(node=Node(), score=0.9)]

    class Index:
        def as_retriever(self, similarity_top_k: int):
            return Retriever()

    class RecordingLLM:
        calls = 0
        prompt = ""

        def complete(self, prompt: str):
            self.calls += 1
            self.prompt = prompt
            return SimpleNamespace(text="Trois jours par semaine.")

    llm = RecordingLLM()
    monkeypatch.setattr(
        "app.core.providers.get_llamaindex_llm", lambda settings: llm
    )
    settings = backend_settings(tmp_path, "llamaindex", "chroma")
    backend = LlamaIndexBackend(settings)
    backend._index = lambda create=False: Index()

    response = backend.ask("Combien de jours ?", [])

    assert llm.calls == 1
    assert response.sources[0].source == "policy.md"
    assert "Trois jours" in response.answer
    assert "télétravail" in llm.prompt