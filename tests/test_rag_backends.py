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
    _remove_langchain_operation_ids,
    _file_hash,
    _load_registry,
    _record_files,
    _registry_transaction,
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
        hybrid_search=(engine == "langchain"),
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
    digest = _file_hash(source)
    registry = _load_registry(tmp_path)
    kept, pairs = _split_new_paths([source], registry)
    assert kept and pairs
    _record_files(tmp_path, pairs)
    with _registry_transaction(tmp_path) as updated_registry:
        updated_registry.chunk_counts[digest] = 1
    again, _ = _split_new_paths([source], _load_registry(tmp_path))
    assert not again
    assert digest


def test_llamaindex_generation_uses_filtered_nodes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Node:
        metadata = {"source": "policy.md"}

        def get_content(self) -> str:
            return "Le télétravail est possible trois jours par semaine."

    class Retriever:
        questions: list[str] = []

        def retrieve(self, question: str):
            self.questions.append(question)
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
    assert Retriever.questions == ["Combien de jours ?"]
    assert response.sources[0].source == "policy.md"
    assert "Trois jours" in response.answer
    assert "télétravail" in llm.prompt


@pytest.mark.parametrize("store", ["chroma", "faiss"])
def test_langchain_documents_can_be_listed_and_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: str
) -> None:
    source = tmp_path / "policy.txt"
    source.write_text("L'allocation annuelle est de 250 euros.", encoding="utf-8")
    monkeypatch.setattr(
        "app.core.providers.get_langchain_embeddings", lambda settings: ConstantEmbeddings()
    )
    settings = backend_settings(tmp_path, "langchain", store).model_copy(
        update={"raw_data_dir": tmp_path / "raw"}
    )
    backend = LangChainBackend(settings)

    backend.ingest([source])
    document_id = _file_hash(source)

    listing = backend.list_documents()
    assert [(item.document_id, item.names) for item in listing.documents] == [
        (document_id, ["policy.txt"])
    ]
    assert listing.documents[0].chunks == 1
    assert listing.documents[0].status == "indexed"
    assert listing.documents[0].created_at

    deletion = backend.delete_document(document_id)

    assert deletion.deleted is True
    assert backend.list_documents().documents == []
    assert backend.delete_document(document_id).deleted is False


def test_langchain_document_can_be_reindexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.rag import RagService

    monkeypatch.setattr(
        "app.core.providers.get_langchain_embeddings", lambda settings: ConstantEmbeddings()
    )
    settings = backend_settings(tmp_path, "langchain", "faiss").model_copy(
        update={"raw_data_dir": tmp_path / "raw"}
    )
    content = "L'allocation annuelle est de 250 euros."
    source_seed = tmp_path / "source.txt"
    source_seed.write_text(content, encoding="utf-8")
    document_id = _file_hash(source_seed)
    source = settings.uploads_dir / document_id / "policy.txt"
    source.parent.mkdir(parents=True)
    source.write_text(content, encoding="utf-8")
    backend = LangChainBackend(settings)
    backend.ingest([source])

    response = RagService(settings).reindex_document(document_id)

    assert response.files == ["policy.txt"]
    assert RagService(settings).list_documents().documents[0].chunks == 1
    assert source.exists()


@pytest.mark.parametrize("store", ["chroma", "faiss"])
def test_langchain_ingest_tags_chunks_with_operation_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: str
) -> None:
    source = tmp_path / "policy.txt"
    source.write_text("L'allocation annuelle est de 250 euros.", encoding="utf-8")
    monkeypatch.setattr(
        "app.core.providers.get_langchain_embeddings", lambda settings: ConstantEmbeddings()
    )
    backend = LangChainBackend(backend_settings(tmp_path, "langchain", store))

    backend.ingest([source])
    store_obj = backend._store()

    if store == "chroma":
        payload = store_obj.get(where={"document_id": _file_hash(source)}, include=["metadatas"])
        metadatas = payload.get("metadatas", [])
        assert metadatas
        assert all("operation_id" in (metadata or {}) for metadata in metadatas)
    else:
        metadatas = []
        for item_id in store_obj.index_to_docstore_id.values():
            document = store_obj.docstore.search(item_id)
            metadata = getattr(document, "metadata", {})
            if metadata.get("document_id") == _file_hash(source):
                metadatas.append(metadata)
        assert metadatas
        assert all("operation_id" in metadata for metadata in metadatas)


@pytest.mark.parametrize("store", ["chroma", "faiss"])
def test_langchain_operation_cleanup_only_removes_target_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: str
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("Contrat A", encoding="utf-8")
    second.write_text("Contrat B", encoding="utf-8")
    monkeypatch.setattr(
        "app.core.providers.get_langchain_embeddings", lambda settings: ConstantEmbeddings()
    )
    backend = LangChainBackend(backend_settings(tmp_path, "langchain", store))
    backend.ingest([first, second])

    first_id = _file_hash(first)
    second_id = _file_hash(second)
    index_dir = backend._base_dir()
    store_obj = backend._store()

    if store == "chroma":
        first_meta = (
            store_obj.get(where={"document_id": first_id}, include=["metadatas"])
            .get("metadatas", [])[0]
        )
        first_operation = first_meta["operation_id"]
        _remove_langchain_operation_ids(store_obj, store, index_dir, {first_operation})

        first_ids = store_obj.get(where={"document_id": first_id}, include=[]).get("ids", [])
        second_ids = store_obj.get(where={"document_id": second_id}, include=[]).get("ids", [])
        assert not first_ids
        assert second_ids
    else:
        first_operation = None
        first_chunks = 0
        second_chunks = 0
        for item_id in store_obj.index_to_docstore_id.values():
            document = store_obj.docstore.search(item_id)
            metadata = getattr(document, "metadata", {})
            if metadata.get("document_id") == first_id:
                first_operation = metadata.get("operation_id")
                first_chunks += 1
            if metadata.get("document_id") == second_id:
                second_chunks += 1
        assert first_operation is not None
        _remove_langchain_operation_ids(store_obj, store, index_dir, {first_operation})

        first_chunks_after = 0
        second_chunks_after = 0
        reloaded = backend._store()
        for item_id in reloaded.index_to_docstore_id.values():
            document = reloaded.docstore.search(item_id)
            metadata = getattr(document, "metadata", {})
            if metadata.get("document_id") == first_id:
                first_chunks_after += 1
            if metadata.get("document_id") == second_id:
                second_chunks_after += 1
        assert first_chunks > 0 and first_chunks_after == 0
        assert second_chunks > 0 and second_chunks_after > 0


def test_langchain_reindex_restores_previous_version_on_ingest_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.rag import RagService

    monkeypatch.setattr(
        "app.core.providers.get_langchain_embeddings", lambda settings: ConstantEmbeddings()
    )
    monkeypatch.setattr(
        "app.core.providers.get_langchain_llm",
        lambda settings: FakeListChatModel(responses=["250 euros"]),
    )
    settings = backend_settings(tmp_path, "langchain", "chroma").model_copy(
        update={"raw_data_dir": tmp_path / "raw"}
    )
    content = "L'allocation annuelle est de 250 euros."
    seed = tmp_path / "seed.txt"
    seed.write_text(content, encoding="utf-8")
    document_id = _file_hash(seed)
    source = settings.uploads_dir / document_id / "policy.txt"
    source.parent.mkdir(parents=True)
    source.write_text(content, encoding="utf-8")

    service = RagService(settings)
    service.ingest([source])

    def failing_ingest(paths: list[Path]):
        del paths
        raise ValueError("échec volontaire")

    monkeypatch.setattr(service.backend, "ingest", failing_ingest)

    with pytest.raises(ValueError, match="restaurée"):
        service.reindex_document(document_id)

    listing = service.list_documents()
    assert listing.documents and listing.documents[0].document_id == document_id


@pytest.mark.parametrize(
    ("engine", "store"),
    [
        ("langchain", "chroma"),
        ("langchain", "faiss"),
        ("llamaindex", "chroma"),
        ("llamaindex", "faiss"),
    ],
)
def test_local_lite_with_anthropic_configuration_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    store: str,
) -> None:
    source = tmp_path / f"{engine}-{store}.txt"
    source.write_text("Le contrat A se termine le 31 décembre 2026.", encoding="utf-8")

    monkeypatch.setattr(
        "app.core.providers.get_langchain_llm",
        lambda settings: FakeListChatModel(responses=["31 décembre 2026"]),
    )

    class LlamaMock:
        def complete(self, prompt: str):
            del prompt
            return SimpleNamespace(text="31 décembre 2026")

    monkeypatch.setattr(
        "app.core.providers.get_llamaindex_llm",
        lambda settings: LlamaMock(),
    )

    settings = Settings(
        rag_engine=engine,
        vector_store=store,
        llm_provider="anthropic",
        embed_provider="local-lite",
        hybrid_search=(engine == "langchain"),
        local_embed_dimension=128,
        chroma_dir=tmp_path / "chroma",
        faiss_dir=tmp_path / "faiss",
        chunk_size=200,
        chunk_overlap=20,
        score_threshold=0,
    )

    backend = LangChainBackend(settings) if engine == "langchain" else LlamaIndexBackend(settings)
    ingestion = backend.ingest([source])
    response = backend.ask("Quelle est la date de fin du contrat A ?", [])

    assert ingestion.chunks >= 1
    assert response.answer
    assert response.sources