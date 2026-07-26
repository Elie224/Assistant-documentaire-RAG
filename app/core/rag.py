from __future__ import annotations

import hashlib
import json
import re
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterable, Protocol

from app.core.config import Settings, get_settings
from app.core.documents import load_documents, split_documents
from app.core.exceptions import EmptyIndexError
from app.core.local_embeddings import _hash_vector
from app.core.schemas import (
    ChatMessage,
    ChatResponse,
    DocumentDeletionResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestionResponse,
    SourceChunk,
)


NO_ANSWER_MESSAGE = "Je ne trouve pas cette information dans les documents indexés."

SYSTEM_PROMPT = """Tu es un assistant documentaire précis, naturel et chaleureux. Réponds uniquement à partir du contexte fourni.
Si le contexte ne permet pas de répondre, dis clairement que l'information n'est pas présente dans les documents.
Réponds en français, comme un humain, avec des phrases fluides et directes.
Les sources sont affichées séparément par l'interface : ne mets aucun numéro, crochet ou marqueur de citation dans ta réponse.
N'invente jamais d'information.

Contexte :
{context}

Historique récent :
{history}

Question : {question}
"""


class RagBackend(Protocol):
    def ingest(self, paths: list[Path]) -> IngestionResponse: ...

    def retrieve(self, question: str) -> list[tuple]: ...

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse: ...

    def list_documents(self) -> DocumentListResponse: ...

    def delete_document(
        self, document_id: str, remove_upload: bool = True
    ) -> DocumentDeletionResponse: ...


def _history_text(history: Iterable[ChatMessage]) -> str:
    history = list(history)
    if not history:
        return "Aucun."
    return "\n".join(f"{message.role}: {message.content}" for message in history[-8:])


_CITATION_PATTERN = re.compile(r"\s*\[(?:\d+(?:\s*[-,]\s*\d+)*)\]")
_PUNCTUATION_SPACE = re.compile(r"\s+([.,;:!?])")


def _humanize_answer(answer: str) -> str:
    cleaned = _CITATION_PATTERN.sub("", answer)
    return _PUNCTUATION_SPACE.sub(r"\1", cleaned).strip()


def _page_number(metadata: dict) -> int | None:
    page = metadata.get("page")
    return int(page) + 1 if isinstance(page, int) else None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry_path(index_dir: Path) -> Path:
    return index_dir / "documents.sqlite3"


def _legacy_registry_path(index_dir: Path) -> Path:
    return index_dir / "indexed_files.json"


class _Registry(dict[str, list[str]]):
    def __init__(self, *args, chunk_counts: dict[str, int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_counts = chunk_counts or {}


_REGISTRY_LOCKS: dict[str, Lock] = {}
_REGISTRY_LOCKS_GUARD = Lock()


def _lock_for(index_dir: Path) -> Lock:
    key = str(index_dir.resolve())
    with _REGISTRY_LOCKS_GUARD:
        lock = _REGISTRY_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _REGISTRY_LOCKS[key] = lock
        return lock


@contextmanager
def _file_registry_lock(index_dir: Path, max_wait: float):
    index_dir.mkdir(parents=True, exist_ok=True)
    lock_path = index_dir / ".indexed_files.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + max_wait
        acquired = False
        while time.monotonic() < deadline:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.05)
        if not acquired:
            raise RuntimeError("Registre index occupé, réessayez.")
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _registry_transaction(index_dir: Path, max_wait: float = 5.0):
    """Transaction du registre protégée entre threads et processus."""
    lock = _lock_for(index_dir)
    if not lock.acquire(timeout=max_wait):
        raise RuntimeError("Registre index occupé, réessayez.")
    try:
        with _file_registry_lock(index_dir, max_wait):
            registry = _load_registry(index_dir)
            yield registry
            _save_registry(index_dir, registry)
    finally:
        lock.release()


def _connect_registry(index_dir: Path) -> sqlite3.Connection:
    index_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_registry_path(index_dir))
    connection.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            names_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            chunks INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'indexed'
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS registry_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    )
    return connection


def _migrate_legacy_registry(index_dir: Path, connection: sqlite3.Connection) -> None:
    migrated = connection.execute(
        "SELECT 1 FROM registry_meta WHERE key = 'legacy_json_migrated'"
    ).fetchone()
    if migrated:
        return
    legacy_path = _legacy_registry_path(index_dir)
    if legacy_path.exists():
        try:
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            legacy = {}
        connection.executemany(
            "INSERT OR IGNORE INTO documents (document_id, names_json) VALUES (?, ?)",
            [
                (document_id, json.dumps(names, ensure_ascii=False))
                for document_id, names in legacy.items()
            ],
        )
    connection.execute(
        "INSERT INTO registry_meta (key, value) VALUES ('legacy_json_migrated', '1')"
    )
    connection.commit()


def _load_registry(index_dir: Path) -> _Registry:
    with _connect_registry(index_dir) as connection:
        _migrate_legacy_registry(index_dir, connection)
        rows = connection.execute(
            "SELECT document_id, names_json, chunks FROM documents"
        ).fetchall()
    return _Registry(
        {document_id: json.loads(names_json) for document_id, names_json, _ in rows},
        chunk_counts={document_id: chunks for document_id, _, chunks in rows},
    )


def _save_registry(index_dir: Path, registry: dict[str, list[str]]) -> None:
    with _connect_registry(index_dir) as connection:
        existing = {
            row[0] for row in connection.execute("SELECT document_id FROM documents")
        }
        removed = existing - set(registry)
        if removed:
            connection.executemany(
                "DELETE FROM documents WHERE document_id = ?",
                [(document_id,) for document_id in removed],
            )
        chunk_counts = getattr(registry, "chunk_counts", {})
        connection.executemany(
            """INSERT INTO documents (document_id, names_json, chunks)
               VALUES (?, ?, ?)
               ON CONFLICT(document_id) DO UPDATE SET
                   names_json = excluded.names_json,
                   chunks = CASE
                       WHEN excluded.chunks > 0 THEN excluded.chunks
                       ELSE documents.chunks
                   END,
                   status = 'indexed'""",
            [
                (
                    document_id,
                    json.dumps(names, ensure_ascii=False),
                    chunk_counts.get(document_id, 0),
                )
                for document_id, names in registry.items()
            ],
        )
        connection.commit()

def _split_new_paths(
    paths: list[Path], registry: dict[str, list[str]]
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Return (kept paths, hash pairs) skipping already-indexed content."""
    kept: list[Path] = []
    pairs: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path in paths:
        digest = _file_hash(path)
        if digest in seen:
            continue
        seen.add(digest)
        if digest in registry:
            continue
        kept.append(path)
        pairs.append((path, digest))
    return kept, pairs


def _record_registry_entries(
    registry: dict[str, list[str]],
    pairs: list[tuple[Path, str]],
    chunks: list | None = None,
) -> None:
    for path, digest in pairs:
        names = registry.setdefault(digest, [])
        if path.name not in names:
            names.append(path.name)
    if chunks is not None and hasattr(registry, "chunk_counts"):
        counts: dict[str, int] = {}
        for chunk in chunks:
            document_id = chunk.metadata.get("document_id")
            if document_id:
                counts[document_id] = counts.get(document_id, 0) + 1
        registry.chunk_counts.update(counts)


def _record_files(index_dir: Path, pairs: list[tuple[Path, str]]) -> None:
    if not pairs:
        return
    with _registry_transaction(index_dir) as registry:
        _record_registry_entries(registry, pairs)


def _registry_documents(index_dir: Path) -> DocumentListResponse:
    with _connect_registry(index_dir) as connection:
        _migrate_legacy_registry(index_dir, connection)
        rows = connection.execute(
            """SELECT document_id, names_json, created_at, chunks, status
               FROM documents ORDER BY created_at, document_id"""
        ).fetchall()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                document_id=document_id,
                names=sorted(json.loads(names_json)),
                created_at=created_at,
                chunks=chunks,
                status=status,
            )
            for document_id, names_json, created_at, chunks, status in rows
        ]
    )


def _chunks_with_document_ids(
    paths: list[Path], pairs: list[tuple[Path, str]], settings: Settings
) -> list:
    document_ids = {path: document_id for path, document_id in pairs}
    chunks = []
    for path in paths:
        current_chunks = split_documents(load_documents([path]), settings)
        for chunk in current_chunks:
            chunk.metadata["document_id"] = document_ids[path]
        chunks.extend(current_chunks)
    return chunks


def _remove_uploaded_document(settings: Settings, document_id: str, names: list[str]) -> None:
    document_dir = settings.uploads_dir / document_id
    for name in names:
        candidate = document_dir / Path(name).name
        if candidate.is_file():
            candidate.unlink()
    if document_dir.is_dir() and not any(document_dir.iterdir()):
        document_dir.rmdir()


def _document_confidence(document, score: float | None) -> float | None:
    confidence = document.metadata.get("_retrieval_confidence")
    if confidence is None and score is not None and 0 <= score <= 1:
        confidence = score
    return float(confidence) if confidence is not None else None


def _annotate_vector_confidence(results: list[tuple]) -> None:
    for document, score in results:
        confidence = _document_confidence(document, score)
        if confidence is not None:
            document.metadata["_retrieval_confidence"] = confidence


def _local_lite_confidence(question: str, text: str, dimension: int) -> float:
    query_vector = _hash_vector(question, dimension)
    text_vector = _hash_vector(text, dimension)
    return max(
        0.0,
        min(
            1.0,
            sum(
                query_value * text_value
                for query_value, text_value in zip(
                    query_vector, text_vector, strict=True
                )
            ),
        ),
    )


def _annotate_local_lite_confidence(
    question: str, results: list[tuple], dimension: int
) -> None:
    for document, score in results:
        if _document_confidence(document, score) is None:
            document.metadata["_retrieval_confidence"] = _local_lite_confidence(
                question, document.page_content, dimension
            )


def _filter_sources(
    sources: list[SourceChunk], threshold: float, allow_unscored: bool
) -> list[SourceChunk]:
    return [
        source
        for source in sources
        if _score_is_allowed(
            source.confidence if source.confidence is not None else source.score,
            threshold,
            allow_unscored,
        )
    ]


def _score_is_allowed(
    score: float | None, threshold: float, allow_unscored: bool
) -> bool:
    if score is None:
        return allow_unscored
    return score >= threshold


def _filter_scored_results(
    results: list[tuple], threshold: float, allow_unscored: bool
) -> list[tuple]:
    return [
        (document, score)
        for document, score in results
        if _score_is_allowed(
            _document_confidence(document, score), threshold, allow_unscored
        )
    ]


def _no_answer_response(settings: Settings) -> ChatResponse:
    return ChatResponse(
        answer=NO_ANSWER_MESSAGE,
        sources=[],
        engine=settings.rag_engine,
        vector_store=settings.vector_store,
    )


class LangChainBackend:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _base_dir(self) -> Path:
        return self.settings.isolated_index_dir

    def _ensure_dir(self) -> Path:
        path = self._base_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _search(self, question: str, store) -> list[tuple]:
        candidate_k = max(self.settings.top_k * 5, 20)
        try:
            vector_results = store.similarity_search_with_relevance_scores(
                question, k=candidate_k
            )
        except (NotImplementedError, ValueError):
            vector_results = [
                (doc, None)
                for doc in store.similarity_search(question, k=candidate_k)
            ]

        _annotate_vector_confidence(vector_results)
        if not self.settings.hybrid_search:
            return vector_results[: self.settings.top_k]

        candidates = [document for document, _ in vector_results]
        if not candidates:
            return []
        try:
            bm25_results = _bm25_search(question, candidates, candidate_k)
        except Exception:
            return vector_results[: self.settings.top_k]
        if not bm25_results:
            return vector_results[: self.settings.top_k]
        return _hybrid_retrieve(
            question,
            vector_results,
            bm25_results,
            self.settings.top_k,
            self.settings.bm25_weight,
        )

    def retrieve(self, question: str) -> list[tuple]:
        return self._search(question, self._store())

    def _store(self):
        from app.core.providers import get_langchain_embeddings

        embeddings = get_langchain_embeddings(self.settings)
        index_dir = self._base_dir()
        if self.settings.vector_store == "chroma":
            from langchain_chroma import Chroma

            return Chroma(
                collection_name="rag_documents",
                persist_directory=str(index_dir),
                embedding_function=embeddings,
            )

        from langchain_community.vectorstores import FAISS

        if not (index_dir / "index.faiss").exists():
            raise EmptyIndexError("Aucun index. Importez d'abord des documents.")
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def ingest(self, paths: list[Path]) -> IngestionResponse:
        from app.core.providers import get_langchain_embeddings

        index_dir = self._ensure_dir()
        with _registry_transaction(index_dir) as registry:
            kept, pairs = _split_new_paths(paths, registry)
            if not kept:
                return IngestionResponse(
                    files=[],
                    chunks=0,
                    engine=self.settings.rag_engine,
                    vector_store=self.settings.vector_store,
                )
            chunks = _chunks_with_document_ids(kept, pairs, self.settings)
            if not chunks:
                raise ValueError("Les documents ne contiennent aucun texte exploitable.")

            if self.settings.vector_store == "chroma":
                store = self._store()
                store.add_documents(chunks)
            else:
                from langchain_community.vectorstores import FAISS

                embeddings = get_langchain_embeddings(self.settings)
                if (index_dir / "index.faiss").exists():
                    store = self._store()
                    store.add_documents(chunks)
                else:
                    store = FAISS.from_documents(chunks, embeddings)
                store.save_local(str(index_dir))

            _record_registry_entries(registry, pairs, chunks)
            return IngestionResponse(
                files=[path.name for path in kept],
                chunks=len(chunks),
                engine=self.settings.rag_engine,
                vector_store=self.settings.vector_store,
            )

    def list_documents(self) -> DocumentListResponse:
        return _registry_documents(self._base_dir())

    def delete_document(
        self, document_id: str, remove_upload: bool = True
    ) -> DocumentDeletionResponse:
        index_dir = self._base_dir()
        with _registry_transaction(index_dir) as registry:
            names = registry.get(document_id)
            if names is None:
                return DocumentDeletionResponse(document_id=document_id, deleted=False)

            store = self._store()
            if self.settings.vector_store == "chroma":
                ids = store.get(where={"document_id": document_id}, include=[]).get(
                    "ids", []
                )
                if not ids:
                    for name in names:
                        ids.extend(
                            store.get(where={"source": name}, include=[]).get("ids", [])
                        )
                if ids:
                    store.delete(ids=list(dict.fromkeys(ids)))
            else:
                ids = []
                for item_id in store.index_to_docstore_id.values():
                    document = store.docstore.search(item_id)
                    metadata = getattr(document, "metadata", {})
                    if metadata.get("document_id") == document_id or metadata.get(
                        "source"
                    ) in names:
                        ids.append(item_id)
                if ids:
                    store.delete(ids)
                    store.save_local(str(index_dir))

            registry.pop(document_id)
            if remove_upload:
                _remove_uploaded_document(self.settings, document_id, names)
        return DocumentDeletionResponse(document_id=document_id, deleted=True)

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse:
        from app.core.providers import get_langchain_llm

        scored = self.retrieve(question)
        if self.settings.embed_provider == "local-lite":
            _annotate_local_lite_confidence(
                question, scored, self.settings.local_embed_dimension
            )
        filtered_scored = _filter_scored_results(
            scored,
            self.settings.score_threshold,
            allow_unscored=False,
        )
        if not filtered_scored:
            return _no_answer_response(self.settings)

        sources: list[SourceChunk] = []
        context_segments: list[str] = []
        for position, (document, score) in enumerate(filtered_scored, start=1):
            text = document.page_content.strip()
            context_segments.append(f"--- Extrait {position} ---\n{text}")
            sources.append(
                SourceChunk(
                    source=str(document.metadata.get("source", "Document inconnu")),
                    page=_page_number(document.metadata),
                    score=round(float(score), 4) if score is not None else None,
                    confidence=_document_confidence(document, score),
                    preview=text[:500],
                    content=text,
                )
            )

        prompt = SYSTEM_PROMPT.format(
            context="\n\n".join(context_segments),
            history=_history_text(history),
            question=question,
        )
        response = get_langchain_llm(self.settings).invoke(prompt)
        raw_answer = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        return ChatResponse(
            answer=_humanize_answer(raw_answer),
            sources=sources,
            engine=self.settings.rag_engine,
            vector_store=self.settings.vector_store,
        )


class LlamaIndexBackend:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _base_dir(self) -> Path:
        return self.settings.isolated_index_dir

    def _ensure_dir(self) -> Path:
        path = self._base_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _chroma_index(self, embed_model, create: bool = False):
        import chromadb
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        client = chromadb.PersistentClient(path=str(self._base_dir()))
        collection = client.get_or_create_collection("rag_documents_v2")
        if collection.count() == 0 and not create:
            raise EmptyIndexError("Aucun index. Importez d'abord des documents.")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )

    def _faiss_index(self, embed_model, create: bool = False):
        from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.vector_stores.faiss import FaissVectorStore

        index_dir = self._base_dir()
        if (index_dir / "default__vector_store.json").exists():
            vector_store = FaissVectorStore.from_persist_dir(str(index_dir))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(index_dir),
            )
            return load_index_from_storage(storage_context, embed_model=embed_model)
        if not create:
            raise EmptyIndexError("Aucun index. Importez d'abord des documents.")

        import faiss

        dimension = len(embed_model.get_text_embedding("dimension"))
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(dimension))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            [], storage_context=storage_context, embed_model=embed_model
        )

    def _index(self, create: bool = False):
        from app.core.providers import get_llamaindex_embeddings

        embed_model = get_llamaindex_embeddings(self.settings)
        if self.settings.vector_store == "chroma":
            return self._chroma_index(embed_model, create=create)
        return self._faiss_index(embed_model, create=create)

    def ingest(self, paths: list[Path]) -> IngestionResponse:
        from llama_index.core.schema import TextNode

        index_dir = self._ensure_dir()
        with _registry_transaction(index_dir) as registry:
            kept, pairs = _split_new_paths(paths, registry)
            if not kept:
                return IngestionResponse(
                    files=[],
                    chunks=0,
                    engine=self.settings.rag_engine,
                    vector_store=self.settings.vector_store,
                )
            chunks = _chunks_with_document_ids(kept, pairs, self.settings)
            if not chunks:
                raise ValueError("Les documents ne contiennent aucun texte exploitable.")

            index = self._index(create=True)
            nodes = [
                TextNode(
                    text=chunk.page_content,
                    metadata=dict(chunk.metadata),
                    ref_doc_id=str(chunk.metadata["document_id"]),
                )
                for chunk in chunks
            ]
            index.insert_nodes(nodes)
            if self.settings.vector_store == "faiss":
                index.storage_context.persist(persist_dir=str(self._base_dir()))
            _record_registry_entries(registry, pairs, chunks)
            return IngestionResponse(
                files=[path.name for path in kept],
                chunks=len(chunks),
                engine=self.settings.rag_engine,
                vector_store=self.settings.vector_store,
            )

    def retrieve(self, question: str) -> list[tuple]:
        index = self._index()
        retriever = index.as_retriever(similarity_top_k=self.settings.top_k)
        return retriever.retrieve(question)

    def list_documents(self) -> DocumentListResponse:
        return _registry_documents(self._base_dir())

    def delete_document(
        self, document_id: str, remove_upload: bool = True
    ) -> DocumentDeletionResponse:
        index_dir = self._base_dir()
        with _registry_transaction(index_dir) as registry:
            names = registry.get(document_id)
            if names is None:
                return DocumentDeletionResponse(document_id=document_id, deleted=False)

            index = self._index()
            index.delete_ref_doc(document_id, delete_from_docstore=True)
            if self.settings.vector_store == "faiss":
                index.storage_context.persist(persist_dir=str(index_dir))
            registry.pop(document_id)
            if remove_upload:
                _remove_uploaded_document(self.settings, document_id, names)
        return DocumentDeletionResponse(document_id=document_id, deleted=True)

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse:
        from app.core.providers import get_llamaindex_llm
        retrieved_nodes = self.retrieve(question)
        if self.settings.embed_provider == "local-lite":
            for source_node in retrieved_nodes:
                if _document_confidence(source_node.node, source_node.score) is None:
                    source_node.node.metadata["_retrieval_confidence"] = (
                        _local_lite_confidence(
                            question,
                            source_node.node.get_content(),
                            self.settings.local_embed_dimension,
                        )
                    )
        selected_nodes = [
            source_node
            for source_node in retrieved_nodes
            if _score_is_allowed(
                _document_confidence(source_node.node, source_node.score),
                self.settings.score_threshold,
                allow_unscored=False,
            )
        ]
        if not selected_nodes:
            return _no_answer_response(self.settings)

        sources: list[SourceChunk] = []
        context_segments: list[str] = []
        for position, source_node in enumerate(selected_nodes, start=1):
            metadata = source_node.node.metadata
            text = source_node.node.get_content().strip()
            score = source_node.score
            context_segments.append(f"--- Extrait {position} ---\n{text}")
            sources.append(
                SourceChunk(
                    source=str(metadata.get("source", "Document inconnu")),
                    page=_page_number(metadata),
                    score=round(float(score), 4) if score is not None else None,
                    confidence=_document_confidence(source_node.node, score),
                    preview=text[:500],
                    content=text,
                )
            )

        prompt = SYSTEM_PROMPT.format(
            context="\n\n".join(context_segments),
            history=_history_text(history),
            question=question,
        )
        completion = get_llamaindex_llm(self.settings).complete(prompt)
        answer = getattr(completion, "text", str(completion))
        return ChatResponse(
            answer=_humanize_answer(answer),
            sources=sources,
            engine=self.settings.rag_engine,
            vector_store=self.settings.vector_store,
        )


class RagService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        backend_class = (
            LangChainBackend
            if self.settings.rag_engine == "langchain"
            else LlamaIndexBackend
        )
        self.backend: RagBackend = backend_class(self.settings)
        self._lock = Lock()

    def ingest(self, paths: list[Path]) -> IngestionResponse:
        with self._lock:
            return self.backend.ingest(paths)

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse:
        with self._lock:
            return self.backend.ask(question, history)


    def list_documents(self) -> DocumentListResponse:
        with self._lock:
            return self.backend.list_documents()

    def delete_document(self, document_id: str) -> DocumentDeletionResponse:
        with self._lock:
            return self.backend.delete_document(document_id)

    def reindex_document(self, document_id: str) -> IngestionResponse:
        with self._lock:
            listing = self.backend.list_documents()
            document = next(
                (item for item in listing.documents if item.document_id == document_id),
                None,
            )
            if document is None:
                raise ValueError("Document introuvable.")
            paths = [self.settings.uploads_dir / document_id / name for name in document.names]
            if any(not path.is_file() for path in paths):
                raise ValueError("Le fichier source du document est introuvable.")
            self.backend.delete_document(document_id, remove_upload=False)
            return self.backend.ingest(paths)


_BM25_TOKEN_PATTERN = re.compile(r"\b[\wÀ-ÿ'-]+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _BM25_TOKEN_PATTERN.findall(text.casefold())


def _bm25_search(
    question: str, documents: list, top_k: int
) -> list[tuple]:
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    tokens = _tokenize(question)
    scores = bm25.get_scores(tokens)
    indexed = sorted(
        enumerate(scores), key=lambda item: item[1], reverse=True
    )[:top_k]
    return [(documents[index], float(score)) for index, score in indexed if score > 0]


def _retrieval_key(document) -> str:
    return hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()


def _bm25_confidence_by_key(results: list[tuple]) -> dict[str, float]:
    """Convert positive BM25 evidence into a bounded, non-relative signal."""
    return {
        _retrieval_key(document): max(float(score), 0.0) / (1.0 + max(float(score), 0.0))
        for document, score in results
        if score is not None
    }


def _hybrid_retrieve(
    question: str,
    vector_results: list[tuple],
    bm25_results: list[tuple],
    top_k: int,
    bm25_weight: float,
) -> list[tuple]:
    del question
    seen: dict[str, tuple[float, object, float | None]] = {}
    vector_weight = 1.0 - bm25_weight
    rrf_constant = 60
    bm25_evidence = _bm25_confidence_by_key(bm25_results)

    for rank, (document, vector_score) in enumerate(vector_results, start=1):
        key = _retrieval_key(document)
        current_score, current_document, current_confidence = seen.get(
            key, (0.0, document, None)
        )
        confidence = (
            current_confidence
            if current_confidence is not None
            else _document_confidence(document, vector_score)
        )
        seen[key] = (
            current_score + vector_weight / (rrf_constant + rank),
            current_document,
            confidence,
        )

    for rank, (document, bm25_score) in enumerate(bm25_results, start=1):
        key = _retrieval_key(document)
        current_score, current_document, current_confidence = seen.get(
            key, (0.0, document, None)
        )
        confidence = (
            current_confidence
            if current_confidence is not None
            else bm25_evidence.get(key)
        )
        seen[key] = (
            current_score + bm25_weight / (rrf_constant + rank),
            current_document,
            confidence,
        )

    ranked = sorted(seen.values(), key=lambda item: item[0], reverse=True)
    if not ranked:
        return []

    scores = [score for score, _, _ in ranked]
    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        normalized = [1.0] * len(ranked)
    else:
        normalized = [(score - low) / (high - low) for score in scores]

    results = []
    for rank_score, (_, document, confidence) in zip(
        normalized[:top_k], ranked[:top_k]
    ):
        if confidence is not None:
            document.metadata["_retrieval_confidence"] = round(confidence, 4)
        results.append((document, round(rank_score, 4)))
    return results