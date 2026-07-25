from __future__ import annotations

import hashlib
import json
import re
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterable, Protocol

from app.core.config import Settings, get_settings
from app.core.documents import load_documents, split_documents
from app.core.exceptions import EmptyIndexError
from app.core.schemas import ChatMessage, ChatResponse, IngestionResponse, SourceChunk


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

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse: ...


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
    return index_dir / "indexed_files.json"


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
def _registry_transaction(index_dir: Path, max_wait: float = 5.0):
    """Verrou court sur le registre avec écriture atomique."""
    lock = _lock_for(index_dir)
    if not lock.acquire(timeout=max_wait):
        raise RuntimeError("Registre index occupé, réessayez.")
    try:
        registry = _load_registry(index_dir)
        yield registry
        _save_registry(index_dir, registry)
    finally:
        lock.release()


def _load_registry(index_dir: Path) -> dict[str, list[str]]:
    path = _registry_path(index_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_registry(index_dir: Path, registry: dict[str, list[str]]) -> None:
    target = _registry_path(index_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temp, target)


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
        if path.name in registry.get(digest, []):
            continue
        kept.append(path)
        pairs.append((path, digest))
    return kept, pairs


def _record_files(index_dir: Path, pairs: list[tuple[Path, str]]) -> None:
    if not pairs:
        return
    registry = _load_registry(index_dir)
    with _registry_transaction(index_dir) as registry:
        for path, digest in pairs:
            registry.setdefault(digest, []).append(path.name)


def _filter_sources(
    sources: list[SourceChunk], threshold: float, allow_unscored: bool
) -> list[SourceChunk]:
    result = []
    for source in sources:
        if source.score is None:
            if allow_unscored:
                result.append(source)
            continue
        if source.score >= threshold:
            result.append(source)
    return result


def _no_answer_response(settings: Settings) -> ChatResponse:
    return ChatResponse(
        answer="Je ne trouve pas cette information dans les documents indexés.",
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
        if self.settings.embed_provider == "local-lite":
            vector_results = [
                (doc, None)
                for doc in store.similarity_search(question, k=self.settings.top_k)
            ]
        else:
            try:
                vector_results = store.similarity_search_with_relevance_scores(
                    question, k=self.settings.top_k
                )
            except (NotImplementedError, ValueError):
                vector_results = [
                    (doc, None)
                    for doc in store.similarity_search(question, k=self.settings.top_k)
                ]
        if not self.settings.hybrid_search or self.settings.vector_store != "chroma":
            return vector_results
        try:
            candidate_pool = store.similarity_search(
                question, k=max(self.settings.top_k * 4, 12)
            )
        except Exception:
            return vector_results
        try:
            bm25_results = _bm25_search(question, candidate_pool, self.settings.top_k)
        except Exception:
            return vector_results
        return _hybrid_retrieve(
            question,
            vector_results,
            bm25_results,
            self.settings.top_k,
            self.settings.bm25_weight,
        )

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

        kept, pairs = _split_new_paths(paths, _load_registry(self._base_dir()))
        if not kept:
            return IngestionResponse(
                files=[],
                chunks=0,
                engine=self.settings.rag_engine,
                vector_store=self.settings.vector_store,
            )
        documents = load_documents(kept)
        chunks = split_documents(documents, self.settings)
        if not chunks:
            raise ValueError("Les documents ne contiennent aucun texte exploitable.")

        index_dir = self._ensure_dir()
        index_dir.mkdir(parents=True, exist_ok=True)
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

        _record_files(index_dir, pairs)
        return IngestionResponse(
            files=[path.name for path in kept],
            chunks=len(chunks),
            engine=self.settings.rag_engine,
            vector_store=self.settings.vector_store,
        )

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse:
        from app.core.providers import get_langchain_llm

        store = self._store()
        scored = self._search(question, store)

        sources: list[SourceChunk] = []
        context_segments: list[str] = []
        for position, (document, score) in enumerate(scored, start=1):
            text = document.page_content.strip()
            context_segments.append(f"--- Extrait {position} ---\n{text}")
            sources.append(
                SourceChunk(
                    source=str(document.metadata.get("source", "Document inconnu")),
                    page=_page_number(document.metadata),
                    score=round(float(score), 4) if score is not None else None,
                    preview=text[:500],
                    content=text,
                )
            )
        filtered = _filter_sources(
            sources,
            self.settings.score_threshold,
            allow_unscored=self.settings.embed_provider == "local-lite",
        )
        if not filtered:
            return _no_answer_response(self.settings)

        prompt = SYSTEM_PROMPT.format(
            context="\n\n".join(context_segments[: len(filtered)]),
            history=_history_text(history),
            question=question,
        )
        response = get_langchain_llm(self.settings).invoke(prompt)
        raw_answer = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        return ChatResponse(
            answer=_humanize_answer(raw_answer),
            sources=filtered,
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

        kept, pairs = _split_new_paths(paths, _load_registry(self._base_dir()))
        if not kept:
            return IngestionResponse(
                files=[],
                chunks=0,
                engine=self.settings.rag_engine,
                vector_store=self.settings.vector_store,
            )
        documents = load_documents(kept)
        chunks = split_documents(documents, self.settings)
        if not chunks:
            raise ValueError("Les documents ne contiennent aucun texte exploitable.")

        self._ensure_dir()
        index = self._index(create=True)
        nodes = [
            TextNode(text=chunk.page_content, metadata=dict(chunk.metadata))
            for chunk in chunks
        ]
        index.insert_nodes(nodes)
        if self.settings.vector_store == "faiss":
            index.storage_context.persist(persist_dir=str(self._base_dir()))
        _record_files(self._base_dir(), pairs)
        return IngestionResponse(
            files=[path.name for path in kept],
            chunks=len(chunks),
            engine=self.settings.rag_engine,
            vector_store=self.settings.vector_store,
        )

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse:
        from app.core.providers import get_llamaindex_llm

        index = self._index()
        query_engine = index.as_query_engine(
            llm=get_llamaindex_llm(self.settings),
            similarity_top_k=self.settings.top_k,
            response_mode="compact",
        )
        enriched_question = (
            "Réponds en français, naturellement, uniquement à partir des documents. "
            "Si l'information manque, indique-le. Les sources sont affichées séparément : "
            "n'ajoute aucun numéro, crochet ou marqueur de citation dans la réponse.\n"
            f"Historique récent:\n{_history_text(history)}\n"
            f"Question: {question}"
        )
        response = query_engine.query(enriched_question)
        sources: list[SourceChunk] = []
        context_segments: list[str] = []
        for position, source_node in enumerate(response.source_nodes, start=1):
            metadata = source_node.node.metadata
            text = source_node.node.get_content().strip()
            score = source_node.score
            context_segments.append(f"--- Extrait {position} ---\n{text}")
            sources.append(
                SourceChunk(
                    source=str(metadata.get("source", "Document inconnu")),
                    page=_page_number(metadata),
                    score=round(float(score), 4) if score is not None else None,
                    preview=text[:500],
                    content=text,
                )
            )
        filtered = _filter_sources(
            sources,
            self.settings.score_threshold,
            allow_unscored=self.settings.embed_provider == "local-lite",
        )
        if not filtered:
            return _no_answer_response(self.settings)
        enriched = (
            "Contexte documentaire à utiliser :\n"
            + "\n\n".join(context_segments[: len(filtered)])
            + "\n\n"
            + enriched_question
        )
        rerun = query_engine.query(enriched)
        return ChatResponse(
            answer=_humanize_answer(str(rerun)),
            sources=filtered,
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
import os
from threading import Lock
def _normalize_scores(pairs: list[tuple]) -> list[tuple]:
    scores = [score for _, score in pairs if score is not None]
    if not scores:
        return pairs
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [(doc, (score - lo) if score is not None else None) for doc, score in pairs]
    return [
        (doc, ((score - lo) / (hi - lo)) if score is not None else None)
        for doc, score in pairs
    ]


def _bm25_search(
    question: str, documents: list, top_k: int
) -> list[tuple]:
    from rank_bm25 import BM25Okapi

    tokenized = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    tokens = question.lower().split()
    scores = bm25.get_scores(tokens)
    indexed = sorted(
        enumerate(scores), key=lambda item: item[1], reverse=True
    )[:top_k]
    return [(documents[index], float(score)) for index, score in indexed if score > 0]


def _hybrid_retrieve(
    question: str,
    vector_results: list[tuple],
    bm25_results: list[tuple],
    top_k: int,
    bm25_weight: float,
) -> list[tuple]:
    seen: dict[str, tuple[float, dict | None]] = {}
    vector_weight = 1.0 - bm25_weight
    for doc, score in _normalize_scores(vector_results):
        key = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
        seen[key] = (
            max(seen.get(key, (0.0, doc))[0], (score or 0.0) * vector_weight),
            doc,
        )
    for doc, score in _normalize_scores(bm25_results):
        key = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
        if score is None:
            continue
        weighted = score * bm25_weight
        if key in seen:
            doc_existing = seen[key][1]
            seen[key] = (max(seen[key][0], weighted), doc_existing)
        else:
            seen[key] = (weighted, doc)
    ranked = sorted(seen.items(), key=lambda item: item[1][0], reverse=True)[:top_k]
    return [(doc, score) for _, (score, doc) in ranked]
