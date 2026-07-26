from __future__ import annotations

import hashlib
import json
import re
import os
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

    def retrieve(self, question: str) -> list[tuple]: ...

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
        if digest in registry:
            continue
        kept.append(path)
        pairs.append((path, digest))
    return kept, pairs


def _record_files(index_dir: Path, pairs: list[tuple[Path, str]]) -> None:
    if not pairs:
        return
    with _registry_transaction(index_dir) as registry:
        for path, digest in pairs:
            names = registry.setdefault(digest, [])
            if path.name not in names:
                names.append(path.name)


def _filter_sources(
    sources: list[SourceChunk], threshold: float, allow_unscored: bool
) -> list[SourceChunk]:
    return [
        source
        for source in sources
        if _score_is_allowed(source.score, threshold, allow_unscored)
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
        if _score_is_allowed(score, threshold, allow_unscored)
    ]


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

        scored = self.retrieve(question)
        filtered_scored = _filter_scored_results(
            scored,
            self.settings.score_threshold,
            allow_unscored=self.settings.embed_provider == "local-lite",
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

    def retrieve(self, question: str) -> list[tuple]:
        index = self._index()
        retriever = index.as_retriever(similarity_top_k=self.settings.top_k)
        return retriever.retrieve(question)

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse:
        from app.core.providers import get_llamaindex_llm

        enriched_question = (
            "Réponds en français, naturellement, uniquement à partir des documents. "
            "Si l'information manque, indique-le. Les sources sont affichées séparément : "
            "n'ajoute aucun numéro, crochet ou marqueur de citation dans la réponse.\n"
            f"Historique récent:\n{_history_text(history)}\n"
            f"Question: {question}"
        )
        selected_nodes = [
            source_node
            for source_node in self.retrieve(enriched_question)
            if _score_is_allowed(
                source_node.score,
                self.settings.score_threshold,
                allow_unscored=self.settings.embed_provider == "local-lite",
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


def _hybrid_retrieve(
    question: str,
    vector_results: list[tuple],
    bm25_results: list[tuple],
    top_k: int,
    bm25_weight: float,
) -> list[tuple]:
    del question
    seen: dict[str, tuple[float, object]] = {}
    vector_weight = 1.0 - bm25_weight
    rrf_constant = 60

    for rank, (document, _) in enumerate(vector_results, start=1):
        key = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
        current_score, current_document = seen.get(key, (0.0, document))
        seen[key] = (
            current_score + vector_weight / (rrf_constant + rank),
            current_document,
        )

    for rank, (document, _) in enumerate(bm25_results, start=1):
        key = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
        current_score, current_document = seen.get(key, (0.0, document))
        seen[key] = (
            current_score + bm25_weight / (rrf_constant + rank),
            current_document,
        )

    ranked = sorted(seen.values(), key=lambda item: item[0], reverse=True)
    if not ranked:
        return []

    scores = [score for score, _ in ranked]
    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        normalized = [1.0] * len(ranked)
    else:
        normalized = [(score - low) / (high - low) for score in scores]
    return [
        (document, round(score, 4))
        for score, (_, document) in zip(normalized[:top_k], ranked[:top_k])
    ]
