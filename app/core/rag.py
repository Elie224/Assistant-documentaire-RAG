from __future__ import annotations

import re
from pathlib import Path
from threading import Lock
from typing import Protocol

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


def _history_text(history: list[ChatMessage]) -> str:
    if not history:
        return "Aucun."
    return "\n".join(f"{message.role}: {message.content}" for message in history[-8:])


def _humanize_answer(answer: str) -> str:
    without_citations = re.sub(
        r"\s*\[(?:\d+(?:\s*[-,]\s*\d+)*)\]",
        "",
        answer,
    )
    return re.sub(r"\s+([.,;:!?])", r"\1", without_citations).strip()


def _page_number(metadata: dict) -> int | None:
    page = metadata.get("page")
    return int(page) + 1 if isinstance(page, int) else None


class LangChainBackend:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _store(self):
        from app.core.providers import get_langchain_embeddings

        embeddings = get_langchain_embeddings(self.settings)
        index_dir = self.settings.index_dir
        if self.settings.vector_store == "chroma":
            from langchain_chroma import Chroma

            return Chroma(
                collection_name="rag_documents_v2",
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

        documents = load_documents(paths)
        chunks = split_documents(documents, self.settings)
        if not chunks:
            raise ValueError("Les documents ne contiennent aucun texte exploitable.")

        index_dir = self.settings.index_dir
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

        return IngestionResponse(
            files=[path.name for path in paths],
            chunks=len(chunks),
            engine=self.settings.rag_engine,
            vector_store=self.settings.vector_store,
        )

    def ask(self, question: str, history: list[ChatMessage]) -> ChatResponse:
        from app.core.providers import get_langchain_llm

        store = self._store()
        if self.settings.embed_provider == "local":
            results = [
                (document, None)
                for document in store.similarity_search(question, k=self.settings.top_k)
            ]
        else:
            try:
                results = store.similarity_search_with_relevance_scores(
                    question, k=self.settings.top_k
                )
            except (NotImplementedError, ValueError):
                results = [
                    (document, None)
                    for document in store.similarity_search(question, k=self.settings.top_k)
                ]

        filtered = [
            (document, score)
            for document, score in results
            if score is None or score >= self.settings.score_threshold
        ]
        if not filtered:
            return ChatResponse(
                answer="Je ne trouve pas cette information dans les documents indexés.",
                sources=[],
                engine=self.settings.rag_engine,
                vector_store=self.settings.vector_store,
            )

        context_parts = []
        sources = []
        for position, (document, score) in enumerate(filtered, start=1):
            text = document.page_content.strip()
            context_parts.append(f"--- Extrait {position} ---\n{text}")
            sources.append(
                SourceChunk(
                    source=str(document.metadata.get("source", "Document inconnu")),
                    page=_page_number(document.metadata),
                    score=round(float(score), 4) if score is not None else None,
                    preview=text[:500],
                )
            )

        prompt = SYSTEM_PROMPT.format(
            context="\n\n".join(context_parts),
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

    def _chroma_index(self, embed_model, create: bool = False):
        import chromadb
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        client = chromadb.PersistentClient(path=str(self.settings.index_dir))
        collection = client.get_or_create_collection("rag_documents")
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

        index_dir = self.settings.index_dir
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

        documents = load_documents(paths)
        chunks = split_documents(documents, self.settings)
        if not chunks:
            raise ValueError("Les documents ne contiennent aucun texte exploitable.")

        self.settings.index_dir.mkdir(parents=True, exist_ok=True)
        index = self._index(create=True)
        nodes = [
            TextNode(text=chunk.page_content, metadata=dict(chunk.metadata))
            for chunk in chunks
        ]
        index.insert_nodes(nodes)
        if self.settings.vector_store == "faiss":
            index.storage_context.persist(persist_dir=str(self.settings.index_dir))

        return IngestionResponse(
            files=[path.name for path in paths],
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
        sources = []
        for source_node in response.source_nodes:
            metadata = source_node.node.metadata
            text = source_node.node.get_content().strip()
            score = source_node.score
            sources.append(
                SourceChunk(
                    source=str(metadata.get("source", "Document inconnu")),
                    page=_page_number(metadata),
                    score=round(float(score), 4) if score is not None else None,
                    preview=text[:500],
                )
            )
        return ChatResponse(
            answer=_humanize_answer(str(response)),
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
