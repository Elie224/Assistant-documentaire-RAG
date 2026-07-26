from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.exceptions import EmptyIndexError
from app.core.rag import LangChainBackend, LlamaIndexBackend, _load_registry


def _vector_document_ids(backend: LangChainBackend) -> set[str]:
    store = backend._store()
    if backend.settings.vector_store == "chroma":
        payload = store.get(include=["metadatas"])
        metadatas = payload.get("metadatas", []) or []
        return {
            str(metadata.get("document_id"))
            for metadata in metadatas
            if metadata and metadata.get("document_id")
        }

    document_ids: set[str] = set()
    for item_id in store.index_to_docstore_id.values():
        document = store.docstore.search(item_id)
        metadata = getattr(document, "metadata", {})
        document_id = metadata.get("document_id")
        if document_id:
            document_ids.add(str(document_id))
    return document_ids


def reconcile(settings: Settings) -> dict:
    index_dir = settings.isolated_index_dir
    registry = _load_registry(index_dir)
    registry_ids = set(registry.keys())

    uploads = settings.uploads_dir
    file_ids = {
        entry.name
        for entry in uploads.iterdir()
        if entry.is_dir() and len(entry.name) == 64
    } if uploads.exists() else set()

    vector_ids: set[str]
    try:
        if settings.rag_engine == "langchain":
            vector_ids = _vector_document_ids(LangChainBackend(settings))
        else:
            # LlamaIndex snapshots are harder to introspect consistently across stores;
            # use registry as source of truth until engine-specific probes are added.
            _ = LlamaIndexBackend(settings)._index()
            vector_ids = set(registry_ids)
    except (EmptyIndexError, FileNotFoundError, ValueError):
        vector_ids = set()

    registry_without_vectors = sorted(registry_ids - vector_ids)
    vectors_without_registry = sorted(vector_ids - registry_ids)
    files_without_registry = sorted(file_ids - registry_ids)
    registry_without_files = sorted(registry_ids - file_ids)

    return {
        "workspace_id": settings.workspace_id,
        "engine": settings.rag_engine,
        "vector_store": settings.vector_store,
        "counts": {
            "registry": len(registry_ids),
            "vectors": len(vector_ids),
            "files": len(file_ids),
        },
        "issues": {
            "registry_without_vectors": registry_without_vectors,
            "vectors_without_registry": vectors_without_registry,
            "files_without_registry": files_without_registry,
            "registry_without_files": registry_without_files,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile registry/files/vector-store state")
    parser.add_argument("--workspace", default="default", help="Workspace identifier")
    parser.add_argument(
        "--engine",
        choices=["langchain", "llamaindex"],
        default=None,
        help="Override RAG engine",
    )
    parser.add_argument(
        "--vector-store",
        choices=["chroma", "faiss"],
        default=None,
        help="Override vector store",
    )
    args = parser.parse_args()

    settings = get_settings().for_workspace(args.workspace)
    updates: dict = {}
    if args.engine is not None:
        updates["rag_engine"] = args.engine
    if args.vector_store is not None:
        updates["vector_store"] = args.vector_store
    if updates:
        settings = settings.model_copy(update=updates)

    report = reconcile(settings)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
