import hashlib
import logging
import secrets
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.documents import SUPPORTED_EXTENSIONS
from app.core.exceptions import RagError
from app.core.rag import RagService
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestionResponse,
)


logger = logging.getLogger(__name__)


def _load_settings() -> Settings:
    return get_settings()


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description="API d'assistant documentaire avec LangChain ou LlamaIndex.",
    version="1.1.0",
)


def require_api_key(
    x_api_key: str | None = Header(default=None),
    active_settings: Settings = Depends(_load_settings),
) -> None:
    if active_settings.api_key is None or not active_settings.api_key.get_secret_value():
        return
    expected = active_settings.api_key.get_secret_value()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Clé API invalide ou manquante.")


@lru_cache
def get_rag_service() -> RagService:
    return RagService(settings)


def _safe_upload_path(
    filename: str | None, content_hash: str | None = None
) -> Path:
    safe_name = Path(filename or "document").name
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(415, f"Format non pris en charge. Formats acceptés : {formats}")
    destination = settings.uploads_dir
    if content_hash:
        destination /= content_hash
    return destination / safe_name


async def _save_upload(upload: UploadFile) -> Path:
    temporary: Path | None = None
    try:
        safe_name = Path(upload.filename or "document").name
        _safe_upload_path(safe_name)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        temporary = settings.uploads_dir / f".{uuid.uuid4().hex}.part"
        max_bytes = settings.max_upload_mb * 1024 * 1024
        written = 0
        digest = hashlib.sha256()

        with temporary.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        413,
                        f"{safe_name} dépasse la limite de {settings.max_upload_mb} Mo.",
                    )
                digest.update(chunk)
                output.write(chunk)

        destination = _safe_upload_path(safe_name, digest.hexdigest())
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
        temporary = None
        return destination
    finally:
        await upload.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"message": settings.app_name, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health(_: None = Depends(require_api_key)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        engine=settings.rag_engine,
        vector_store=settings.vector_store,
        llm_provider=settings.llm_provider,
        embed_provider=settings.embed_provider,
    )


@app.post("/documents/ingest", response_model=IngestionResponse)
async def ingest_documents(
    _: None = Depends(require_api_key),
    files: list[UploadFile] = File(...),
) -> IngestionResponse:
    if not files:
        raise HTTPException(400, "Ajoutez au moins un document.")
    paths = [await _save_upload(upload) for upload in files]
    try:
        return await run_in_threadpool(get_rag_service().ingest, paths)
    except (RagError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        logger.exception("Échec de l'indexation des documents")
        raise HTTPException(
            503, "Échec de l'indexation. Consultez les logs du serveur."
        ) from error


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _: None = Depends(require_api_key)) -> ChatResponse:
    try:
        return await run_in_threadpool(
            get_rag_service().ask,
            request.question,
            request.history,
        )
    except RagError as error:
        raise HTTPException(400, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        logger.exception("Échec du traitement RAG")
        raise HTTPException(
            503, "Le moteur RAG est indisponible. Consultez les logs du serveur."
        ) from error
