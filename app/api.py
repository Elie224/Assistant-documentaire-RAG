from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings

def _load_settings() -> Settings:
    return get_settings()
from app.core.documents import SUPPORTED_EXTENSIONS
from app.core.exceptions import RagError
from app.core.rag import RagService
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestionResponse,
)


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
    if not x_api_key or x_api_key != expected:
        raise HTTPException(401, "Clé API invalide ou manquante.")


@lru_cache
def get_rag_service() -> RagService:
    return RagService(settings)


def _safe_upload_path(filename: str | None) -> Path:
    safe_name = Path(filename or "document").name
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(415, f"Format non pris en charge. Formats acceptés : {formats}")
    return settings.uploads_dir / safe_name


async def _save_upload(upload: UploadFile) -> Path:
    destination = _safe_upload_path(upload.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0

    try:
        with temporary.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        413,
                        f"{destination.name} dépasse la limite de {settings.max_upload_mb} Mo.",
                    )
                output.write(chunk)
        temporary.replace(destination)
    finally:
        await upload.close()
        if temporary.exists():
            temporary.unlink()
    return destination


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
        raise HTTPException(503, f"Échec de l'indexation : {error}") from error


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
        raise HTTPException(503, f"Le moteur RAG est indisponible : {error}") from error
