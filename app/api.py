import hashlib
import logging
from collections import OrderedDict
import re
import secrets
import time
from threading import Lock
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.auth import AuthError, authenticate, change_password, login, register, revoke
from app.core.config import Settings, get_settings
from app.core.documents import SUPPORTED_EXTENSIONS
from app.core.exceptions import RagError
from app.core.rag import RagService
from app.core.schemas import (
    AuthResponse,
    ChangePasswordRequest,
    ChatRequest,
    ChatResponse,
    DocumentDeletionResponse,
    DocumentListResponse,
    HealthResponse,
    IngestionResponse,
    LoginRequest,
    RegisterRequest,
    UserResponse,
)


logger = logging.getLogger(__name__)

_AUTH_ATTEMPTS: dict[str, list[float]] = {}
_AUTH_ATTEMPTS_LOCK = Lock()
_AUTH_RATE_LIMIT = 10
_AUTH_RATE_WINDOW = 60.0
_SERVICE_CACHE_MAXSIZE = 256
_SERVICE_CACHE: OrderedDict[tuple, RagService] = OrderedDict()
_SERVICE_CACHE_LOCK = Lock()


def _check_auth_rate_limit(request: Request) -> None:
    key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _AUTH_ATTEMPTS_LOCK:
        attempts = [stamp for stamp in _AUTH_ATTEMPTS.get(key, []) if now - stamp < _AUTH_RATE_WINDOW]
        if len(attempts) >= _AUTH_RATE_LIMIT:
            raise HTTPException(429, "Trop de tentatives. Réessayez dans une minute.")
        attempts.append(now)
        _AUTH_ATTEMPTS[key] = attempts


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
    authorization: str | None = Header(default=None),
    x_workspace_id: str | None = Header(default=None, alias="X-Workspace-ID"),
    active_settings: Settings = Depends(_load_settings),
) -> tuple[str, str, str] | None:
    workspace_id = (x_workspace_id or "default").strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        identity = authenticate(active_settings.project_path(active_settings.auth_db_path), token)
        if identity is None:
            raise HTTPException(401, "Session invalide ou expirée.")
        if workspace_id != identity[2]:
            raise HTTPException(403, "Accès interdit à ce workspace.")
        return identity
    if active_settings.api_key_workspaces:
        allowed_workspaces = next(
            (
                workspaces
                for configured_key, workspaces in active_settings.api_key_workspaces.items()
                if x_api_key and secrets.compare_digest(x_api_key, configured_key)
            ),
            None,
        )
        if allowed_workspaces is None:
            raise HTTPException(401, "Clé API invalide ou manquante.")
        if "*" not in allowed_workspaces and workspace_id not in allowed_workspaces:
            raise HTTPException(403, "Accès interdit à ce workspace.")
        return None
    if active_settings.api_key is None or not active_settings.api_key.get_secret_value():
        if active_settings.allow_unauthenticated:
            return None
        raise HTTPException(401, "Authentification requise.")
    expected = active_settings.api_key.get_secret_value()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Clé API invalide ou manquante.")
    if workspace_id != "default":
        raise HTTPException(403, "La clé API historique est limitée au workspace default.")
    return None


def workspace_settings(
    x_workspace_id: str | None = Header(default=None, alias="X-Workspace-ID"),
    active_settings: Settings = Depends(_load_settings),
) -> Settings:
    try:
        return active_settings.for_workspace(x_workspace_id)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


def get_rag_service(active_settings: Settings) -> RagService:
    cache_key = (
        active_settings.workspace_id,
        active_settings.rag_engine,
        active_settings.vector_store,
        active_settings.llm_provider,
        active_settings.llm_model,
        active_settings.anthropic_model,
        active_settings.ollama_llm_model,
        active_settings.embed_provider,
        active_settings.embed_model,
        active_settings.ollama_embed_model,
        active_settings.local_semantic_model,
        active_settings.local_embed_dimension,
        active_settings.chunk_size,
        active_settings.chunk_overlap,
        active_settings.top_k,
        active_settings.score_threshold,
        active_settings.hybrid_search,
        active_settings.bm25_weight,
    )
    with _SERVICE_CACHE_LOCK:
        service = _SERVICE_CACHE.get(cache_key)
        if service is None:
            service = RagService(active_settings)
            _SERVICE_CACHE[cache_key] = service
            if len(_SERVICE_CACHE) > _SERVICE_CACHE_MAXSIZE:
                _SERVICE_CACHE.popitem(last=False)
            return service
        _SERVICE_CACHE.move_to_end(cache_key)
        return service


def _safe_upload_path(
    filename: str | None,
    content_hash: str | None = None,
    active_settings: Settings | None = None,
) -> Path:
    current_settings = active_settings or settings
    safe_name = Path(filename or "document").name
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(415, f"Format non pris en charge. Formats acceptés : {formats}")
    destination = current_settings.uploads_dir
    if content_hash:
        destination /= content_hash
    return destination / safe_name


async def _save_upload(
    upload: UploadFile, active_settings: Settings | None = None
) -> Path:
    current_settings = active_settings or settings
    temporary: Path | None = None
    try:
        safe_name = Path(upload.filename or "document").name
        _safe_upload_path(safe_name, active_settings=current_settings)
        current_settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        temporary = current_settings.uploads_dir / f".{uuid.uuid4().hex}.part"
        max_bytes = current_settings.max_upload_mb * 1024 * 1024
        written = 0
        digest = hashlib.sha256()

        with temporary.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        413,
                        f"{safe_name} dépasse la limite de {current_settings.max_upload_mb} Mo.",
                    )
                digest.update(chunk)
                output.write(chunk)

        destination = _safe_upload_path(
            safe_name, digest.hexdigest(), active_settings=current_settings
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
        temporary = None
        return destination
    finally:
        await upload.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()


@app.post("/auth/register", response_model=AuthResponse, status_code=201)
def auth_register(request: RegisterRequest, http_request: Request, active_settings: Settings = Depends(_load_settings)) -> AuthResponse:
    _check_auth_rate_limit(http_request)
    try:
        user_id, workspace_id = register(
            active_settings.project_path(active_settings.auth_db_path),
            request.email,
            request.password,
        )
        token, _, _, expires_in = login(
            active_settings.project_path(active_settings.auth_db_path),
            request.email,
            request.password,
            active_settings.session_ttl_hours,
        )
    except AuthError as error:
        raise HTTPException(400, str(error)) from error
    return AuthResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=user_id,
        workspace_id=workspace_id,
    )


@app.post("/auth/login", response_model=AuthResponse)
def auth_login(request: LoginRequest, http_request: Request, active_settings: Settings = Depends(_load_settings)) -> AuthResponse:
    _check_auth_rate_limit(http_request)
    try:
        token, user_id, workspace_id, expires_in = login(
            active_settings.project_path(active_settings.auth_db_path),
            request.email,
            request.password,
            active_settings.session_ttl_hours,
        )
    except AuthError as error:
        raise HTTPException(401, str(error)) from error
    return AuthResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=user_id,
        workspace_id=workspace_id,
    )


@app.post("/auth/password", status_code=204)
def auth_change_password(
    request: ChangePasswordRequest,
    identity: tuple[str, str, str] | None = Depends(require_api_key),
    active_settings: Settings = Depends(_load_settings),
) -> None:
    if identity is None:
        raise HTTPException(401, "Une session Bearer est requise.")
    try:
        change_password(
            active_settings.project_path(active_settings.auth_db_path),
            identity[0],
            request.current_password,
            request.new_password,
        )
    except AuthError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/auth/logout", status_code=204)
def auth_logout(
    authorization: str | None = Header(default=None),
    active_settings: Settings = Depends(_load_settings),
) -> None:
    if authorization and authorization.lower().startswith("bearer "):
        revoke(
            active_settings.project_path(active_settings.auth_db_path),
            authorization[7:].strip(),
        )


@app.get("/auth/me", response_model=UserResponse)
def auth_me(
    identity: tuple[str, str, str] | None = Depends(require_api_key),
) -> UserResponse:
    if identity is None:
        raise HTTPException(401, "Une session Bearer est requise.")
    return UserResponse(user_id=identity[0], email=identity[1], workspace_id=identity[2])


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"message": settings.app_name, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health(
    _: None = Depends(require_api_key),
    active_settings: Settings = Depends(workspace_settings),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        engine=active_settings.rag_engine,
        vector_store=active_settings.vector_store,
        llm_provider=active_settings.llm_provider,
        embed_provider=active_settings.embed_provider,
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    _: None = Depends(require_api_key),
    active_settings: Settings = Depends(workspace_settings),
) -> DocumentListResponse:
    service = get_rag_service(active_settings)
    return await run_in_threadpool(service.list_documents)


@app.post("/documents/{document_id}/reindex", response_model=IngestionResponse)
async def reindex_document(
    document_id: str,
    _: None = Depends(require_api_key),
    active_settings: Settings = Depends(workspace_settings),
) -> IngestionResponse:
    if not re.fullmatch(r"[0-9a-f]{64}", document_id):
        raise HTTPException(400, "Identifiant documentaire invalide.")
    service = get_rag_service(active_settings)
    try:
        return await run_in_threadpool(service.reindex_document, document_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    except RagError as error:
        raise HTTPException(400, str(error)) from error


@app.delete(
    "/documents/{document_id}", response_model=DocumentDeletionResponse
)
async def delete_document(
    document_id: str,
    _: None = Depends(require_api_key),
    active_settings: Settings = Depends(workspace_settings),
) -> DocumentDeletionResponse:
    if not re.fullmatch(r"[0-9a-f]{64}", document_id):
        raise HTTPException(400, "Identifiant documentaire invalide.")
    service = get_rag_service(active_settings)
    try:
        response = await run_in_threadpool(service.delete_document, document_id)
    except (RagError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    if not response.deleted:
        raise HTTPException(404, "Document introuvable.")
    return response




def _cleanup_failed_uploads(paths: list[Path], existing_ids: set[str]) -> None:
    for path in paths:
        document_id = path.parent.name
        if document_id in existing_ids:
            continue
        try:
            path.unlink(missing_ok=True)
            if path.parent.is_dir() and not any(path.parent.iterdir()):
                path.parent.rmdir()
        except OSError:
            logger.warning("Impossible de nettoyer l'upload échoué: %s", path)

@app.post("/documents/ingest", response_model=IngestionResponse)
async def ingest_documents(
    _: None = Depends(require_api_key),
    active_settings: Settings = Depends(workspace_settings),
    files: list[UploadFile] = File(...),
) -> IngestionResponse:
    if not files:
        raise HTTPException(400, "Ajoutez au moins un document.")
    service = get_rag_service(active_settings)
    try:
        listing = await run_in_threadpool(service.list_documents)
        existing_ids = {document.document_id for document in listing.documents}
    except Exception as error:
        logger.exception("Impossible de lire le registre documentaire")
        raise HTTPException(
            503, "Le registre documentaire est indisponible."
        ) from error
    paths: list[Path] = []
    try:
        for upload in files:
            paths.append(await _save_upload(upload, active_settings))
        return await run_in_threadpool(service.ingest, paths)
    except HTTPException:
        _cleanup_failed_uploads(paths, existing_ids)
        raise
    except (RagError, ValueError) as error:
        _cleanup_failed_uploads(paths, existing_ids)
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        _cleanup_failed_uploads(paths, existing_ids)
        logger.exception("Échec de l'indexation des documents")
        raise HTTPException(
            503, "Échec de l'indexation. Consultez les logs du serveur."
        ) from error


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: None = Depends(require_api_key),
    active_settings: Settings = Depends(workspace_settings),
) -> ChatResponse:
    try:
        service = get_rag_service(active_settings)
        return await run_in_threadpool(
            service.ask,
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