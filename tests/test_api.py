import pytest
from fastapi.testclient import TestClient

from pydantic import SecretStr

from app.api import app, _load_settings, _safe_upload_path
from app.core.config import get_settings


def _with_api_key(token: str | None) -> TestClient:
    base = get_settings()
    api_key = SecretStr(token) if token else None
    overridden = base.model_copy(update={"api_key": api_key})
    app.dependency_overrides[_load_settings] = lambda: overridden
    return TestClient(app)


def _reset_overrides() -> None:
    app.dependency_overrides.pop(_load_settings, None)


def test_health_exposes_active_stack() -> None:
    client = _with_api_key(None)
    try:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["engine"] in {"langchain", "llamaindex"}
        assert body["vector_store"] in {"chroma", "faiss"}
    finally:
        _reset_overrides()


def test_ingestion_rejects_unsupported_file() -> None:
    client = _with_api_key(None)
    try:
        response = client.post(
            "/documents/ingest",
            files={"files": ("malware.exe", b"content", "application/octet-stream")},
        )
        assert response.status_code == 415
    finally:
        _reset_overrides()


def test_api_key_is_required_when_configured() -> None:
    client = _with_api_key("secret-token")
    try:
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers={"X-API-Key": "nope"}).status_code == 401
        assert (
            client.get("/health", headers={"X-API-Key": "secret-token"}).status_code == 200
        )
    finally:
        _reset_overrides()


def test_upload_path_is_partitioned_by_content_hash() -> None:
    destination = _safe_upload_path("contrat.pdf", "abc123")

    assert destination.name == "contrat.pdf"
    assert destination.parent.name == "abc123"


def test_workspace_header_is_validated() -> None:
    client = _with_api_key(None)
    try:
        assert client.get("/health", headers={"X-Workspace-ID": "team-a"}).status_code == 200
        assert client.get("/health", headers={"X-Workspace-ID": "../other"}).status_code == 400
    finally:
        _reset_overrides()