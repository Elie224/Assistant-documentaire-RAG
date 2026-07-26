from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from pydantic import SecretStr

from app.api import app, _load_settings, _safe_upload_path
from app.core.config import Settings, get_settings
from app.core.schemas import DocumentDeletionResponse, DocumentInfo, DocumentListResponse


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


def test_document_lifecycle_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    document_id = "a" * 64

    class Service:
        def list_documents(self) -> DocumentListResponse:
            return DocumentListResponse(
                documents=[
                    DocumentInfo(
                        document_id=document_id,
                        names=["guide.txt"],
                        created_at="2026-07-26 12:00:00",
                        chunks=3,
                    )
                ]
            )

        def reindex_document(self, requested_id: str):
            if requested_id != document_id:
                raise ValueError("Document introuvable.")
            return {"files": ["guide.txt"], "chunks": 1, "engine": "langchain", "vector_store": "chroma"}

        def delete_document(self, requested_id: str) -> DocumentDeletionResponse:
            return DocumentDeletionResponse(
                document_id=requested_id, deleted=requested_id == document_id
            )

    monkeypatch.setattr("app.api.get_rag_service", lambda workspace_id: Service())
    client = _with_api_key(None)
    try:
        listing = client.get("/documents")
        assert listing.status_code == 200
        assert listing.json()["documents"][0]["document_id"] == document_id
        assert client.post(f"/documents/{document_id}/reindex").status_code == 200
        invalid_document_id = "b" * 64
        assert client.post(f"/documents/{invalid_document_id}/reindex").status_code == 404
        assert client.delete(f"/documents/{document_id}").status_code == 200
        assert client.delete(f"/documents/{'b' * 64}").status_code == 404
        assert client.delete("/documents/not-a-hash").status_code == 400
    finally:
        _reset_overrides()


def test_api_key_is_scoped_to_allowed_workspaces() -> None:
    base = get_settings()
    overridden = base.model_copy(
        update={
            "api_key": None,
            "api_key_workspaces": {
                "team-a-token": ["team-a"],
                "admin-token": ["*"],
            },
        }
    )
    app.dependency_overrides[_load_settings] = lambda: overridden
    client = TestClient(app)
    try:
        allowed = {"X-API-Key": "team-a-token", "X-Workspace-ID": "team-a"}
        forbidden = {"X-API-Key": "team-a-token", "X-Workspace-ID": "team-b"}
        admin = {"X-API-Key": "admin-token", "X-Workspace-ID": "team-b"}

        assert client.get("/health", headers=allowed).status_code == 200
        assert client.get("/health", headers=forbidden).status_code == 403
        assert client.get("/health", headers=admin).status_code == 200
        assert client.get("/health", headers={"X-Workspace-ID": "team-a"}).status_code == 401
    finally:
        _reset_overrides()


def test_user_authentication_is_persisted_and_workspace_scoped(tmp_path: Path) -> None:
    base = get_settings()
    overridden = base.model_copy(
        update={"api_key": None, "api_key_workspaces": {}, "auth_db_path": tmp_path / "auth.sqlite3"}
    )
    app.dependency_overrides[_load_settings] = lambda: overridden
    client = TestClient(app)
    try:
        registered = client.post(
            "/auth/register",
            json={"email": "user@example.com", "password": "correct horse battery"},
        )
        assert registered.status_code == 201
        auth = registered.json()
        token = auth["access_token"]
        workspace = auth["workspace_id"]
        headers = {"Authorization": f"Bearer {token}", "X-Workspace-ID": workspace}

        me = client.get("/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["email"] == "user@example.com"
        assert client.get("/health", headers=headers).status_code == 200
        assert client.get("/health", headers={"Authorization": f"Bearer {token}"}).status_code == 403
        assert client.get("/auth/me", headers={"Authorization": "Bearer invalid"}).status_code == 401

        logged_in = client.post(
            "/auth/login",
            json={"email": "USER@example.com", "password": "correct horse battery"},
        )
        assert logged_in.status_code == 200
        assert logged_in.json()["workspace_id"] == workspace
        assert client.post(
            "/auth/register",
            json={"email": "user@example.com", "password": "another password"},
        ).status_code == 400
    finally:
        _reset_overrides()


def test_logout_revokes_bearer_session_and_auth_rate_limit(tmp_path: Path) -> None:
    from app import api as api_module

    base = get_settings()
    overridden = base.model_copy(
        update={"api_key": None, "api_key_workspaces": {}, "auth_db_path": tmp_path / "auth.sqlite3"}
    )
    app.dependency_overrides[_load_settings] = lambda: overridden
    api_module._AUTH_ATTEMPTS.clear()
    client = TestClient(app)
    try:
        registered = client.post(
            "/auth/register",
            json={"email": "logout@example.com", "password": "correct horse battery"},
        )
        auth = registered.json()
        headers = {
            "Authorization": f"Bearer {auth['access_token']}",
            "X-Workspace-ID": auth["workspace_id"],
        }
        assert client.post("/auth/logout", headers=headers).status_code == 204
        assert client.get("/auth/me", headers=headers).status_code == 401

        for _ in range(9):
            client.post("/auth/login", json={"email": "missing@example.com", "password": "wrong"})
        assert client.post(
            "/auth/login", json={"email": "missing@example.com", "password": "wrong"}
        ).status_code == 429
    finally:
        _reset_overrides()
        api_module._AUTH_ATTEMPTS.clear()


def test_change_password_revokes_existing_sessions(tmp_path: Path) -> None:
    base = get_settings()
    overridden = base.model_copy(
        update={"api_key": None, "api_key_workspaces": {}, "auth_db_path": tmp_path / "auth.sqlite3"}
    )
    app.dependency_overrides[_load_settings] = lambda: overridden
    from app import api as api_module
    api_module._AUTH_ATTEMPTS.clear()
    client = TestClient(app)
    try:
        response = client.post(
            "/auth/register",
            json={"email": "password@example.com", "password": "old password"},
        )
        auth = response.json()
        headers = {
            "Authorization": f"Bearer {auth['access_token']}",
            "X-Workspace-ID": auth["workspace_id"],
        }
        assert client.post(
            "/auth/password",
            headers=headers,
            json={"current_password": "old password", "new_password": "new password"},
        ).status_code == 204
        assert client.get("/auth/me", headers=headers).status_code == 401
        login_response = client.post(
            "/auth/login",
            json={"email": "password@example.com", "password": "new password"},
        )
        assert login_response.status_code == 200
        assert client.post(
            "/auth/login",
            json={"email": "password@example.com", "password": "old password"},
        ).status_code == 401
    finally:
        _reset_overrides()
        api_module._AUTH_ATTEMPTS.clear()