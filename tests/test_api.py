from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_exposes_active_stack() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["engine"] in {"langchain", "llamaindex"}
    assert body["vector_store"] in {"chroma", "faiss"}


def test_ingestion_rejects_unsupported_file() -> None:
    response = client.post(
        "/documents/ingest",
        files={"files": ("malware.exe", b"content", "application/octet-stream")},
    )

    assert response.status_code == 415
