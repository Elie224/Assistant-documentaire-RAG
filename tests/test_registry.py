from pathlib import Path

from app.core.rag import _registry_path, _registry_transaction, _load_registry


def test_registry_writes_atomically(tmp_path: Path) -> None:
    with _registry_transaction(tmp_path) as registry:
        registry.setdefault("abc", []).append("doc.txt")

    assert _registry_path(tmp_path).exists()
    data = _load_registry(tmp_path)
    assert data["abc"] == ["doc.txt"]
