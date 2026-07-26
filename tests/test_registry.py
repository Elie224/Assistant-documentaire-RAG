from pathlib import Path

from app.core.rag import (
    _file_hash,
    _load_registry,
    _registry_path,
    _registry_transaction,
    _split_new_paths,
)


def test_registry_writes_atomically(tmp_path: Path) -> None:
    with _registry_transaction(tmp_path) as registry:
        registry.setdefault("abc", []).append("doc.txt")

    assert _registry_path(tmp_path).exists()
    data = _load_registry(tmp_path)
    assert data["abc"] == ["doc.txt"]


def test_registry_deduplicates_same_content_with_different_names(tmp_path: Path) -> None:
    first = tmp_path / "guide.txt"
    second = tmp_path / "copie.txt"
    first.write_text("Même contenu", encoding="utf-8")
    second.write_text("Même contenu", encoding="utf-8")
    digest = _file_hash(first)

    kept, pairs = _split_new_paths([first, second], {})
    assert kept == [first]
    assert pairs == [(first, digest)]

    kept, pairs = _split_new_paths([second], {digest: [first.name]})
    assert kept == []
    assert pairs == []


def test_registry_creates_interprocess_lock_file(tmp_path: Path) -> None:
    with _registry_transaction(tmp_path):
        assert (tmp_path / ".indexed_files.lock").exists()