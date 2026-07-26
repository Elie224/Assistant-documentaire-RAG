import json
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


def test_legacy_json_registry_is_migrated_to_sqlite(tmp_path: Path) -> None:
    legacy = tmp_path / "indexed_files.json"
    legacy.write_text(
        json.dumps({"abc": ["guide.txt"]}, ensure_ascii=False), encoding="utf-8"
    )

    registry = _load_registry(tmp_path)

    assert registry == {"abc": ["guide.txt"]}
    assert _registry_path(tmp_path).name == "documents.sqlite3"
    assert _registry_path(tmp_path).exists()


def test_legacy_registry_is_not_restored_after_deletion(tmp_path: Path) -> None:
    legacy = tmp_path / "indexed_files.json"
    legacy.write_text(json.dumps({"abc": ["guide.txt"]}), encoding="utf-8")
    assert _load_registry(tmp_path) == {"abc": ["guide.txt"]}

    with _registry_transaction(tmp_path) as registry:
        registry.clear()

    assert _load_registry(tmp_path) == {}
