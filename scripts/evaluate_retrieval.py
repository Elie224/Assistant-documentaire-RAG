from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import Settings
from app.core.rag import LangChainBackend


def evaluate(dataset_path: Path) -> dict[str, float | int]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    settings = Settings()
    backend = LangChainBackend(settings)
    store = backend._store()
    hits = 0
    reciprocal_rank = 0.0

    for item in dataset:
        results = backend._search(item["question"], store)
        sources = [
            str(document.metadata.get("source", ""))
            for document, _ in results
        ]
        expected_source = item["expected_source"]
        if expected_source in sources:
            hits += 1
            reciprocal_rank += 1 / (sources.index(expected_source) + 1)

    total = len(dataset)
    return {
        "questions": total,
        "recall_at_k": hits / total if total else 0.0,
        "mrr": reciprocal_rank / total if total else 0.0,
        "top_k": settings.top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue le retrieval du projet RAG.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/rag_dataset.json"),
        help="Chemin vers le jeu de questions JSON.",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
