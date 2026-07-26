from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.core.config import Settings
from app.core.rag import RagService

_TOKEN_PATTERN = re.compile(r"\b[\wÀ-ÿ'-]+\b", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(text.casefold()))


def evaluate(dataset_path: Path) -> dict[str, float | int]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    service = RagService(Settings())
    answer_hits = 0
    source_hits = 0
    refusal_hits = 0
    answer_total = 0
    source_total = 0
    refusal_total = 0

    for item in dataset:
        response = service.ask(item["question"], [])
        answer = normalize(response.answer)
        expected_answer = normalize(item.get("expected_answer", ""))
        expected_source = item.get("expected_source")
        answer_total += bool(expected_answer)
        source_total += bool(expected_source)
        refusal_total += bool(item.get("should_refuse"))
        if expected_answer and expected_answer in answer:
            answer_hits += 1
        if expected_source and any(
            source.source == expected_source for source in response.sources
        ):
            source_hits += 1
        if item.get("should_refuse") and not response.sources:
            refusal_hits += 1

    total = len(dataset)
    return {
        "questions": total,
        "answer_contains_expected": answer_hits / answer_total if answer_total else 0.0,
        "expected_source_recall": source_hits / source_total if source_total else 0.0,
        "refusal_accuracy": refusal_hits / refusal_total if refusal_total else 0.0,
        "top_k": service.settings.top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue les réponses générées par le pipeline RAG."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/generation_dataset.json"),
        help="Chemin vers le jeu de questions JSON.",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()