from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from app.core.config import Settings
from app.core.rag import NO_ANSWER_MESSAGE, RagService

_TOKEN_PATTERN = re.compile(r"\b[\wÀ-ÿ'-]+\b", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def normalize(text: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(text.casefold()))


def token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = Counter(normalize(predicted).split())
    expected_tokens = Counter(normalize(expected).split())
    overlap = sum((predicted_tokens & expected_tokens).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(predicted_tokens.values())
    recall = overlap / sum(expected_tokens.values())
    return 2 * precision * recall / (precision + recall)


def numbers_match(predicted: str, expected: str) -> bool:
    predicted_numbers = set(_NUMBER_PATTERN.findall(predicted.casefold()))
    expected_numbers = set(_NUMBER_PATTERN.findall(expected.casefold()))
    return not expected_numbers or predicted_numbers == expected_numbers


def evaluate(dataset_path: Path) -> dict[str, float | int]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    service = RagService(Settings())
    answer_hits = 0
    exact_hits = 0
    source_hits = 0
    refusal_hits = 0
    f1_total = 0.0
    answer_total = 0
    source_total = 0
    refusal_total = 0

    for item in dataset:
        response = service.ask(item["question"], [])
        expected_answer = item.get("expected_answer", "")
        expected_source = item.get("expected_source")
        should_refuse = bool(item.get("should_refuse"))
        if expected_answer:
            answer_total += 1
            score = token_f1(response.answer, expected_answer)
            f1_total += score
            exact_hits += normalize(response.answer) == normalize(expected_answer)
            if score >= 0.8 and numbers_match(response.answer, expected_answer):
                answer_hits += 1
        if expected_source:
            source_total += 1
            if any(source.source == expected_source for source in response.sources):
                source_hits += 1
        if should_refuse:
            refusal_total += 1
            if (
                not response.sources
                and normalize(response.answer) == normalize(NO_ANSWER_MESSAGE)
            ):
                refusal_hits += 1

    total = len(dataset)
    return {
        "questions": total,
        "answer_accuracy": answer_hits / answer_total if answer_total else 0.0,
        "exact_match": exact_hits / answer_total if answer_total else 0.0,
        "token_f1": f1_total / answer_total if answer_total else 0.0,
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