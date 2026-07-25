import math

from app.core.local_embeddings import (
    LocalLangChainEmbeddings,
    LocalLlamaIndexEmbedding,
    embed_locally,
)
from app.core.rag import _humanize_answer


def test_local_embedding_is_deterministic_and_normalized() -> None:
    first = embed_locally("allocation annuelle équipement", 128)
    second = embed_locally("allocation annuelle équipement", 128)

    assert first == second
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


def test_framework_adapters_share_the_same_vectors() -> None:
    text = "Le support est ouvert à 8 heures."
    langchain = LocalLangChainEmbeddings(dimension=128)
    llamaindex = LocalLlamaIndexEmbedding(dimension=128)

    assert langchain.embed_query(text) == llamaindex.get_query_embedding(text)


def test_technical_citations_are_removed_from_answer() -> None:
    answer = "L'allocation est de **250 euros** [1]. Elle finance le matériel [2, 3]."

    assert _humanize_answer(answer) == (
        "L'allocation est de **250 euros**. Elle finance le matériel."
    )
