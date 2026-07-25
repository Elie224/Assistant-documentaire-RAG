import math

from app.core.local_embeddings import (
    LiteEmbeddings,
    LiteLlamaIndexEmbedding,
    _hash_vector,
)
from app.core.rag import _humanize_answer


def test_local_embedding_is_deterministic_and_normalized() -> None:
    first = _hash_vector("allocation annuelle équipement", 128)
    second = _hash_vector("allocation annuelle équipement", 128)

    assert first == second
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


def test_framework_adapters_share_the_same_vectors() -> None:
    text = "Le support est ouvert à 8 heures."
    langchain = LiteEmbeddings(dimension=128)
    llamaindex = LiteLlamaIndexEmbedding(dimension=128)

    assert langchain.embed_query(text) == llamaindex.get_query_embedding(text)


def test_technical_citations_are_removed_from_answer() -> None:
    answer = "L'allocation est de **250 euros** [1]. Elle finance le matériel [2, 3]."

    assert _humanize_answer(answer) == (
        "L'allocation est de **250 euros**. Elle finance le matériel."
    )


def test_lite_embeddings_have_normalized_vectors() -> None:
    vector = LiteEmbeddings(dimension=64).embed_query("Le télétravail est ouvert à tous.")
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)
