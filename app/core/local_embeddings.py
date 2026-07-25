"""Local embedding providers.

Two backends are available:

* ``semantic`` uses :mod:`sentence_transformers` and produces real semantic
  vectors. The model is downloaded the first time it is used.
* ``local-lite`` is a deterministic hashing vectorizer (bag of words + bigrams).
  It has no external dependency but only matches literal vocabulary.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from llama_index.core.embeddings import BaseEmbedding


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _hash_vector(text: str, dimension: int) -> list[float]:
    tokens = TOKEN_PATTERN.findall(text.casefold())
    features = tokens + [
        f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)
    ]
    vector = [0.0] * dimension
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        position = int.from_bytes(digest[:4], "little") % dimension
        vector[position] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class LiteEmbeddings(Embeddings):
    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(text, self.dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_vector(text, self.dimension)


class LiteLlamaIndexEmbedding(BaseEmbedding):
    dimension: int = 768

    def _get_query_embedding(self, query: str) -> list[float]:
        return _hash_vector(query, self.dimension)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return _hash_vector(text, self.dimension)


class SemanticEmbeddings(Embeddings):
    def __init__(self, model):
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._model.encode(texts, normalize_embeddings=True)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.encode([text], normalize_embeddings=True)[0])


class SemanticLlamaIndexEmbedding(BaseEmbedding):
    def __init__(self, model) -> None:
        super().__init__()
        self._model = model

    def _get_query_embedding(self, query: str) -> list[float]:
        return list(self._model.encode([query], normalize_embeddings=True)[0])

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return list(self._model.encode([text], normalize_embeddings=True)[0])


def _load_semantic_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def build_local_embeddings(mode: str, model_name: str, dimension: int) -> Embeddings:
    if mode == "semantic":
        return SemanticEmbeddings(_load_semantic_model(model_name))
    if mode == "local-lite":
        return LiteEmbeddings(dimension=dimension)
    raise ValueError(f"Mode d'embedding local inconnu : {mode}")


def build_llamaindex_embedding(mode: str, model_name: str, dimension: int) -> BaseEmbedding:
    if mode == "semantic":
        return SemanticLlamaIndexEmbedding(_load_semantic_model(model_name))
    if mode == "local-lite":
        return LiteLlamaIndexEmbedding(dimension=dimension)
    raise ValueError(f"Mode d'embedding local inconnu : {mode}")
