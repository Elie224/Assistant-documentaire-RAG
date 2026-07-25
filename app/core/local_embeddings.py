import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from llama_index.core.embeddings import BaseEmbedding


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def embed_locally(text: str, dimension: int) -> list[float]:
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


class LocalLangChainEmbeddings(Embeddings):
    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_locally(text, self.dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return embed_locally(text, self.dimension)


class LocalLlamaIndexEmbedding(BaseEmbedding):
    dimension: int = 768

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed_locally(query, self.dimension)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed_locally(text, self.dimension)
