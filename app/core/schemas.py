from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    score: float | None = None
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    engine: str
    vector_store: str


class IngestionResponse(BaseModel):
    files: list[str]
    chunks: int
    engine: str
    vector_store: str


class HealthResponse(BaseModel):
    status: str
    engine: str
    vector_store: str
    llm_provider: str
    embed_provider: str
