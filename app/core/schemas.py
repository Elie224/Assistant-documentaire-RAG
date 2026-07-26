from typing import Literal

from pydantic import BaseModel, EmailStr, Field


DocumentStatus = Literal[
    "pending",
    "processing",
    "indexed",
    "failed",
    "deleting",
    "reindexing",
]


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
    confidence: float | None = None
    preview: str
    content: str = ""


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


class DocumentInfo(BaseModel):
    document_id: str
    names: list[str]
    created_at: str
    chunks: int
    status: DocumentStatus = "indexed"


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DocumentDeletionResponse(BaseModel):
    document_id: str
    deleted: bool


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user_id: str
    workspace_id: str


class UserResponse(BaseModel):
    user_id: str
    email: str
    workspace_id: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=8, max_length=256)
