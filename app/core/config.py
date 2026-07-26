from functools import lru_cache
from pathlib import Path
import re
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Assistant RAG"
    environment: Literal["development", "test", "production"] = "development"
    allow_unauthenticated: bool = False
    workspace_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
    )
    rag_engine: Literal["langchain", "llamaindex"] = "langchain"
    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    embed_provider: Literal["semantic", "local-lite", "openai", "ollama"] = "local-lite"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    local_embed_dimension: int = Field(default=768, ge=64, le=4096)
    local_semantic_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    hybrid_search: bool = Field(default=True)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    vector_store: Literal["chroma", "faiss"] = "chroma"
    chroma_dir: Path = Path("./data/index/chroma")
    faiss_dir: Path = Path("./data/index/faiss")
    raw_data_dir: Path = Path("./data/raw")
    auth_db_path: Path = Path("./data/auth.sqlite3")
    session_ttl_hours: int = Field(default=24, ge=1, le=720)

    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    score_threshold: float = Field(default=0.25, ge=0, le=1)
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_API_KEY", "API_KEY", "api_key"),
    )
    api_key_workspaces: dict[str, list[str]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "RAG_API_KEY_WORKSPACES", "API_KEY_WORKSPACES", "api_key_workspaces"
        ),
    )

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP doit être inférieur à CHUNK_SIZE.")
        if self.environment == "production" and self.allow_unauthenticated:
            raise ValueError(
                "ALLOW_UNAUTHENTICATED doit être false en production."
            )
        if self.rag_engine == "llamaindex" and self.hybrid_search:
            raise ValueError(
                "HYBRID_SEARCH est disponible uniquement avec RAG_ENGINE=langchain."
            )
        return self

    def project_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def index_dir(self) -> Path:
        base = self.chroma_dir if self.vector_store == "chroma" else self.faiss_dir
        return self.project_path(base) / self.rag_engine

    @property
    def isolated_index_dir(self) -> Path:
        """Chemin d'index isolé par workspace, moteur et provider d'embeddings."""
        base = self.index_dir
        if self.workspace_id != "default":
            base /= self.workspace_id
        return base / self.embed_provider

    @property
    def uploads_dir(self) -> Path:
        base = self.project_path(self.raw_data_dir)
        if self.workspace_id != "default":
            base /= self.workspace_id
        return base

    def for_workspace(self, workspace_id: str | None) -> "Settings":
        value = (workspace_id or "default").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
            raise ValueError("Identifiant de workspace invalide.")
        return self.model_copy(update={"workspace_id": value})

    def openai_key(self) -> str:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY est requis pour utiliser OpenAI.")
        return self.openai_api_key.get_secret_value()

    def anthropic_key(self) -> str:
        if self.anthropic_api_key is None or not self.anthropic_api_key.get_secret_value():
            raise ValueError("ANTHROPIC_API_KEY est requis pour utiliser Anthropic.")
        return self.anthropic_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
