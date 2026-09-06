from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CivicNexus"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./storage/civicnexus.db"
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = ""
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_persist_directory: str = "./storage/chroma"
    retrieval_collection_name: str = "civic_cases"
    bm25_index_path: str = "./storage/retrieval/bm25_index.json"
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = Field(default=64, ge=1, le=256)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    embedding_max_retries: int = Field(default=3, ge=0, le=10)
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    max_review_rounds: int = 2
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_tokens: int = Field(default=256, ge=64, le=2048)
    context_max_units: int = Field(default=3000, ge=500)
    context_recent_messages: int = Field(default=6, ge=1, le=20)
    event_retention_days: int = Field(default=30, ge=1)
    session_memory_retention_days: int = Field(default=30, ge=1)
    task_cache_ttl_seconds: int = Field(default=86400, ge=60)
    skill_directory: str = "configs/skills"
    admin_username: str = "admin"
    admin_password: SecretStr = Field(default_factory=lambda: SecretStr("change-me"))
    admin_token_ttl_seconds: int = Field(default=28800, ge=300)
    user_username: str = "demo"
    user_password: SecretStr = Field(default_factory=lambda: SecretStr("demo-me"))
    user_id: str = "demo-user"
    user_display_name: str = "演示用户"
    user_token_ttl_seconds: int = Field(default=28800, ge=300)
    postgres_enabled: bool = True
    redis_enabled: bool = True
    use_local_storage: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
