from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Agentic Translation QA"
    app_version: str = "0.1.0"
    ollama_model: str = "gemma3:4b"
    translation_model: str = "Helsinki-NLP/opus-mt-ko-en"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 3
    max_attempts: int = 3
    quality_pass_threshold: float = 0.8
    ollama_base_url: str = "http://localhost:11434"
    data_dir: str = "data"
    outputs_dir: str = "outputs/benchmark_results"
    request_timeout_s: float = 90.0
    log_level: str = "INFO"
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "agentic-translation-qa"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def outputs_path(self) -> Path:
        return Path(self.outputs_dir)


settings = Settings()
