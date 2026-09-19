"""Application settings (env vars / .env)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Jev via OpenRouter (no waitlist).
    openrouter_api_key: str | None = None
    jev_model: str = "typesafe/jev-1.13"
    jev_decisions_url: str = "https://openrouter.ai/api/alpha/decisions"

    # Jev via TypeSafe direct. Takes precedence over OpenRouter when both are set.
    typesafe_api_key: str | None = None
    typesafe_model: str = "jev-latest"
    typesafe_systemone_url: str = "https://api.typesafe.ai/v1/systemone"

    # Optional generative hook (e.g. gpt-oss-120b on the Spark's llama-server).
    # Unset LLM_BASE_URL means every escalation is queued for human review.
    llm_base_url: str | None = None
    llm_model: str = "gpt-oss-120b"
    llm_api_key: str | None = None

    runs_dir: Path = Path("runs")
    timeout_s: float = 60.0


def get_settings() -> Settings:
    return Settings()
