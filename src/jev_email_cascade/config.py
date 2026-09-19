"""Application settings (env vars / .env)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

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
    # Which model the escalation hook (--llm) calls: "local" = LLM_BASE_URL above,
    # "frontier" = the hosted frontier model below.
    llm_provider: str = "local"

    # Hosted frontier model (OpenAI) -- the reference column in the comparison. Same bare
    # OPENAI_API_KEY / OPENAI_BASE_URL names llama-cpp-spark reads, so one .env serves both.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    frontier_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL"
    )
    frontier_model: str = "gpt-5.6-terra"
    # List prices, $ per million tokens (docs/models/gpt-5.6-terra.md in llama-cpp-spark).
    # The API reports token counts, never dollars, so the run's cost is always list-price math.
    frontier_price_input_per_mtok: float = 2.0
    frontier_price_cached_input_per_mtok: float = 0.20
    frontier_price_output_per_mtok: float = 12.0

    runs_dir: Path = Path("runs")
    timeout_s: float = 60.0


def get_settings() -> Settings:
    return Settings()
