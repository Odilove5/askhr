"""AskHR configuration.

Values come from environment variables that start with ``ASKHR_`` and from a
local ``.env`` file (gitignored). Environment variables win over ``.env``.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASKHR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local model (Ollama). Used by default and always for personal questions.
    ollama_model: str = "hf.co/unsloth/gemma-4-12b-it-GGUF:UD-Q5_K_XL"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = Field(default=8192, gt=0)
    # Bigger window for the contextual notes, which read a whole document (up to ~53k chars).
    ollama_num_ctx_long: int = Field(default=32768, gt=0)
    ollama_reasoning: bool = False

    # Cloud model (Claude API). Off unless explicitly turned on: the kill switch.
    cloud_enabled: bool = False
    anthropic_api_key: SecretStr | None = None
    claude_model_fast: str = "claude-haiku-5-5"
    claude_model_strong: str = "claude-sonnet-5-5"
    claude_max_tokens: int = Field(default=1024, gt=0)

    request_timeout_s: float = Field(default=60.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Read the settings once and reuse them."""
    return Settings()
