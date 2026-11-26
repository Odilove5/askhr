"""One place that builds chat models, local (Ollama) or cloud (Claude API).

The cloud branch fails closed: if the cloud is turned off or no key is set,
it raises instead of quietly falling back to another model.
"""

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from askhr.settings import Settings, get_settings

ModelKind = Literal["local", "cloud"]


class CloudDisabledError(RuntimeError):
    """The cloud model was requested but is not allowed or not configured."""


def get_chat_model(
    kind: ModelKind, settings: Settings | None = None, *, strong: bool = False
) -> BaseChatModel:
    """Return a chat model. ``strong`` picks the stronger Claude model (cloud only)."""
    settings = settings or get_settings()
    if kind == "local":
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            reasoning=settings.ollama_reasoning,
            temperature=0,
        )
    if kind == "cloud":
        if not settings.cloud_enabled:
            raise CloudDisabledError("Cloud models are off. Set ASKHR_CLOUD_ENABLED=true to allow.")
        key = settings.anthropic_api_key
        if key is None or not key.get_secret_value().strip():
            raise CloudDisabledError("No Claude API key. Set ASKHR_ANTHROPIC_API_KEY in .env.")
        model = settings.claude_model_strong if strong else settings.claude_model_fast
        return ChatAnthropic(
            model_name=model,
            api_key=key,
            max_tokens_to_sample=settings.claude_max_tokens,
            timeout=settings.request_timeout_s,
            max_retries=2,
            temperature=0,
            stop=None,
        )
    raise ValueError(f"Unknown model kind: {kind!r}")


def main() -> None:
    """Smoke test: ask the local model one question through a tiny LCEL chain."""
    settings = get_settings()
    chain = get_chat_model("local", settings) | StrOutputParser()
    print(f"model: {settings.ollama_model}")
    print(chain.invoke("Réponds en un seul mot, en français : quelle est la capitale du Canada ?"))


if __name__ == "__main__":
    main()
