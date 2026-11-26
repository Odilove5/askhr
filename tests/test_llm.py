import os
from pathlib import Path

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from pydantic import SecretStr, ValidationError

from askhr.llm import CloudDisabledError, get_chat_model
from askhr.settings import Settings, get_settings

FAKE_KEY = "sk-ant-test-not-a-real-key-123"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide any real ASKHR_ or ANTHROPIC_ variables from the tests."""
    for name in list(os.environ):
        if name.startswith(("ASKHR_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    get_settings.cache_clear()


def make_settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


def test_defaults_are_local_and_cloud_off() -> None:
    settings = make_settings()
    assert settings.ollama_model == "hf.co/unsloth/gemma-4-12b-it-GGUF:UD-Q5_K_XL"
    assert settings.cloud_enabled is False
    assert settings.anthropic_api_key is None


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKHR_OLLAMA_MODEL", "other-model")
    monkeypatch.setenv("ASKHR_CLOUD_ENABLED", "true")
    settings = make_settings()
    assert settings.ollama_model == "other-model"
    assert settings.cloud_enabled is True


def test_env_file_is_read_and_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ASKHR_OLLAMA_MODEL=from-file\nASKHR_CLAUDE_MAX_TOKENS=99\n")
    monkeypatch.setenv("ASKHR_CLAUDE_MAX_TOKENS", "7")
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    assert settings.ollama_model == "from-file"
    assert settings.claude_max_tokens == 7


def test_invalid_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKHR_CLOUD_ENABLED", "maybe")
    with pytest.raises(ValidationError):
        make_settings()


def test_api_key_never_shows_in_output() -> None:
    settings = make_settings(anthropic_api_key=SecretStr(FAKE_KEY))
    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings)
    assert FAKE_KEY not in settings.model_dump_json()


def test_local_model_uses_settings() -> None:
    settings = make_settings(ollama_model="m", ollama_base_url="http://host:1")
    model = get_chat_model("local", settings)
    assert isinstance(model, ChatOllama)
    assert model.model == "m"
    assert model.base_url == "http://host:1"
    assert model.temperature == 0
    assert model.reasoning is False


def test_local_model_overrides_window_and_answer_length() -> None:
    model = get_chat_model("local", make_settings(), num_ctx=32768, num_predict=200)
    assert isinstance(model, ChatOllama)
    assert model.num_ctx == 32768
    assert model.num_predict == 200
    default = get_chat_model("local", make_settings())
    assert isinstance(default, ChatOllama)
    assert default.num_ctx == 8192
    assert default.num_predict is None


def test_local_only_options_are_refused_for_cloud() -> None:
    settings = make_settings(cloud_enabled=True, anthropic_api_key=SecretStr(FAKE_KEY))
    with pytest.raises(ValueError, match="local model"):
        get_chat_model("cloud", settings, num_ctx=32768)


def test_cloud_is_refused_when_disabled_even_with_a_key() -> None:
    settings = make_settings(cloud_enabled=False, anthropic_api_key=SecretStr(FAKE_KEY))
    with pytest.raises(CloudDisabledError, match="ASKHR_CLOUD_ENABLED"):
        get_chat_model("cloud", settings)


@pytest.mark.parametrize("key", [None, "", "   "])
def test_cloud_is_refused_without_a_key(key: str | None) -> None:
    secret = None if key is None else SecretStr(key)
    settings = make_settings(cloud_enabled=True, anthropic_api_key=secret)
    with pytest.raises(CloudDisabledError, match="ASKHR_ANTHROPIC_API_KEY"):
        get_chat_model("cloud", settings)


def test_cloud_ignores_the_generic_anthropic_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    settings = make_settings(cloud_enabled=True)
    with pytest.raises(CloudDisabledError):
        get_chat_model("cloud", settings)


@pytest.mark.parametrize(("strong", "expected"), [(False, "fast-m"), (True, "strong-m")])
def test_cloud_model_picks_fast_or_strong(strong: bool, expected: str) -> None:
    settings = make_settings(
        cloud_enabled=True,
        anthropic_api_key=SecretStr(FAKE_KEY),
        claude_model_fast="fast-m",
        claude_model_strong="strong-m",
    )
    model = get_chat_model("cloud", settings, strong=strong)
    assert isinstance(model, ChatAnthropic)
    assert model.model == expected
    assert model.anthropic_api_key.get_secret_value() == FAKE_KEY
    assert FAKE_KEY not in repr(model)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model kind"):
        get_chat_model("gpu", make_settings())  # type: ignore[arg-type]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
