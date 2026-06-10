import os

import pytest

from structured_output.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, load_dotenv, load_settings


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)


def test_load_settings_requires_key(clean_env):
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        load_settings()


def test_load_settings_defaults(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = load_settings()
    assert settings.anthropic_api_key == "sk-test"
    assert settings.model == DEFAULT_MODEL
    assert settings.max_tokens == DEFAULT_MAX_TOKENS


def test_load_settings_env_overrides(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-fake")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "100")
    settings = load_settings()
    assert settings.model == "claude-fake"
    assert settings.max_tokens == 100


def test_load_dotenv_reads_file(tmp_path, clean_env):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nANTHROPIC_API_KEY="sk-dot"\nANTHROPIC_MODEL=claude-from-file\n\n'
    )
    load_dotenv(env_file)
    try:
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-dot"
        assert os.environ["ANTHROPIC_MODEL"] == "claude-from-file"
    finally:
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
            os.environ.pop(k, None)


def test_load_dotenv_does_not_override_existing(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-existing")
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-from-file\n")
    load_dotenv(env_file)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-existing"


def test_load_dotenv_missing_file_noop(tmp_path, clean_env):
    load_dotenv(tmp_path / "nope.env")  # must not raise
