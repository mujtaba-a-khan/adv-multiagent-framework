"""Tests for adversarial_framework.config.settings module."""

from __future__ import annotations

import os
from unittest import mock

from pytest import approx

from adversarial_framework.config.settings import (
    Settings,
    get_settings,
)


def _settings(**overrides: object) -> Settings:
    """Create Settings ignoring .env file for deterministic tests."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestSettingsDefaults:
    def test_default_ollama_base_url(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_attacker_model(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.attacker_model == "phi4-reasoning:14b"

    def test_default_analyzer_model(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.analyzer_model == "phi4-reasoning:14b"

    def test_default_defender_model(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.defender_model == "qwen3:8b"

    def test_default_target_model(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.target_model == "llama3:8b"

    def test_default_max_turns(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.max_turns == 20

    def test_default_max_cost(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.max_cost_usd == approx(50.0)

    def test_default_max_errors(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.max_errors == 5

    def test_default_log_level(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.log_level == "INFO"

    def test_default_log_json(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.log_json is False

    def test_default_environment(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.environment == "development"

    def test_default_api_host(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.api_host == "0.0.0.0"

    def test_default_api_port(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.api_port == 8000

    def test_default_cors_origins(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert "http://localhost:3000" in s.cors_origins

    def test_default_redis_url(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.redis_url == "redis://localhost:6379/0"

    def test_default_daily_max_cost(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.daily_max_cost_usd == approx(200.0)

    def test_default_api_keys_empty(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.openai_api_key == ""
        assert s.anthropic_api_key == ""
        assert s.google_api_key == ""

    def test_default_database_urls_empty(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.database_url == ""
        assert s.database_url_direct == ""


class TestSettingsEnvOverrides:
    def test_override_ollama_base_url(self):
        env = {"ADV_OLLAMA_BASE_URL": "http://gpu:11434"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.ollama_base_url == "http://gpu:11434"

    def test_override_attacker_model(self):
        env = {"ADV_ATTACKER_MODEL": "custom:latest"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.attacker_model == "custom:latest"

    def test_override_max_turns(self):
        env = {"ADV_MAX_TURNS": "50"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.max_turns == 50

    def test_override_max_cost(self):
        env = {"ADV_MAX_COST_USD": "100.0"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.max_cost_usd == approx(100.0)

    def test_override_log_level(self):
        env = {"ADV_LOG_LEVEL": "DEBUG"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.log_level == "DEBUG"

    def test_override_log_json(self):
        env = {"ADV_LOG_JSON": "true"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.log_json is True

    def test_override_environment(self):
        env = {"ADV_ENVIRONMENT": "production"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.environment == "production"

    def test_override_api_port(self):
        env = {"ADV_API_PORT": "9000"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.api_port == 9000

    def test_override_redis_url(self):
        env = {"ADV_REDIS_URL": "redis://redis:6379/1"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.redis_url == "redis://redis:6379/1"


class TestSettingsSslFix:
    def test_sslmode_replaced_in_database_url(self):
        env = {
            "ADV_DATABASE_URL": ("postgresql://u:p@host/db?sslmode=require"),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert "ssl=require" in s.database_url
        assert "sslmode=" not in s.database_url

    def test_sslmode_replaced_in_direct_url(self):
        env = {
            "ADV_DATABASE_URL_DIRECT": ("postgresql://u:p@host/db?sslmode=require"),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert "ssl=require" in s.database_url_direct
        assert "sslmode=" not in s.database_url_direct

    def test_no_sslmode_unchanged(self):
        env = {
            "ADV_DATABASE_URL": ("postgresql://u:p@host/db"),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert s.database_url == ("postgresql://u:p@host/db")

    def test_empty_database_url_no_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = _settings()
        assert s.database_url == ""


class TestSettingsExtraIgnored:
    def test_extra_env_vars_ignored(self):
        env = {
            "ADV_NONEXISTENT_FIELD": "should_not_crash",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = _settings()
        assert not hasattr(s, "nonexistent_field")


class TestGetSettings:
    def test_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_returns_fresh_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        # get_settings() creates a new instance each call
        assert s1 is not s2
