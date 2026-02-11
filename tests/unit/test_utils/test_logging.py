"""Tests for adversarial_framework.utils.logging module."""

from __future__ import annotations

import logging

import structlog

from adversarial_framework.utils.logging import setup_logging


class TestSetupLogging:
    def test_sets_info_level_by_default(self):
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_sets_debug_level(self):
        setup_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_sets_warning_level(self):
        setup_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_sets_error_level(self):
        setup_logging(level="ERROR")
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_case_insensitive_level(self):
        setup_logging(level="debug")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_invalid_level_defaults_to_info(self):
        setup_logging(level="NONEXISTENT")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_json_output_mode(self):
        # Should not raise
        setup_logging(json_output=True)
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_console_output_mode(self):
        # Should not raise
        setup_logging(json_output=False)

    def test_structlog_configured(self):
        setup_logging()
        log = structlog.get_logger("test")
        assert log is not None

    def test_handlers_have_formatter(self):
        setup_logging()
        root = logging.getLogger()
        for handler in root.handlers:
            assert handler.formatter is not None

    def test_multiple_calls_no_error(self):
        # Calling setup_logging multiple times should not raise
        setup_logging(level="INFO")
        setup_logging(level="DEBUG")
        setup_logging(level="INFO", json_output=True)
