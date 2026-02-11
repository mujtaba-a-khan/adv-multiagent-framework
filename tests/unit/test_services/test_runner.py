"""Tests for the fine-tuning job runner — cancellation and dispatch."""

from __future__ import annotations

import pytest

from adversarial_framework.services.finetuning.runner import (
    CancelledError,
    _cancelled_jobs,
    is_cancelled,
    request_cancellation,
)


class TestCancellationRegistry:
    """Cooperative cancellation is via an in-memory set."""

    def setup_method(self) -> None:
        _cancelled_jobs.clear()

    def test_initially_not_cancelled(self):
        assert is_cancelled("job-1") is False

    def test_request_then_check(self):
        request_cancellation("job-1")
        assert is_cancelled("job-1") is True

    def test_cancel_specific_job(self):
        request_cancellation("job-1")
        assert is_cancelled("job-2") is False

    def test_multiple_jobs(self):
        request_cancellation("job-1")
        request_cancellation("job-2")
        assert is_cancelled("job-1") is True
        assert is_cancelled("job-2") is True

    def test_idempotent_cancel(self):
        """Calling request_cancellation twice is safe."""
        request_cancellation("job-1")
        request_cancellation("job-1")
        assert is_cancelled("job-1") is True


class TestCancelledError:
    def test_is_exception(self):
        assert issubclass(CancelledError, Exception)

    def test_message(self):
        err = CancelledError("Job abc was cancelled")
        assert "abc" in str(err)

    def test_raise_and_catch(self):
        with pytest.raises(CancelledError, match="cancelled"):
            raise CancelledError("Job was cancelled")
