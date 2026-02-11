# Multi-stage Python backend Dockerfile
# Uses uv for fast, reproducible dependency installation

# ── Stage 1: Build
# Pin: run `docker pull python:3.12-slim` and update digest
FROM python:3.12-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab AS builder

# Install uv (pinned to minor version — update when upgrading uv.lock)
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps in production, include finetuning extras)
RUN uv sync --frozen --no-dev --extra finetuning --no-install-project

# Copy source code and README (required by hatchling build)
COPY README.md ./
COPY src/ src/
COPY configs/ configs/
COPY prompt_templates/ prompt_templates/

# Install the project itself
RUN uv sync --frozen --no-dev --extra finetuning

# ── Stage 2: Runtime (same digest as builder)
FROM python:3.12-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab AS runtime

# Security: run as non-root
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/false appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/prompt_templates /app/prompt_templates

# Copy alembic config if present
COPY alembic.ini* ./
COPY src/adversarial_framework/db/migrations/ src/adversarial_framework/db/migrations/

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV ADV_ENVIRONMENT=production
ENV ADV_LOG_JSON=true

# Non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Run with uvicorn
CMD ["uvicorn", "adversarial_framework.api.app:create_app", \
    "--factory", "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "2", "--loop", "uvloop", "--http", "httptools"]
