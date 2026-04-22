# ==========================================================================
# Builder Stage
# ==========================================================================
FROM python:3.12-slim AS builder

# 1. Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Install build dependencies (needed to compile certain Python packages)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy only the files needed for dependency resolution
COPY pyproject.toml uv.lock ./

# 4. Sync dependencies into a local .venv (using cache for speed)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ==========================================================================
# Final Stage
# ==========================================================================
FROM python:3.12-slim

WORKDIR /app

RUN mkdir -p /app/data /app/db && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# 1. Install RUNTIME system dependencies 
# - libgomp1: required by LightGBM/XGBoost
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH"

COPY --chown=appuser:appuser --from=builder /app/.venv /app/.venv

COPY --chown=appuser:appuser ./src/ ./

USER appuser

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]