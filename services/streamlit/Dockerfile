# Multi-Stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install Poetry
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install poetry

# Configure poetry to not create a virtual env
RUN poetry config virtualenvs.create false

# Copy project files needed to lock and install dependencies
COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --no-root --only main --no-interaction --no-ansi

# ==========================================================================
# Final Stage
# ==========================================================================
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy the executable binaries (For Streamlit)
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ./src/ ./
COPY ./.streamlit/ ./.streamlit/

# Make sure scripts are on PATH
ENV PATH=/root/.local/bin:$PATH

RUN mkdir -p /app/data /app/db

# Run as non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /app/data /app/db

USER appuser
