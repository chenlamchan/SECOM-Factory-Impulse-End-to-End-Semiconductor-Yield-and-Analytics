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

# Install runtime dependencies for psycopg2 (required in final stage too)
RUN apt-get update && apt-get install -y libpq5 && rm -rf /var/lib/apt/lists/*

# Copy only the installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy the executable binaries (For Streamlit)
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ./src/ ./

# Run as non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app 

USER appuser

ENTRYPOINT ["python"]
