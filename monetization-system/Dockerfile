# Build argument for Flask-only deployment
ARG FLASK_ONLY=false

# Stage 1: Builder for dependencies
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first
RUN pip install --upgrade pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Minimal image
FROM python:3.10-slim

# Pass build argument to runtime stage
ARG FLASK_ONLY=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    FLASK_APP=wsgi:app \
    PORT=8080

WORKDIR /app

# Install runtime dependencies - conditional based on FLASK_ONLY
RUN if [ "$FLASK_ONLY" = "true" ]; then \
        # Flask-only: minimal dependencies
        apt-get update && apt-get install -y --no-install-recommends \
            curl \
            libpq5 \
            && rm -rf /var/lib/apt/lists/*; \
    else \
        # Full build: include browser automation dependencies
        apt-get update && apt-get install -y --no-install-recommends \
            curl \
            libpq5 \
            sqlite3 \
            libsqlite3-dev \
            libnss3 \
            libatk1.0-0 \
            libatk-bridge2.0-0 \
            libcups2 \
            libxkbcommon0 \
            libxcomposite1 \
            libxdamage1 \
            libxrandr2 \
            libgbm1 \
            libpango1.0-0 \
            libasound2 \
            wget \
            && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

# Copy application code (as non-root user context)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Use Gunicorn for production
#CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "wsgi:app"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]