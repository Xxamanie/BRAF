FROM python:3.11-slim

WORKDIR /app

# Install academic dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy academic research code
COPY . .

# Academic environment variables
ENV ACADEMIC_ENVIRONMENT=research
ENV PYTHONPATH=/app

# Academic health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/academic/health')"

# Academic entry point
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "academic_main:app"]
