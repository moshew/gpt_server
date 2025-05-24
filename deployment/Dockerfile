FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application from parent directory
COPY . .

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:/app"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Command to run the application using Gunicorn with Uvicorn workers optimized for SSE and large files
CMD ["gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "50", "--limit-request-line", "8190", "--limit-request-field_size", "8190"]
