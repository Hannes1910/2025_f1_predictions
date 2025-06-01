# Multi-stage Docker build for F1 ML Production Service
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ml_production_service.py .
COPY f1_predictions_test.db .
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p f1_cache models/ensemble

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Start the application
CMD ["python", "ml_production_service.py"]