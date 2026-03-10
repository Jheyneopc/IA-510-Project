FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY model/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Runtime environment variables
ENV ARTIFACTS_DIR=model/artifacts
ENV PORT=8000
ENV MIN_CONFIDENCE=0.55
ENV APP_VERSION=1.3

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]