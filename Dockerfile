# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy just the requirements file first for better caching
COPY requirements.txt ./

# Install dependencies into a virtual environment for isolation
RUN python -m venv /opt/venv
# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    net-tools \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    # Create non-root user
    useradd -m -u 1000 appuser && \
# Create necessary directories with proper permissions
    mkdir -p /app/vectorstore /app/data && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Install our package in development mode
RUN pip install -e .

# Copy Python modules to a secure location
RUN mkdir -p /app/python_modules/data
COPY data/__init__.py data/document.py data/vector_store.py data/vector_loader.py /app/python_modules/data/

# Ensure document_chunks.txt exists for BM25 retrieval
RUN mkdir -p /app/data
COPY --chown=appuser:appuser data/document_chunks.txt /app/data/
COPY --chown=appuser:appuser data/*.pdf /app/data/

# Make scripts executable
RUN chmod +x start.sh cleanup.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV METRICS_PORT=8099

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8501
EXPOSE 8099

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application using our startup script
CMD ["./start.sh"] 