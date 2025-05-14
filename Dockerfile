# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements
COPY requirements.txt ./

# Install dependencies into a virtual environment for isolation
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    # Create non-root user
    useradd -m -u 1000 appuser && \
    # Create necessary directories with proper permissions
    mkdir -p /app/vectorstore /app/data && \
    chown -R appuser:appuser /app

# Copy only necessary application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser core/ ./core/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser monitoring/ ./monitoring/
COPY --chown=appuser:appuser streamlit_app.py ./
COPY --chown=appuser:appuser setup.py ./

# Install package in development mode
RUN pip install -e .

# Create data directory and copy required files
RUN mkdir -p /app/data
COPY --chown=appuser:appuser data/document_chunks.txt /app/data/
COPY --chown=appuser:appuser data/*.pdf /app/data/

# Switch to non-root user
USER appuser

# Expose ports (documentation only)
EXPOSE 8501 8099

# Run Streamlit
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py"] 