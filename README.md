# PDF Chatbot

A chatbot application for querying PDF documents using RAG (Retrieval-Augmented Generation).

## Overview

This application allows users to ask questions about PDF documents and receive accurate answers based on the random machine learning document content. It uses vector embedding search combined with traditional keyword search to retrieve relevant information before generating responses.

## Requirements

- Python 3.9+
- Docker
- Kubernetes (or Minikube for local development)
- OpenAI API key
- Pinecone API key

## Quick Start

1. Set API keys:
   ```bash
   export OPENAI_API_KEY="your_key"
   export PINECONE_API_KEY="your_key"
   export PINECONE_ENVIRONMENT="us-east-1" # location dependent
   ```

2. Deploy using the provided script:
   ```bash
   ./deploy.sh
   ```

3. Access the application:
   - UI: http://localhost:8501
   - Monitoring: http://localhost:3000 (admin/admin)

## Deployment

The project includes a comprehensive deployment script (`deploy.sh`) that automates:

- Docker image building with multi-stage optimization
- Kubernetes resource deployment (pods, services, secrets)
- Vector database initialization
- Monitoring stack setup (Prometheus, Grafana)
- Port forwarding configuration

To deploy with Minikube:

```bash
# Start Minikube
minikube start --memory=4096 --cpus=4

# Use Minikube's Docker daemon
eval $(minikube docker-env)

# Run the deployment script
./deploy.sh
```

## Monitoring

The application includes a fully integrated monitoring solution:

- **Metrics Exposure**: Custom metrics served on port 8099
- **Prometheus**: Configured to scrape application metrics
- **Grafana**: Pre-configured dashboards for visualizing performance data

Access points:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)
- Raw metrics: http://localhost:8099/metrics

## Key Features & Optimizations

1. **Hybrid Retrieval**:
   - Combined vector and BM25 keyword search with configurable weights
   - Optimized chunking for better context retrieval

2. **Response Caching**:
   - LRU cache implementation reduces redundant API calls
   - Cached embeddings for frequently accessed content

3. **Resource Management**:
   - Container resource limits optimized for performance
   - Kubernetes readiness/liveness probes for reliability

## Limitations and Future Work

1. **PDF File Handling**:
   - Currently uses a hardcoded PDF file path in the Kubernetes deployment
   - Future: Implement dynamic file upload and selection capabilities
   - Support for multiple document sources and formats

2. **Planned Improvements**:
   - Multi-user support with session management
   - Enhanced document preprocessing for better chunking
   - Support for larger documents with improved memory management

## Local Development

```bash
# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
```

## Architecture

The application is structured into modular components:
- Data processing and vector storage
- Core RAG implementation
- Streamlit web interface
- Prometheus/Grafana monitoring