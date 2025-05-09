#!/bin/bash
set -e

# Set current directory
cd "$(dirname "$0")"

# Helper function for error handling
handle_error() {
  echo "Error: $1"
  exit 1
}

# Check required environment variables
echo "Checking required environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
  handle_error "OPENAI_API_KEY environment variable is not set. Please set it with: export OPENAI_API_KEY=your_api_key"
fi

if [ -z "$PINECONE_API_KEY" ]; then
  handle_error "PINECONE_API_KEY environment variable is not set. Please set it with: export PINECONE_API_KEY=your_api_key"
fi

if [ -z "$PINECONE_ENVIRONMENT" ]; then
  echo "PINECONE_ENVIRONMENT not set, defaulting to us-east-1..."
  export PINECONE_ENVIRONMENT="us-east-1"
fi

# Check if PDF file exists
if [ ! -f "random machine learing pdf.pdf" ]; then
  handle_error "PDF file not found! Please make sure 'random machine learing pdf.pdf' exists in the current directory."
fi

echo "Building the Docker image for the PDF Chatbot..."
docker build -t pdf-chatbot . || handle_error "Docker build failed"

# Set up Kubernetes environment
echo "Setting up Kubernetes environment..."

# Ensure the default namespace exists
kubectl get namespace default || kubectl create namespace default

# Create secrets from environment variables
echo "Setting up API key secrets..."
kubectl create secret generic openai-secret \
  --from-literal=api-key="$OPENAI_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f - || handle_error "Failed to create OpenAI secret"

kubectl create secret generic pinecone-secret \
  --from-literal=api-key="$PINECONE_API_KEY" \
  --from-literal=environment="$PINECONE_ENVIRONMENT" \
  --dry-run=client -o yaml | kubectl apply -f - || handle_error "Failed to create Pinecone secret"

# Create ConfigMap for PDF files
echo "Skipping ConfigMap creation for PDF files..."

# Apply ConfigMaps with force to prevent conflicts
echo "Applying ConfigMaps..."

# Apply Prometheus config
echo "Applying Prometheus configuration..."
kubectl apply -f k8s/prometheus-config.yaml --force || echo "No prometheus-config.yaml found or application failed, skipping"

# Ensure Prometheus deployment has at least 1 replica
echo "Ensuring Prometheus deployment has correct replicas..."
kubectl scale deployment prometheus --replicas=1 2>/dev/null || echo "Prometheus deployment scaling failed, may need to be created first"

# Create service
echo "Creating service..."
kubectl apply -f k8s/service.yaml --force || echo "Warning: Service application had issues, continuing anyway..."

# Run the initialization job
echo "Running initialization job to load vector store..."
kubectl delete job pdf-chatbot-init 2>/dev/null || true  # Delete any previous job
kubectl apply -f k8s/init-job.yaml || handle_error "Failed to create initialization job"

# Wait for initialization job to complete
echo "Waiting for initialization job to complete..."
kubectl wait --for=condition=complete job/pdf-chatbot-init --timeout=30m || echo "Initialization job did not complete in time, continuing anyway..."

# Create deployment
echo "Creating deployment..."
kubectl apply -f k8s/deployment.yaml || handle_error "Failed to create deployment"

# Deploy Prometheus and Grafana
echo "Deploying monitoring stack..."
kubectl apply -f k8s/grafana-datasources.yaml --force || echo "No grafana-datasources.yaml found, skipping"
kubectl apply -f k8s/grafana-dashboard-provider.yaml --force || echo "No grafana-dashboard-provider.yaml found, skipping"
kubectl apply -f k8s/grafana-dashboard.yaml --force || echo "No grafana-dashboard.yaml found, skipping" 
kubectl apply -f k8s/grafana-secret.yaml --force || echo "No grafana-secret.yaml found, skipping"
kubectl apply -f k8s/grafana.yaml --force || echo "No grafana.yaml found, skipping"

# Wait for deployments to become ready
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment pdf-chatbot --timeout=5m || echo "Deployment may not be fully ready yet"
kubectl rollout status deployment prometheus --timeout=2m || echo "Prometheus deployment not found or not ready, skipping"
kubectl rollout status deployment grafana --timeout=2m || echo "Grafana deployment not found or not ready, skipping"

# Kill any existing port forwarding processes
echo "Cleaning up existing port-forwarding processes..."
# More aggressively cleanup port-forwarding processes
./cleanup.sh || true

# Wait to ensure ports are fully released
echo "Waiting for ports to be released..."
sleep 3

echo "Setting up port forwarding..."
# Function to safely start port-forwarding
start_port_forward() {
    local service=$1
    local local_port=$2
    local remote_port=$3
    local description=$4
    
    # Check if port is already in use
    if lsof -i:$local_port -sTCP:LISTEN &>/dev/null; then
        echo "Warning: Port $local_port is already in use. Cannot forward $description."
    else
        kubectl port-forward --address 0.0.0.0 service/$service $local_port:$remote_port &>/dev/null &
        local result=$?
        if [ $result -eq 0 ]; then
            echo "- Started port forwarding for $description on port $local_port"
            sleep 1
            return 0
        else
            echo "- Failed to start port forwarding for $description on port $local_port"
            return 1
        fi
    fi
    return 1
}

# Forward the Streamlit UI
start_port_forward pdf-chatbot-service 8501 8501 "Streamlit UI"

# Forward Prometheus
start_port_forward prometheus 9090 9090 "Prometheus" 

# Forward metrics endpoint for app metrics
start_port_forward pdf-chatbot-service 8099 8099 "App Metrics"

# Forward Grafana
start_port_forward grafana 3000 3000 "Grafana"

echo ""
echo "Deployment complete! Access the services at:"
echo "- Streamlit UI: http://localhost:8501"
echo "- Prometheus: http://localhost:9090"
echo "- App Metrics: http://localhost:8099/metrics"
echo "- Grafana: http://localhost:3000 (default login: admin/admin)"
echo ""
echo "To stop port forwarding, run: ./cleanup.sh"
echo ""

# Trap to catch Ctrl+C and clean up port forwarding
trap './cleanup.sh' INT TERM

# Keep the script running to maintain port forwarding
wait 