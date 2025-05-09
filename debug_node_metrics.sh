#!/bin/bash

# Make the script executable
chmod +x debug_node_metrics.sh

# Save the original deploy.sh
cp deploy.sh deploy.sh.original

# Find what's causing the Node Metrics port forwarding
echo "Modifying deploy.sh to debug..."
sed -i.bak 's/start_port_forward pdf-chatbot-service 8501 80 "Streamlit UI"/echo "DEBUG: Before Streamlit UI"; start_port_forward pdf-chatbot-service 8501 80 "Streamlit UI"; echo "DEBUG: After Streamlit UI"/' deploy.sh
sed -i.bak 's/start_port_forward prometheus 9090 9090 "Prometheus"/echo "DEBUG: Before Prometheus"; start_port_forward prometheus 9090 9090 "Prometheus"; echo "DEBUG: After Prometheus"/' deploy.sh
sed -i.bak 's/start_port_forward pdf-chatbot-service 8099 8099 "App Metrics"/echo "DEBUG: Before App Metrics"; start_port_forward pdf-chatbot-service 8099 8099 "App Metrics"; echo "DEBUG: After App Metrics"/' deploy.sh
sed -i.bak 's/start_port_forward grafana 3000 3000 "Grafana"/echo "DEBUG: Before Grafana"; start_port_forward grafana 3000 3000 "Grafana"; echo "DEBUG: After Grafana"/' deploy.sh

# Add debug for the function itself
sed -i.bak 's/start_port_forward() {/start_port_forward() {\n    echo "DEBUG: start_port_forward called with $1 $2 $3 $4"/' deploy.sh

echo "Running deploy.sh with debug flags..."
./deploy.sh > deploy_debug.log 2>&1 &

echo "Waiting for deployment to complete..."
sleep 5

echo "Checking debug output..."
grep -A2 -B2 "Node Metrics" deploy_debug.log

# Restore the original deploy.sh
echo "Restoring original deploy.sh..."
mv deploy.sh.original deploy.sh 