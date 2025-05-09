#!/bin/bash
set -e

echo "Shutting down PDF Chatbot application..."

# Kill any port forwarding processes
echo "Stopping port forwarding..."
if command -v killall &> /dev/null; then
  killall kubectl || true
elif command -v lsof &> /dev/null; then
  for PORT in 8501 9090 8099 3000; do
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
  done
fi

# Either scale down or delete deployments
echo "Scaling down deployments to 0 replicas..."
kubectl scale deployment pdf-chatbot --replicas=0
kubectl scale deployment prometheus --replicas=0
kubectl scale deployment grafana --replicas=0

# Wait for pods to terminate
echo "Waiting for pods to terminate..."
sleep 5

# Show status
kubectl get pods

echo "Shutdown complete. Your Kubernetes resources are still defined but scaled to 0."
echo ""
echo "To completely remove all resources, run:"
echo "kubectl delete deployment pdf-chatbot prometheus grafana"
echo "kubectl delete service pdf-chatbot-service prometheus grafana"
echo "kubectl delete job pdf-chatbot-init"
echo "kubectl delete configmap chatbot-metrics prometheus-config grafana-datasources grafana-dashboards grafana-dashboard-provider"
echo ""
echo "To restart the application later, run: ./deploy.sh" 