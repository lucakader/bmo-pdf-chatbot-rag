#!/bin/bash
# Script to check metrics endpoints in the cluster

echo "Checking Prometheus connectivity..."
kubectl port-forward service/prometheus 9090:9090 &
PROM_PID=$!
sleep 2
curl -s http://localhost:9090/api/v1/targets | grep -o '"health":".*"' | sort | uniq -c
kill $PROM_PID 2>/dev/null

echo ""
echo "Checking PDF chatbot metrics endpoint..."
# Get pod IPs
PDF_CHATBOT_PODS=$(kubectl get pods -l app=pdf-chatbot -o jsonpath='{.items[*].metadata.name}')
for POD in $PDF_CHATBOT_PODS; do
  echo "Checking metrics on pod $POD..."
  kubectl port-forward pod/$POD 8099:8099 &
  PORT_FWD_PID=$!
  sleep 2
  echo "Curl output from metrics endpoint:"
  curl -s http://localhost:8099/metrics | head -n 10
  echo "..."
  kill $PORT_FWD_PID 2>/dev/null
  echo ""
done

echo "Checking service metrics endpoint..."
kubectl port-forward service/pdf-chatbot-service 8099:8099 &
SVC_PID=$!
sleep 2
echo "Curl output from service metrics endpoint:"
curl -s http://localhost:8099/metrics | head -n 10
echo "..."
kill $SVC_PID 2>/dev/null

echo ""
echo "Done checking metrics endpoints." 