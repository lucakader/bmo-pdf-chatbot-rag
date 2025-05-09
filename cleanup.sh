#!/bin/bash

# Function to check and kill processes on a port
cleanup_port() {
    local port=$1
    echo "Checking port $port..."
    
    # Try lsof first (macOS and many Linux)
    if command -v lsof &> /dev/null; then
        PIDs=$(lsof -ti:$port 2>/dev/null)
        if [ -n "$PIDs" ]; then
            echo "Killing processes on port $port: $PIDs"
            kill -15 $PIDs 2>/dev/null || kill -9 $PIDs 2>/dev/null || true
            return 0
        fi
    fi
    
    # Fallback to netstat (other systems)
    if command -v netstat &> /dev/null; then
        # Different formats for different OS versions
        PIDs=$(netstat -tulnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1)
        if [ -n "$PIDs" ]; then
            for PID in $PIDs; do
                if [ -n "$PID" ] && [ "$PID" != "" ]; then
                    echo "Killing process $PID on port $port"
                    kill -15 $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
                fi
            done
            return 0
        fi
    fi
    
    echo "No processes found on port $port"
    return 1
}

echo "Killing port-forwarding processes..."

# Kill all kubectl port-forward processes belonging to current user
if command -v pgrep &> /dev/null; then
    PIDS=$(pgrep -f "kubectl port-forward" || true)
    if [ -n "$PIDS" ]; then
        echo "Killing kubectl port-forward processes: $PIDS"
        kill -15 $PIDS 2>/dev/null || kill -9 $PIDS 2>/dev/null || true
    fi
elif command -v ps &> /dev/null && command -v grep &> /dev/null; then
    # Alternative approach using ps and grep
    ps aux | grep "[k]ubectl port-forward" | awk '{print $2}' | xargs kill -15 2>/dev/null || xargs kill -9 2>/dev/null || true
fi

# Clean up each port individually
cleanup_port 8501
cleanup_port 9090
cleanup_port 8099
cleanup_port 3000

echo "Waiting for processes to terminate..."
sleep 2

# Check if any port-forward processes remain
echo "Checking for remaining port-forward processes..."
if command -v ps &> /dev/null && command -v grep &> /dev/null; then
    REMAINING=$(ps aux | grep "[k]ubectl port-forward" | wc -l)
    if [ "$REMAINING" -gt 0 ]; then
        echo "Warning: $REMAINING port-forward processes still running"
        ps aux | grep "[k]ubectl port-forward"
    else
        echo "No port-forward processes found."
    fi
fi

echo "Cleanup complete!" 