#!/bin/bash
# Start the AlexPose server with clean state (kills any existing instances)

set -e

PORT=${1:-8000}
HOST=${2:-127.0.0.1}

echo "🚀 AlexPose Server Clean Startup"
echo "============================================================"

# Step 1: Check for existing processes on port
echo ""
echo "📡 Checking port $PORT..."

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
    echo "⚠️  Found existing process on port $PORT (PID: $PID)"
    
    # Kill the process
    echo "🔪 Killing stale process..."
    kill -9 $PID 2>/dev/null || true
    sleep 1  # Give OS time to release port
    
    echo "✅ Stale process killed successfully"
else
    echo "✅ Port $PORT is available"
fi

# Step 2: Verify configuration
echo ""
echo "📋 Verifying configuration..."

if python -c "from ambient.core.config import ConfigurationManager; cm = ConfigurationManager('config', 'development'); result = cm.validate_configuration(); exit(0 if result else 1)" 2>/dev/null; then
    echo "✅ Configuration is valid"
else
    echo "⚠️  Configuration has warnings (continuing anyway)"
fi

# Step 3: Start the server
echo ""
echo "🚀 Starting server..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Reload: Enabled"
echo ""

uvicorn server.main:app --reload --host $HOST --port $PORT
