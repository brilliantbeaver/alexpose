#!/bin/bash
# AlexPose Development Server Stop Script (Unix/Linux/macOS)

echo "🛑 Stopping AlexPose Development Servers..."
echo ""

# Stop backend
if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        echo "✓ Backend server stopped"
    else
        echo "⚠️  Backend server not running"
    fi
    rm -f .backend.pid
else
    # Try to find and kill by port
    BACKEND_PID=$(lsof -ti:8000)
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        echo "✓ Backend server stopped"
    else
        echo "⚠️  No backend server found"
    fi
fi

# Stop frontend
if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        echo "✓ Frontend server stopped"
    else
        echo "⚠️  Frontend server not running"
    fi
    rm -f .frontend.pid
else
    # Try to find and kill by port
    FRONTEND_PID=$(lsof -ti:3000)
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        echo "✓ Frontend server stopped"
    else
        echo "⚠️  No frontend server found"
    fi
fi

echo ""
echo "✅ All servers stopped"
