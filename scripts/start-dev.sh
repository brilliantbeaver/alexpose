#!/bin/bash
# AlexPose Development Server Startup Script (Unix/Linux/macOS)
# This script starts both the backend FastAPI server and frontend Next.js server

set -e

echo "🚀 Starting AlexPose Development Servers..."
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: UV is not installed. Please install it first:"
    echo "   https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
echo "✓ UV found: $(uv --version)"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed. Please install it first:"
    echo "   https://nodejs.org/"
    exit 1
fi
echo "✓ Node.js found: $(node --version)"

echo ""
echo "📦 Checking dependencies..."

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "⚠️  Frontend dependencies not found. Installing..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "🔧 Starting Backend Server (FastAPI)..."
echo "   URL: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"

# Start backend in background
uv run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "   PID: $BACKEND_PID"

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend server is running!"
else
    echo "⚠️  Backend server may still be starting..."
fi

echo ""
echo "🎨 Starting Frontend Server (Next.js)..."
echo "   URL: http://localhost:3000"

# Start frontend in background
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "   PID: $FRONTEND_PID"

echo ""
echo "✅ Both servers are running!"
echo ""
echo "📍 Access Points:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo "   GAVD Upload: http://localhost:3000/training/gavd"
echo ""
echo "💡 To stop the servers:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "📝 Logs:"
echo "   Backend:  logs/backend.log"
echo "   Frontend: logs/frontend.log"
echo ""

# Save PIDs to file for easy cleanup
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

# Wait for user interrupt
trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f .backend.pid .frontend.pid; echo '✅ Servers stopped'; exit 0" INT TERM

echo "Press Ctrl+C to stop both servers"
wait
