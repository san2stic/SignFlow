#!/bin/bash

echo "🚀 Starting SignFlow Application..."
echo ""

# Check if backend .env exists
if [ ! -f backend/.env ]; then
    echo "⚙️  Creating backend .env from example..."
    cp backend/.env.example backend/.env
fi

# Check if frontend .env exists
if [ ! -f frontend/.env ]; then
    echo "⚙️  Creating frontend .env from example..."
    cp frontend/.env.example frontend/.env
fi

# Start backend in background
echo "🔧 Starting backend on http://localhost:8000..."
cd backend
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend on http://localhost:5173..."
cd frontend
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
