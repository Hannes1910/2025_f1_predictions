#!/bin/bash

# Start ML Production Service with D1 Connection
echo "🚀 Starting F1 ML Production Service with D1 Connection..."

# Export environment variables
export WORKER_URL="https://f1-predictions-api.vprifntqe.workers.dev"
export PREDICTIONS_API_KEY="test-key-123"  # Replace with your actual API key

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models/ensemble

# Run the ML service
echo "Starting ML service on port 8001..."
python ml_production_service_d1.py