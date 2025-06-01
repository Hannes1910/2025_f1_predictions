#!/bin/bash

# Start Clean F1 ML System - NO MOCK DATA
echo "🚀 Starting F1 ML System - CLEAN VERSION"
echo "✅ NO MOCK DATA - Real F1 data only"

# Check Python version
python3 --version || { echo "❌ Python 3 required"; exit 1; }

# Setup virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Setup environment variables
export WORKER_URL="https://f1-predictions-api.vprifntqe.workers.dev"
export PREDICTIONS_API_KEY="your-api-key-here"

# Create directories
mkdir -p models/production
mkdir -p cache/fastf1
mkdir -p logs

echo ""
echo "🎯 CLEAN SYSTEM STARTUP OPTIONS:"
echo ""
echo "1️⃣  Setup clean database with real F1 data:"
echo "   python3 f1_data_pipeline_clean.py"
echo ""
echo "2️⃣  Start ML service (requires trained models):"
echo "   python3 ml_service_production_clean.py"
echo ""
echo "3️⃣  Test system health:"
echo "   python3 test_clean_system.py"
echo ""
echo "📋 REQUIREMENTS:"
echo "✅ Real F1 data (FastF1)"
echo "✅ Trained ML models in models/production/"
echo "✅ Clean database schema"
echo "❌ NO mock data anywhere"
echo ""

# Offer to run data pipeline
read -p "🔄 Run clean data pipeline now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Running clean data pipeline..."
    python3 f1_data_pipeline_clean.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Data pipeline completed successfully"
        echo ""
        read -p "🚀 Start ML service now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🚀 Starting clean ML service..."
            python3 ml_service_production_clean.py
        fi
    else
        echo "❌ Data pipeline failed"
        exit 1
    fi
fi

echo ""
echo "🎉 Clean system ready!"
echo "📖 See CLEAN_IMPLEMENTATION_PLAN.md for details"