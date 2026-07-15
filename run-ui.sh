#!/bin/bash

# Healthcare Agent - Streamlit UI Runner

echo "🏥 Starting Healthcare Agent Chat UI..."
echo ""
echo "Make sure services are running (./start-services.sh)"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run streamlit app from src directory
streamlit run src/streamlit_app.py --server.port 8501 --server.address localhost
