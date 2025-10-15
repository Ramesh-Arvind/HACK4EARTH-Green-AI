#!/bin/bash

# EcoGrow Live Dashboard Launcher
# HACK4EARTH BUIDL Challenge - Real-Time Monitoring

echo "🌱 Starting EcoGrow Live Dashboard..."
echo "================================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly
fi

# Change to dashboard directory
cd "$(dirname "$0")/dashboard"

# Launch dashboard
echo "✅ Launching dashboard on http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "================================================"
echo ""

streamlit run streamlit_app.py --server.port=8501 --server.address=localhost
