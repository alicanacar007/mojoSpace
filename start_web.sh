#!/bin/bash

echo "🔥 Starting MojoX Web Interface"
echo "=============================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit
fi

# Create output directory if it doesn't exist
mkdir -p output

echo "🌐 Launching web interface..."
echo "📍 Access the app at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start Streamlit app
streamlit run web_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false 