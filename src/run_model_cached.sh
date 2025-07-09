#!/bin/bash

# mODEl Cached Run Script
# Optimized for data persistence and caching
# by Dobrovolny Lab, Texas Christian University

echo "🧮 Starting mODEl with enhanced caching..."

# Set environment variables for optimal caching
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_THEME_PRIMARY_COLOR="#0066CC"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create cache directory if it doesn't exist
mkdir -p .streamlit/cache

echo "💾 Cache directory: .streamlit/cache"
echo "🔄 Session persistence: ENABLED"
echo "⚡ Performance optimization: ENABLED"
echo ""

# Run Streamlit with optimized settings
streamlit run app_clean.py \
    --server.port 8502 \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#0066CC" \
    --client.showErrorDetails true

echo ""
echo "mODEl session ended. Cache preserved for next session." 