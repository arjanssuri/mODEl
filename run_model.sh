#!/bin/bash

# mODEl - Advanced ODE Model Fitting Tool
# by Dobrovolny Lab, Texas Christian University

echo "🧮 Starting mODEl - Advanced ODE Model Fitting Tool"
echo "   by Dobrovolny Lab, Texas Christian University"
echo ""

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the mODEl root directory"
    exit 1
fi

# Install dependencies if not already installed
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Change to src directory and run the app
echo "🚀 Starting mODEl application..."
cd src

# Check for command line arguments for server configuration
if [ $# -eq 0 ]; then
    # Default: run on localhost
    echo "🌐 Running on localhost (default)"
    streamlit run app.py
else
    # Pass all arguments to streamlit (for IP binding, port configuration, etc.)
    echo "🌐 Running with custom configuration: $*"
    streamlit run app.py "$@"
fi 