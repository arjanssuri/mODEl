#!/bin/bash

# Install dependencies if not already installed
echo "Installing dependencies..."
pip install -r ../requirements.txt

# Run the Streamlit app from src directory
echo "Starting mODEl - Advanced ODE Model Fitting Tool..."

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