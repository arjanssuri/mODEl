#!/bin/bash

# Install dependencies if not already installed
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting ODE Model Fitting Tool..."
streamlit run app.py 