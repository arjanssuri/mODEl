#!/bin/bash

# mODEl - Network Launch Script
# by Dobrovolny Lab, Texas Christian University
# For running mODEl on a network IP address

echo "🧮 Starting mODEl - Network Mode"
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

# Get IP address options
echo "🌐 Network Configuration Options:"
echo "1. Auto-detect and use your local IP address"
echo "2. Bind to all interfaces (0.0.0.0)"
echo "3. Enter custom IP address"
echo "4. Enter custom IP and port"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        # Auto-detect IP
        if command -v ip &> /dev/null; then
            LOCAL_IP=$(ip route get 1.1.1.1 | awk '{print $7; exit}')
        elif command -v ifconfig &> /dev/null; then
            LOCAL_IP=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1)
        else
            echo "❌ Could not auto-detect IP address. Please choose option 3."
            exit 1
        fi
        
        if [ -z "$LOCAL_IP" ]; then
            echo "❌ Could not auto-detect IP address. Please choose option 3."
            exit 1
        fi
        
        echo "🔍 Detected IP address: $LOCAL_IP"
        SERVER_ARGS="--server.address $LOCAL_IP --server.port 8501"
        ;;
    2)
        # Bind to all interfaces
        echo "🌍 Binding to all interfaces (0.0.0.0)"
        SERVER_ARGS="--server.address 0.0.0.0 --server.port 8501"
        ;;
    3)
        # Custom IP
        read -p "Enter IP address: " CUSTOM_IP
        echo "🎯 Using custom IP: $CUSTOM_IP"
        SERVER_ARGS="--server.address $CUSTOM_IP --server.port 8501"
        ;;
    4)
        # Custom IP and port
        read -p "Enter IP address: " CUSTOM_IP
        read -p "Enter port: " CUSTOM_PORT
        echo "🎯 Using custom IP: $CUSTOM_IP:$CUSTOM_PORT"
        SERVER_ARGS="--server.address $CUSTOM_IP --server.port $CUSTOM_PORT"
        ;;
    *)
        echo "❌ Invalid option. Exiting."
        exit 1
        ;;
esac

# Additional Streamlit configuration for network use
ADDITIONAL_ARGS="--server.headless true --server.enableCORS false --server.enableXsrfProtection false"

echo ""
echo "🚀 Starting mODEl application..."
echo "📍 Server configuration: $SERVER_ARGS"
echo "⚙️  Additional settings: $ADDITIONAL_ARGS"
echo ""

# Change to src directory and run the app
cd src

# Run Streamlit with network configuration
echo "🌐 mODEl will be available at: http://$(echo $SERVER_ARGS | grep -o '[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*\|0\.0\.0\.0'):$(echo $SERVER_ARGS | grep -o '[0-9]*$')"
echo "🔒 Make sure your firewall allows connections on this port"
echo ""

streamlit run app.py $SERVER_ARGS $ADDITIONAL_ARGS 