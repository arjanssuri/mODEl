# 🌐 mODEl Network Deployment Guide

This guide helps you run mODEl on a network so it can be accessed from other computers.

## 🚀 Quick Network Setup

### Option 1: Interactive Network Script (Recommended)
```bash
# Make script executable
chmod +x run_model_network.sh

# Run with guided setup
./run_model_network.sh
```

The script will guide you through:
1. **Auto-detect IP**: Automatically finds your computer's IP address
2. **Bind to all interfaces**: Makes mODEl accessible from any IP (0.0.0.0)
3. **Custom IP**: Specify a particular IP address
4. **Custom IP + Port**: Full control over network configuration

### Option 2: Manual Configuration
```bash
# Run on specific IP address
./run_model.sh --server.address 192.168.1.100 --server.port 8501

# Run on all interfaces (accessible from any IP)
./run_model.sh --server.address 0.0.0.0 --server.port 8501 --server.headless true

# Full network configuration
./run_model.sh --server.address 0.0.0.0 --server.port 8080 --server.headless true --server.enableCORS false
```

## 🔧 Common Network Configurations

### Home Network Access
```bash
# For access within your home/office network
./run_model_network.sh
# Choose option 1 (auto-detect) or 2 (all interfaces)
```

### Server Deployment
```bash
# For server deployment with specific port
./run_model.sh --server.address 0.0.0.0 --server.port 8080 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```

### Development/Testing
```bash
# For development with specific IP
./run_model.sh --server.address 192.168.1.100 --server.port 8501
```

## 🐛 Troubleshooting Network Issues

### "FileNotFoundError: app.py" Error

**Problem**: This error occurs when Streamlit's file watcher can't find the app.py file during network deployment.

**Root Cause**: Streamlit's file watching system gets confused about the working directory when binding to IP addresses.

**Solutions**:

1. **Use the network script** (automatically handles this):
   ```bash
   ./run_model_network.sh
   ```

2. **Ensure correct working directory**:
   ```bash
   # Always run from mODEl root directory
   pwd  # Should show .../mODEl
   ./run_model.sh --server.address YOUR_IP
   ```

3. **Manual fix**:
   ```bash
   cd mODEl  # Ensure you're in the root directory
   cd src    # Move to src directory
   streamlit run app.py --server.address YOUR_IP --server.port 8501
   ```

### Firewall Issues

**Problem**: Can't access mODEl from other computers

**Solutions**:
- **Windows**: Allow Python/Streamlit through Windows Firewall
- **macOS**: System Preferences → Security & Privacy → Firewall → Allow incoming connections
- **Linux**: Configure iptables or ufw to allow the port

```bash
# Example: Allow port 8501 on Ubuntu/Debian
sudo ufw allow 8501

# Example: Allow port 8501 on CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

### Port Already in Use

**Problem**: "Address already in use" error

**Solutions**:
```bash
# Find what's using the port
lsof -i :8501

# Kill the process (replace PID with actual process ID)
kill -9 PID

# Or use a different port
./run_model.sh --server.address 0.0.0.0 --server.port 8502
```

## 🔒 Security Considerations

### For Production Use
- Use a reverse proxy (nginx, Apache) for HTTPS
- Configure proper firewall rules
- Consider authentication if needed
- Monitor access logs

### For Development/Testing
- Use within trusted networks only
- Be aware that `0.0.0.0` makes the app accessible to anyone on the network
- Consider using specific IP addresses instead of `0.0.0.0`

## 📱 Accessing mODEl

Once running, mODEl will be accessible at:
- **Local access**: `http://localhost:8501`
- **Network access**: `http://YOUR_IP:8501`
- **All interfaces**: `http://ANY_COMPUTER_IP:8501` (when using 0.0.0.0)

### Finding Your IP Address

**Windows**:
```cmd
ipconfig | findstr IPv4
```

**macOS/Linux**:
```bash
# Method 1
ifconfig | grep "inet " | grep -v 127.0.0.1

# Method 2
ip route get 1.1.1.1 | awk '{print $7; exit}'

# Method 3
hostname -I
```

## 🔧 Advanced Configuration

### Custom Streamlit Config
Create `.streamlit/config.toml` in the mODEl directory:
```toml
[server]
address = "0.0.0.0"
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#0066CC"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Environment Variables
```bash
# Set environment variables for network deployment
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_PORT="8501"
export STREAMLIT_SERVER_HEADLESS="true"

# Then run normally
./run_model.sh
```

## 📋 Network Deployment Checklist

- [ ] Choose appropriate IP binding (localhost, specific IP, or 0.0.0.0)
- [ ] Configure firewall to allow the chosen port
- [ ] Test local access first (`http://localhost:8501`)
- [ ] Test network access from another computer
- [ ] Verify all mODEl features work over the network
- [ ] Document the access URL for users
- [ ] Consider security implications for your deployment environment

---

**Need help?** Check the main README.md troubleshooting section or open an issue in the repository. 