#!/usr/bin/env python
"""
DeClickify - Backend API Server Launcher
Starts the Flask API server with proper error handling
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = ['flask', 'tensorflow', 'numpy', 'pandas']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies found")
    return True

def start_api_server():
    """Start the Flask API server"""
    print("\n" + "="*50)
    print("  DeClickify - API Server")
    print("="*50 + "\n")
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting API Server...")
    print("📍 Backend URL: http://localhost:5000")
    print("📊 API Docs: http://localhost:5000/api")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        from api_server import app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
    except Exception as e:
        print(f"\n❌ Error starting API server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Ensure we're in the right directory
    if not Path('api_server.py').exists():
        print("❌ api_server.py not found. Please run from project root.")
        sys.exit(1)
    
    start_api_server()
