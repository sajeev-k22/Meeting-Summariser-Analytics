#!/usr/bin/env python3
"""
Startup script for the Meeting Summarizer & Analytics App
"""

import os
import sys
from app import app

def main():
    """Main function to start the Flask application"""
    print("🎤 Starting Meeting Summarizer & Analytics App...")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found!")
        print("   Please copy env_example.txt to .env and configure your API keys")
        print("   The app will run but some features may not work without proper configuration")
        print()
    
    # Check for required directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    print("✅ Required directories created/verified")
    
    print("🚀 Starting Flask development server...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
