#!/usr/bin/env python3
"""
Simple test script to verify the Meeting Summarizer app setup
"""

import os
import sys

import pytest


def check_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")

    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import whisper
        print("✅ Whisper imported successfully")
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        print("   Run: pip install openai-whisper")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        print("   Run: pip install google-generativeai")
        return False
    
    try:
        from reportlab.lib.pagesizes import letter
        print("✅ ReportLab imported successfully")
    except ImportError as e:
        print(f"❌ ReportLab import failed: {e}")
        print("   Run: pip install reportlab")
        return False
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        print("   Run: pip install matplotlib")
        return False
    
    try:
        from wordcloud import WordCloud
        print("✅ WordCloud imported successfully")
    except ImportError as e:
        print(f"❌ WordCloud import failed: {e}")
        print("   Run: pip install wordcloud")
        return False
    
    return True


def test_imports():
    assert check_imports()

def test_directories():
    """Test if required directories exist"""
    print("\n📁 Testing directory structure...")
    
    directories = ['uploads', 'temp', 'templates']
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/ directory exists")
        else:
            print(f"❌ {directory}/ directory missing")
            os.makedirs(directory, exist_ok=True)
            print(f"   Created {directory}/ directory")

def test_environment():
    """Test environment configuration"""
    print("\n🔧 Testing environment configuration...")
    
    if os.path.exists('.env'):
        print("✅ .env file found")
        
        # Load and check for API key
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.environ.get('GOOGLE_API_KEY')
            if api_key and api_key != 'your_google_api_key_here':
                print("✅ Google API key configured")
            else:
                print("⚠️  Google API key not configured or using placeholder")
                print("   Update .env file with your actual Google API key")
        except ImportError:
            print("⚠️  python-dotenv not installed")
    else:
        print("❌ .env file not found")
        print("   Copy env_example.txt to .env and configure your API keys")

def check_app_import():
    """Test if the main app can be imported"""
    print("\n🚀 Testing app import...")

    try:
        from app import app, analyzer
        print("✅ Main app imported successfully")
        print("✅ MeetingAnalyzer class imported successfully")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False


def test_app_import():
    assert check_app_import()


def test_calculate_meeting_metrics_handles_incomplete_segments():
    from app import analyzer

    transcript = "Progress was made during the productive planning session"
    segments = [
        {"start": 2.0, "end": 5.0, "speaker": "Alice"},
        {"start": 1.0, "end": 3.0},
        {"start": None, "end": 10.0, "speaker": "Bob"},
        {"start": 7.0, "end": 9.5, "speaker": "Alice"},
        "invalid",
    ]

    metrics = analyzer.calculate_meeting_metrics(transcript, segments)

    assert metrics["duration"] == pytest.approx(9.5)
    assert metrics["speaker_stats"]["Alice"] == pytest.approx(5.5)
    assert metrics["speaker_stats"]["Unknown"] == pytest.approx(2.0)

def main():
    """Main test function"""
    print("🎤 Meeting Summarizer & Analytics App - Setup Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= check_imports()
    test_directories()
    test_environment()
    all_tests_passed &= check_app_import()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! The app should be ready to run.")
        print("   Run: python run.py")
    else:
        print("❌ Some tests failed. Please fix the issues above before running the app.")
        sys.exit(1)

if __name__ == '__main__':
    main()

