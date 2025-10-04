#!/usr/bin/env python3
"""
Setup script for Quantization-Aware Training project
Automates the initial setup and provides quick start options
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    directories = ["data", "logs", "models", "plots"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running test suite...")
    try:
        result = subprocess.run([sys.executable, "test_qat.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_demo():
    """Run the demo script"""
    print("\n🚀 Running demo...")
    try:
        subprocess.run([sys.executable, "demo.py"])
        print("✅ Demo completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def run_original():
    """Run the original implementation"""
    print("\n🧠 Running original implementation...")
    try:
        subprocess.run([sys.executable, "0155.py"])
        print("✅ Original implementation completed!")
        return True
    except Exception as e:
        print(f"❌ Error running original: {e}")
        return False

def launch_ui():
    """Launch the Streamlit UI"""
    print("\n🌐 Launching Streamlit UI...")
    print("The web interface will open in your browser.")
    print("Press Ctrl+C to stop the server.")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        return True
    except KeyboardInterrupt:
        print("\n✅ UI stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='QAT Project Setup')
    parser.add_argument('--install-only', action='store_true',
                       help='Only install requirements, skip other steps')
    parser.add_argument('--test', action='store_true',
                       help='Run tests after setup')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo after setup')
    parser.add_argument('--original', action='store_true',
                       help='Run original implementation after setup')
    parser.add_argument('--ui', action='store_true',
                       help='Launch web UI after setup')
    
    args = parser.parse_args()
    
    print("🧠 Quantization-Aware Training Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    if args.install_only:
        print("\n✅ Setup completed (install only)")
        return
    
    # Create directories
    create_directories()
    
    # Run requested actions
    if args.test:
        run_tests()
    
    if args.demo:
        run_demo()
    
    if args.original:
        run_original()
    
    if args.ui:
        launch_ui()
    
    # If no specific action requested, show options
    if not any([args.test, args.demo, args.original, args.ui]):
        print("\n🎯 Setup completed! Available options:")
        print("  python setup.py --test     # Run test suite")
        print("  python setup.py --demo     # Run demo")
        print("  python setup.py --original # Run original implementation")
        print("  python setup.py --ui      # Launch web UI")
        print("  streamlit run app.py       # Launch UI directly")
        print("  python qat_trainer.py     # Run advanced trainer")
        print("\n📚 See README.md for detailed usage instructions")

if __name__ == "__main__":
    main()
