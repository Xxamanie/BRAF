#!/usr/bin/env python3
"""
BRAF Monetization System Setup Script
This script sets up the development environment and installs all dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def setup_virtual_environment():
    """Set up virtual environment"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("🔄 Creating virtual environment...")
        run_command(f"{sys.executable} -m venv venv", "Virtual environment creation")
    else:
        print("✅ Virtual environment already exists")

def install_dependencies():
    """Install Python dependencies"""
    # Determine the correct pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"
    
    # Upgrade pip first
    run_command(f"{pip_path} install --upgrade pip", "Upgrading pip")
    
    # Install dependencies
    run_command(f"{pip_path} install -r requirements.txt", "Installing Python dependencies")
    
    # Install Playwright browsers
    run_command(f"{python_path} -m playwright install", "Installing Playwright browsers")
    run_command(f"{python_path} -m playwright install-deps", "Installing Playwright system dependencies")

def setup_environment_file():
    """Set up environment configuration"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔄 Creating .env file from template...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ .env file created. Please edit it with your actual configuration.")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env.example file found")

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data",
        "certificates",
        "uploads",
        "backups",
        "monitoring/grafana/dashboards",
        "monitoring/grafana/datasources",
        "nginx/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")

def main():
    """Main setup function"""
    print("🚀 Setting up BRAF Monetization System...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Set up virtual environment
    setup_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Set up environment file
    setup_environment_file()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Edit the .env file with your actual configuration")
    print("2. Run: docker-compose up -d --build")
    print("3. Or for development: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
    print("4. Then: python main.py")
    print("\n📊 Service URLs (after deployment):")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("   • Grafana Dashboard: http://localhost:3000")
    print("   • Prometheus Metrics: http://localhost:9090")

if __name__ == "__main__":
    main()