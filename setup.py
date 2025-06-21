#!/usr/bin/env python3
"""
MLOps Anomaly Detection Pipeline Setup
======================================

Quick setup script for the anomaly detection pipeline.
"""

import subprocess
import sys
import os
import time
import requests

def run_command(command, description):
    """Run a shell command and print status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def check_service(url, name, timeout=5):
    """Check if a service is responding."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name} is running")
            return True
        else:
            print(f"❌ {name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name} is not responding: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 MLOps Anomaly Detection Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 11:
        print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    else:
        print(f"❌ Python 3.11+ required. Current: {python_version.major}.{python_version.minor}")
        return False
    
    # Check if virtual environment exists
    if not os.path.exists("venv"):
        print("📦 Creating virtual environment...")
        if not run_command("python3 -m venv venv", "Create virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Install dependencies
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    if not run_command(f"{activate_cmd} && pip install -r requirements.txt", "Install dependencies"):
        return False
    
    # Check Docker
    if not run_command("docker --version", "Check Docker installation"):
        print("❌ Docker is required. Please install Docker and try again.")
        return False
    
    # Start Docker services
    print("🐳 Starting Docker services...")
    if not run_command("docker-compose up -d", "Start Docker services"):
        return False
    
    # Wait for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Check services
    services = [
        ("http://localhost:9090/-/healthy", "Prometheus"),
        ("http://localhost:3001/api/health", "Grafana"),
        ("http://localhost:8080", "Kafka UI")
    ]
    
    all_services_ok = True
    for url, name in services:
        if not check_service(url, name):
            all_services_ok = False
    
    if not all_services_ok:
        print("⚠️ Some services are not responding. They may need more time to start.")
        print("   Try running 'python test_pipeline.py' in a few minutes.")
    
    # Setup Grafana dashboard
    print("📊 Setting up Grafana dashboard...")
    if not run_command(f"{activate_cmd} && python setup_grafana_dashboard.py", "Setup Grafana dashboard"):
        print("⚠️ Grafana dashboard setup failed. You can try again manually later.")
    
    print("\n🎉 Setup completed!")
    print("\n🚀 Next steps:")
    print("1. Start the anomaly detection pipeline:")
    print(f"   {activate_cmd} && python pipeline/real_metrics_collector.py")
    print("\n2. Open Grafana dashboard:")
    print("   http://localhost:3001 (admin/admin123)")
    print("\n3. Generate test load:")
    print(f"   {activate_cmd} && python auto_generate_load.py")
    print("\n4. Run health check:")
    print(f"   {activate_cmd} && python test_pipeline.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)