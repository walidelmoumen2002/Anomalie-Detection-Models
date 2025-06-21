#!/usr/bin/env python3
"""
Test MLOps Pipeline
==================

Quick test to verify all components are working correctly.
"""

import requests
import time
import json

def test_prometheus():
    """Test Prometheus metrics endpoint."""
    print("🔍 Testing Prometheus metrics...")
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            if "system_cpu_usage_percent" in metrics:
                print("  ✅ Anomaly detection metrics are being exposed")
                return True
            else:
                print("  ❌ Expected metrics not found")
                return False
        else:
            print(f"  ❌ HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False

def test_prometheus_server():
    """Test Prometheus server."""
    print("🔍 Testing Prometheus server...")
    try:
        response = requests.get("http://localhost:9090/-/healthy", timeout=5)
        if response.status_code == 200:
            print("  ✅ Prometheus server is healthy")
            return True
        else:
            print(f"  ❌ HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False

def test_grafana():
    """Test Grafana."""
    print("🔍 Testing Grafana...")
    try:
        response = requests.get("http://localhost:3001/api/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ Grafana is healthy")
            return True
        else:
            print(f"  ❌ HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False

def test_kafka():
    """Test Kafka UI."""
    print("🔍 Testing Kafka UI...")
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            print("  ✅ Kafka UI is accessible")
            return True
        else:
            print(f"  ❌ HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False

def get_current_metrics():
    """Get current system metrics from our service."""
    print("📊 Getting current system metrics...")
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            lines = metrics.split('\n')
            
            cpu_usage = None
            memory_usage = None
            anomaly_score = None
            is_anomaly = None
            
            for line in lines:
                if line.startswith('system_cpu_usage_percent'):
                    cpu_usage = float(line.split()[-1])
                elif line.startswith('system_memory_usage_percent'):
                    memory_usage = float(line.split()[-1])
                elif line.startswith('anomaly_detection_score'):
                    anomaly_score = float(line.split()[-1])
                elif line.startswith('anomaly_detection_is_anomaly'):
                    is_anomaly = float(line.split()[-1])
            
            print(f"  💻 CPU Usage: {cpu_usage:.1f}%")
            print(f"  🧠 Memory Usage: {memory_usage:.1f}%")
            print(f"  🎯 Anomaly Score: {anomaly_score:.3f}")
            print(f"  🚨 Is Anomaly: {'YES' if is_anomaly == 1.0 else 'NO'}")
            
            return True
        else:
            print(f"  ❌ HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Failed to get metrics: {e}")
        return False

def main():
    print("🧪 MLOps Pipeline Health Check")
    print("="*40)
    
    tests = [
        test_prometheus,
        test_prometheus_server,
        test_grafana,
        test_kafka
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)
    
    print(f"\n📈 Current System Status:")
    get_current_metrics()
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All systems operational!")
        print("\n🚀 Next steps:")
        print("1. Open Grafana: http://localhost:3001 (admin/admin123)")
        print("2. Import the anomaly detection dashboard")
        print("3. Monitor real-time metrics and anomaly detection")
        print("4. Generate load to test anomaly detection")
    else:
        print("\n⚠️ Some systems are not responding correctly")
        print("Check the setup and try again")
    
    return 0 if passed == len(tests) else 1

if __name__ == "__main__":
    exit(main())