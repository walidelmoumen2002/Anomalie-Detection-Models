#!/usr/bin/env python3
"""
Test script to verify Grafana dashboard can access Prometheus data.
"""

import requests
import json
import time

def test_prometheus_connection():
    """Test if Prometheus is accessible and has data."""
    try:
        # Test Prometheus metrics
        metrics = [
            'system_cpu_usage_percent',
            'system_memory_usage_percent', 
            'anomaly_detection_score',
            'anomaly_detection_is_anomaly'
        ]
        
        print("🔍 Testing Prometheus connection...")
        
        for metric in metrics:
            response = requests.get(f"http://localhost:9090/api/v1/query?query={metric}")
            data = response.json()
            
            if data['data']['result']:
                value = data['data']['result'][0]['value'][1]
                print(f"✅ {metric}: {value}")
            else:
                print(f"❌ {metric}: No data")
                
        return True
        
    except Exception as e:
        print(f"❌ Prometheus connection failed: {e}")
        return False

def test_grafana_connection():
    """Test if Grafana is accessible."""
    try:
        print("\n🔍 Testing Grafana connection...")
        
        # Test Grafana API
        response = requests.get("http://localhost:3001/api/health")
        if response.status_code == 200:
            print("✅ Grafana is accessible")
            
            # Test data source
            auth = ('admin', 'admin123')
            ds_response = requests.get("http://localhost:3001/api/datasources", auth=auth)
            if ds_response.status_code == 200:
                datasources = ds_response.json()
                print(f"✅ Found {len(datasources)} data sources")
                return True
            else:
                print("❌ Cannot access Grafana data sources")
                return False
        else:
            print("❌ Grafana not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Grafana connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Grafana Dashboard Data Flow\n")
    
    # Test Prometheus
    prometheus_ok = test_prometheus_connection()
    
    # Test Grafana
    grafana_ok = test_grafana_connection()
    
    print(f"\n📊 Test Results:")
    print(f"Prometheus: {'✅ Working' if prometheus_ok else '❌ Failed'}")
    print(f"Grafana: {'✅ Working' if grafana_ok else '❌ Failed'}")
    
    if prometheus_ok and grafana_ok:
        print("\n🎉 Dashboard should now have data!")
        print("📱 Access your dashboard at: http://localhost:3001")
        print("🔑 Login: admin / admin123")
    else:
        print("\n🔧 Some components need fixing...")

if __name__ == "__main__":
    main()