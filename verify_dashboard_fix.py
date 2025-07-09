#!/usr/bin/env python3
"""
Verification script to confirm the Grafana dashboard is now working properly.
"""

import requests
import json
import time

def test_metrics_availability():
    """Test if metrics are available in Prometheus."""
    print("🔍 Testing metrics availability...")
    
    metrics = [
        'system_cpu_usage_percent',
        'system_memory_usage_percent',
        'system_disk_usage_percent',
        'anomaly_detection_score',
        'anomaly_detection_is_anomaly'
    ]
    
    all_good = True
    for metric in metrics:
        try:
            response = requests.get(f"http://localhost:9090/api/v1/query?query={metric}")
            data = response.json()
            
            if data['data']['result']:
                value = data['data']['result'][0]['value'][1]
                print(f"✅ {metric}: {value}")
            else:
                print(f"❌ {metric}: No data")
                all_good = False
        except Exception as e:
            print(f"❌ {metric}: Error - {e}")
            all_good = False
    
    return all_good

def test_grafana_datasource():
    """Test if Grafana can query Prometheus."""
    print("\n🔍 Testing Grafana data source...")
    
    try:
        # Test through Grafana proxy
        auth = ('admin', 'admin123')
        response = requests.get(
            "http://localhost:3001/api/datasources/proxy/1/api/v1/query?query=system_cpu_usage_percent",
            auth=auth
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data', {}).get('result'):
                result = data['data']['result'][0]
                cpu_value = result['value'][1]
                print(f"✅ Grafana → Prometheus: CPU {cpu_value}%")
                return True
            else:
                print("❌ Grafana → Prometheus: No data returned")
                return False
        else:
            print(f"❌ Grafana → Prometheus: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Grafana → Prometheus: Error - {e}")
        return False

def test_dashboard_access():
    """Test if the dashboard is accessible."""
    print("\n🔍 Testing dashboard access...")
    
    try:
        auth = ('admin', 'admin123')
        
        # Check if dashboards exist
        response = requests.get("http://localhost:3001/api/search", auth=auth)
        
        if response.status_code == 200:
            dashboards = response.json()
            anomaly_dashboards = [d for d in dashboards if 'anomaly' in d.get('title', '').lower()]
            
            if anomaly_dashboards:
                print(f"✅ Found {len(anomaly_dashboards)} anomaly detection dashboard(s)")
                for dashboard in anomaly_dashboards:
                    print(f"   📊 {dashboard['title']} - {dashboard['url']}")
                return True
            else:
                print("❌ No anomaly detection dashboards found")
                return False
        else:
            print(f"❌ Dashboard search failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard access error: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Verifying Grafana Dashboard Fix\n")
    
    # Test metrics
    metrics_ok = test_metrics_availability()
    
    # Test Grafana connection  
    grafana_ok = test_grafana_datasource()
    
    # Test dashboard
    dashboard_ok = test_dashboard_access()
    
    print(f"\n📊 Verification Results:")
    print(f"Metrics Generation: {'✅ Working' if metrics_ok else '❌ Failed'}")
    print(f"Grafana → Prometheus: {'✅ Working' if grafana_ok else '❌ Failed'}")
    print(f"Dashboard Access: {'✅ Working' if dashboard_ok else '❌ Failed'}")
    
    if metrics_ok and grafana_ok and dashboard_ok:
        print(f"\n🎉 SUCCESS! Grafana dashboard is now working!")
        print(f"📱 Access your dashboard at: http://localhost:3001")
        print(f"🔑 Login: admin / admin123")
        print(f"📊 Look for 'Anomaly Detection Dashboard' in the dashboard list")
    else:
        print(f"\n⚠️ Some issues remain. Check the failed components above.")

if __name__ == "__main__":
    main()