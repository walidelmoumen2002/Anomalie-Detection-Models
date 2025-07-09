#!/usr/bin/env python3
"""
Verification script to confirm anomalies are visible in Grafana dashboard.
"""

import requests
import json
import time

def test_current_anomaly_state():
    """Check current anomaly metrics."""
    print("🔍 Checking current anomaly state...")
    
    try:
        # Test direct Prometheus queries
        queries = {
            'CPU Usage': 'system_cpu_usage_percent',
            'Memory Usage': 'system_memory_usage_percent', 
            'Anomaly Score': 'anomaly_detection_score',
            'Is Anomaly': 'anomaly_detection_is_anomaly',
            'CPU Temperature': 'system_cpu_temperature_celsius',
            'Load Average': 'system_load_average'
        }
        
        current_values = {}
        for name, query in queries.items():
            response = requests.get(f"http://localhost:9090/api/v1/query?query={query}")
            data = response.json()
            
            if data['data']['result']:
                value = float(data['data']['result'][0]['value'][1])
                current_values[name] = value
                
                # Determine status
                if name == 'CPU Usage':
                    status = "🚨" if value > 80 else "⚠️" if value > 60 else "✅"
                elif name == 'Anomaly Score':
                    status = "🚨" if value > 0.7 else "⚠️" if value > 0.3 else "✅"
                elif name == 'Is Anomaly':
                    status = "🚨" if value == 1 else "✅"
                elif name == 'CPU Temperature':
                    status = "🔥" if value > 70 else "⚠️" if value > 60 else "✅"
                elif name == 'Load Average':
                    status = "⚡" if value > 4 else "⚠️" if value > 2 else "✅"
                else:
                    status = "📊"
                
                print(f"{status} {name}: {value:.1f}")
            else:
                print(f"❌ {name}: No data")
                
        return current_values
        
    except Exception as e:
        print(f"❌ Error checking metrics: {e}")
        return {}

def test_grafana_data_visualization():
    """Test if Grafana can retrieve the anomaly data."""
    print("\n🔍 Testing Grafana data visualization...")
    
    try:
        auth = ('admin', 'admin123')
        
        # Test key metrics through Grafana proxy
        test_queries = [
            ('CPU Usage', 'system_cpu_usage_percent'),
            ('Anomaly Score', 'anomaly_detection_score'), 
            ('Anomaly Status', 'anomaly_detection_is_anomaly')
        ]
        
        grafana_values = {}
        for name, query in test_queries:
            response = requests.get(
                f"http://localhost:3001/api/datasources/proxy/1/api/v1/query?query={query}",
                auth=auth
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data', {}).get('result'):
                    value = float(data['data']['result'][0]['value'][1])
                    grafana_values[name] = value
                    
                    status = "🚨" if (name == 'CPU Usage' and value > 80) or \
                                    (name == 'Anomaly Score' and value > 0.7) or \
                                    (name == 'Anomaly Status' and value == 1) else "✅"
                    
                    print(f"{status} Grafana → {name}: {value:.1f}")
                else:
                    print(f"❌ Grafana → {name}: No data")
            else:
                print(f"❌ Grafana → {name}: HTTP {response.status_code}")
        
        return grafana_values
        
    except Exception as e:
        print(f"❌ Grafana test error: {e}")
        return {}

def analyze_anomaly_visibility(prometheus_data, grafana_data):
    """Analyze if anomalies are clearly visible."""
    print(f"\n📊 Anomaly Visibility Analysis:")
    print("-" * 40)
    
    # Check if we have strong anomaly signals
    cpu_high = prometheus_data.get('CPU Usage', 0) > 80
    anomaly_score_high = prometheus_data.get('Anomaly Score', 0) > 0.7
    is_anomaly = prometheus_data.get('Is Anomaly', 0) == 1
    temp_high = prometheus_data.get('CPU Temperature', 0) > 70
    load_high = prometheus_data.get('Load Average', 0) > 4
    
    anomaly_signals = sum([cpu_high, anomaly_score_high, is_anomaly, temp_high, load_high])
    
    print(f"Strong Anomaly Signals: {anomaly_signals}/5")
    
    if anomaly_signals >= 3:
        print("🚨 STRONG ANOMALY DETECTED - Dashboard should show clear alerts!")
        visibility = "EXCELLENT"
    elif anomaly_signals >= 2:
        print("⚠️ MODERATE ANOMALY - Dashboard should show warnings")
        visibility = "GOOD"
    elif anomaly_signals >= 1:
        print("📊 WEAK ANOMALY - Dashboard should show some indicators")
        visibility = "FAIR"
    else:
        print("✅ NORMAL STATE - Dashboard should show normal metrics")
        visibility = "NORMAL"
    
    # Check data consistency between Prometheus and Grafana
    consistent = True
    for metric in ['CPU Usage', 'Anomaly Score']:
        if metric in prometheus_data and metric in grafana_data:
            diff = abs(prometheus_data[metric] - grafana_data[metric])
            if diff > 1.0:  # Allow small differences
                print(f"⚠️ Data inconsistency in {metric}: {diff:.1f} difference")
                consistent = False
    
    if consistent:
        print("✅ Data consistency: Prometheus ↔ Grafana")
    else:
        print("❌ Data inconsistency detected")
    
    return visibility

def main():
    """Run visualization verification."""
    print("🚀 Verifying Anomaly Visualization in Dashboard\n")
    
    # Test current state
    prometheus_data = test_current_anomaly_state()
    
    # Test Grafana access
    grafana_data = test_grafana_data_visualization()
    
    # Analyze visibility
    visibility = analyze_anomaly_visibility(prometheus_data, grafana_data)
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"Dashboard Visibility: {visibility}")
    print(f"🔗 Access your dashboard: http://localhost:3001")
    print(f"📊 Look for the 'Anomaly Detection Dashboard'")
    
    if visibility in ["EXCELLENT", "GOOD"]:
        print("\n🎉 SUCCESS! Your dashboard should now clearly show anomalies!")
        print("Expected visuals:")
        print("- CPU usage graphs spiking to 95%+")
        print("- Anomaly score near 1.0") 
        print("- Temperature warnings")
        print("- Load average alerts")
        print("- Anomaly detection status showing 'ACTIVE'")
    else:
        print(f"\n⚠️ Anomaly visibility may be limited. Current state: {visibility}")

if __name__ == "__main__":
    main()