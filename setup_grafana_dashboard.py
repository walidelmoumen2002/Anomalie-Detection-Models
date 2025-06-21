#!/usr/bin/env python3
"""
Setup Grafana Dashboard Automatically
====================================

This script will automatically configure the Grafana dashboard for anomaly detection.
"""

import requests
import json
import time

GRAFANA_URL = "http://localhost:3001"
GRAFANA_USER = "admin"
GRAFANA_PASS = "admin123"

def setup_grafana():
    """Setup Grafana dashboard automatically."""
    
    print("🎛️ Setting up Grafana dashboard...")
    
    # Wait for Grafana to be ready
    print("⏳ Waiting for Grafana to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(2)
    else:
        print("❌ Grafana not responding")
        return False
    
    # Basic auth
    auth = (GRAFANA_USER, GRAFANA_PASS)
    
    # Check if Prometheus datasource exists
    print("🔍 Checking Prometheus datasource...")
    response = requests.get(f"{GRAFANA_URL}/api/datasources", auth=auth)
    
    if response.status_code == 200:
        datasources = response.json()
        prometheus_exists = any(ds['name'] == 'Prometheus' for ds in datasources)
        
        if not prometheus_exists:
            print("➕ Adding Prometheus datasource...")
            datasource_config = {
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://host.docker.internal:9090",
                "access": "proxy",
                "isDefault": True
            }
            
            response = requests.post(f"{GRAFANA_URL}/api/datasources", 
                                   auth=auth, 
                                   json=datasource_config)
            
            if response.status_code in [200, 409]:
                print("✅ Prometheus datasource added")
            else:
                print(f"❌ Failed to add datasource: {response.text}")
        else:
            print("✅ Prometheus datasource already exists")
    
    # Create dashboard
    print("📊 Creating anomaly detection dashboard...")
    
    dashboard_config = {
        "dashboard": {
            "id": None,
            "title": "Anomaly Detection Live",
            "tags": ["anomaly", "mlops"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "🚨 Anomaly Alert",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "anomaly_detection_is_anomaly",
                            "legendFormat": "Anomaly Status"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {
                                "mode": "thresholds"
                            },
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "red", "value": 1}
                                ]
                            },
                            "mappings": [
                                {"options": {"0": {"text": "NORMAL"}}, "type": "value"},
                                {"options": {"1": {"text": "ANOMALY!"}}, "type": "value"}
                            ]
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "🎯 Anomaly Score",
                    "type": "gauge",
                    "targets": [
                        {
                            "expr": "anomaly_detection_score",
                            "legendFormat": "Score"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "min": 0,
                            "max": 1,
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 0.3},
                                    {"color": "red", "value": 0.7}
                                ]
                            }
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
                },
                {
                    "id": 3,
                    "title": "💻 CPU Usage",
                    "type": "gauge",
                    "targets": [
                        {
                            "expr": "system_cpu_usage_percent",
                            "legendFormat": "CPU %"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100,
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 70},
                                    {"color": "red", "value": 90}
                                ]
                            }
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
                },
                {
                    "id": 4,
                    "title": "🧠 Memory Usage",
                    "type": "gauge",
                    "targets": [
                        {
                            "expr": "system_memory_usage_percent",
                            "legendFormat": "Memory %"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100,
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 80},
                                    {"color": "red", "value": 95}
                                ]
                            }
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0}
                },
                {
                    "id": 5,
                    "title": "📈 System Metrics Timeline",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "system_cpu_usage_percent",
                            "legendFormat": "CPU %"
                        },
                        {
                            "expr": "system_memory_usage_percent",
                            "legendFormat": "Memory %"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100
                        }
                    },
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                },
                {
                    "id": 6,
                    "title": "🚨 Anomaly Detection Timeline",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "anomaly_detection_is_anomaly",
                            "legendFormat": "Anomaly Detected"
                        },
                        {
                            "expr": "anomaly_detection_score",
                            "legendFormat": "Anomaly Score"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
                }
            ],
            "time": {
                "from": "now-5m",
                "to": "now"
            },
            "refresh": "5s"
        },
        "overwrite": True
    }
    
    response = requests.post(f"{GRAFANA_URL}/api/dashboards/db", 
                           auth=auth, 
                           json=dashboard_config)
    
    if response.status_code == 200:
        result = response.json()
        dashboard_url = f"{GRAFANA_URL}/d/{result['uid']}/anomaly-detection-live"
        print("✅ Dashboard created successfully!")
        print(f"🔗 Dashboard URL: {dashboard_url}")
        return True
    else:
        print(f"❌ Failed to create dashboard: {response.text}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Grafana Dashboard for Anomaly Detection")
    print("=" * 55)
    
    if setup_grafana():
        print("\n🎉 Setup complete!")
        print(f"📊 Open Grafana: {GRAFANA_URL}")
        print("👤 Login: admin/admin123")
        print("📈 Navigate to 'Anomaly Detection Live' dashboard")
        print("\n💡 To see anomalies in action:")
        print("   python generate_load.py")
    else:
        print("\n❌ Setup failed. Please check Grafana manually.")
        print(f"🌐 Grafana URL: {GRAFANA_URL}")
        print("👤 Login: admin/admin123")