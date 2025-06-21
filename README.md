# MLOps Anomaly Detection Pipeline

A production-ready MLOps pipeline for real-time anomaly detection using multiple machine learning models with comprehensive monitoring, metrics collection, and visualization.

## 🚀 Features

- **Real-time System Metrics Collection**: CPU, Memory, and Disk usage monitoring
- **Multiple ML Models**: Isolation Forest, Random Forest, and LSTM for anomaly detection
- **Production Monitoring**: Prometheus metrics exposition and Grafana dashboards
- **Data Streaming**: Kafka integration for real-time data processing
- **MLflow Integration**: Model tracking, versioning, and registry
- **Docker Infrastructure**: Complete containerized monitoring stack
- **Automated Alerts**: Real-time anomaly detection with configurable thresholds

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Monitoring & Dashboards](#monitoring--dashboards)
- [API Reference](#api-reference)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   System        │    │   ML Models     │    │   Monitoring    │
│   Metrics       │────▶│   (Isolation   │────▶│   Stack         │
│   Collection    │    │   Forest, RF,   │    │   (Prometheus,  │
│                 │    │   LSTM)         │    │   Grafana)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kafka         │    │   MLflow        │    │   Real-time     │
│   Streaming     │    │   Tracking      │    │   Alerts        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.11+ (required for TensorFlow compatibility)
- **Docker**: Latest version with Docker Compose
- **Memory**: Minimum 8GB RAM recommended
- **Storage**: 2GB free space for data and models

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd "Anomalie Detection Models"

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Infrastructure Services

```bash
# Start Docker services (Prometheus, Grafana, Kafka)
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 3. Launch Anomaly Detection

```bash
# Start the real-time metrics collection and anomaly detection
python pipeline/real_metrics_collector.py
```

### 4. Configure Grafana Dashboard

```bash
# Setup Grafana dashboard automatically
python setup_grafana_dashboard.py
```

### 5. Access Services

- **Grafana Dashboard**: http://localhost:3001 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Kafka UI**: http://localhost:8080
- **Metrics Endpoint**: http://localhost:8000/metrics

## 📁 Project Structure

```
├── core/                          # ML Models Implementation
│   ├── base_model.py             # Abstract base class for all models
│   ├── isolation_forest.py      # Isolation Forest implementation
│   ├── random_forest.py         # Random Forest implementation
│   └── lstm_model.py             # LSTM neural network implementation
├── pipeline/                     # Data Pipeline & Orchestration
│   ├── real_metrics_collector.py # Real-time system metrics collection
│   ├── main_orchestrator.py     # Main pipeline orchestrator
│   ├── model_comparison.py      # Model comparison utilities
│   └── realtime_connector.py    # Real-time data connectors
├── config/                       # Configuration Files
│   ├── local.json               # Local development configuration
│   └── production.json          # Production configuration
├── monitoring/                   # Monitoring Infrastructure
│   ├── prometheus/
│   │   └── prometheus.yml       # Prometheus configuration
│   └── grafana/
│       ├── datasources/         # Grafana datasource configurations
│       └── dashboards/          # Grafana dashboard definitions
├── tests/                        # Test Suite
│   └── test_complete_pipeline.py # Integration tests
├── docker-compose.yml           # Docker infrastructure definition
├── requirements.txt             # Python dependencies
├── test_pipeline.py             # Health check script
├── auto_generate_load.py        # CPU load generator for testing
└── setup_grafana_dashboard.py   # Automated Grafana setup
```

## ⚙️ Configuration

### Local Configuration (`config/local.json`)

```json
{
  "models": {
    "isolation_forest": {
      "contamination": 0.1,
      "n_estimators": 50
    },
    "random_forest": {
      "n_estimators": 100,
      "contamination": 0.1
    },
    "lstm": {
      "sequence_length": 10,
      "epochs": 50,
      "batch_size": 32
    }
  },
  "data_collection": {
    "interval_seconds": 15,
    "baseline_duration_seconds": 30,
    "collection_duration_minutes": 30
  },
  "services": {
    "prometheus_port": 8000,
    "kafka_bootstrap_servers": "localhost:9092",
    "kafka_topic": "anomaly-metrics"
  }
}
```

### Environment Variables

```bash
# MLflow Configuration
export MLFLOW_TRACKING_URI="file:./mlruns"

# Kafka Configuration
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"

# Prometheus Configuration
export PROMETHEUS_PORT=8000
```

## 📊 Usage

### Basic Anomaly Detection

```python
# Run the complete pipeline
python pipeline/real_metrics_collector.py

# The system will:
# 1. Collect baseline metrics (30 seconds)
# 2. Train anomaly detection models
# 3. Start real-time monitoring
# 4. Expose metrics on port 8000
# 5. Send data to Kafka
```

### Manual Model Training

```python
from core.isolation_forest import IsolationForestModel
import numpy as np

# Create and train model
model = IsolationForestModel()
training_data = np.random.randn(100, 5)  # 100 samples, 5 features
model.train(training_data)

# Detect anomalies
test_data = np.random.randn(10, 5)
predictions = model.predict(test_data)
scores = model.predict_proba(test_data)
```

### Testing Anomaly Detection

```python
# Generate CPU load to trigger anomalies
python auto_generate_load.py

# This will:
# - Generate high CPU load for 20 seconds
# - Trigger anomaly detection
# - Show alerts in Grafana dashboard
```

## 📈 Monitoring & Dashboards

### Grafana Dashboard Features

1. **🚨 Anomaly Alert**: Real-time anomaly status indicator
2. **🎯 Anomaly Score**: Gauge showing current anomaly probability
3. **💻 CPU Usage**: Real-time CPU utilization percentage
4. **🧠 Memory Usage**: System memory consumption
5. **📈 System Metrics Timeline**: Historical trends
6. **🚨 Anomaly Detection Timeline**: Anomaly events over time

### Prometheus Metrics

- `system_cpu_usage_percent`: CPU utilization percentage
- `system_memory_usage_percent`: Memory usage percentage
- `system_disk_usage_percent`: Disk usage percentage
- `anomaly_detection_score`: Current anomaly probability (0-1)
- `anomaly_detection_is_anomaly`: Binary anomaly indicator (0/1)

### Kafka Topics

- `anomaly-metrics`: Real-time system metrics and anomaly scores
- Message format:
  ```json
  {
    "timestamp": "2024-01-01T12:00:00",
    "cpu_percent": 45.2,
    "memory_percent": 78.1,
    "disk_percent": 65.3,
    "anomaly_score": 0.23,
    "is_anomaly": false,
    "model_type": "isolation_forest"
  }
  ```

## 🔧 API Reference

### Metrics Endpoint

**GET** `http://localhost:8000/metrics`

Returns Prometheus-formatted metrics:

```
# HELP system_cpu_usage_percent CPU usage percentage
# TYPE system_cpu_usage_percent gauge
system_cpu_usage_percent 42.5

# HELP anomaly_detection_score Anomaly detection score
# TYPE anomaly_detection_score gauge
anomaly_detection_score 0.156
```

### Health Check

```python
# Run health check for all services
python test_pipeline.py

# Expected output:
# ✅ Anomaly detection metrics are being exposed
# ✅ Prometheus server is healthy
# ✅ Grafana is healthy
# ✅ Kafka UI is accessible
```

## 🛠️ Development

### Adding New Models

1. Extend the `BaseModel` class in `core/base_model.py`
2. Implement required methods: `train()`, `predict()`, `predict_proba()`
3. Add model configuration to `config/local.json`
4. Update the model factory in `real_metrics_collector.py`

```python
from core.base_model import BaseModel

class CustomModel(BaseModel):
    def train(self, data):
        # Implementation
        pass
    
    def predict(self, data):
        # Implementation
        pass
    
    def predict_proba(self, data):
        # Implementation
        pass
```

### Custom Metrics Collection

Extend `RealMetricsCollector` in `pipeline/real_metrics_collector.py`:

```python
def collect_custom_metrics(self):
    """Add your custom metrics here"""
    custom_metric = get_your_metric()
    return {
        'custom_metric': custom_metric,
        **self.collect_system_metrics()
    }
```

### Testing

```bash
# Run integration tests
python tests/test_complete_pipeline.py

# Test individual components
python -m pytest tests/ -v
```

## 🐛 Troubleshooting

### Common Issues

**1. Grafana Dashboard Shows No Data**
```bash
# Check if metrics service is running
curl http://localhost:8000/metrics

# Verify Prometheus can scrape metrics
curl http://localhost:9090/api/v1/query?query=up

# Update Grafana datasource if needed
python setup_grafana_dashboard.py
```

**2. Docker Services Won't Start**
```bash
# Check port conflicts
docker-compose ps
netstat -tulpn | grep :9090

# Restart services
docker-compose down
docker-compose up -d
```

**3. Python Dependencies Issues**
```bash
# Ensure Python 3.11 is used
python --version

# Recreate virtual environment
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. MLflow Tracking Issues**
```bash
# Check MLflow directory permissions
ls -la mlruns/

# Reset MLflow if needed
rm -rf mlruns/
```

### Performance Tuning

- **Memory Usage**: Adjust `sequence_length` for LSTM models
- **CPU Usage**: Modify `collection_interval` in configuration
- **Model Accuracy**: Tune `contamination` parameter for unsupervised models

### Logs and Debugging

```bash
# View real-time logs
tail -f /path/to/logs

# Docker service logs
docker-compose logs prometheus
docker-compose logs grafana

# Python application logs
python pipeline/real_metrics_collector.py --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: This README and inline code documentation
- **Community**: Discussions and Q&A in GitHub Discussions

---

**🎯 Ready to Start?** Follow the [Quick Start](#quick-start) guide to get your anomaly detection pipeline running in minutes!