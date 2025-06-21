#!/usr/bin/env python3
"""
Real System Metrics Collector
=============================

Collects actual system metrics from your local machine and feeds them to the
anomaly detection models. This uses real CPU, memory, disk, and network data.

Requirements:
    pip install psutil

Usage:
    python pipeline/real_metrics_collector.py --config config/local.json
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
import argparse
import os
import sys

# Add path to core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("❌ psutil not installed. Install with: pip install psutil")
    PSUTIL_AVAILABLE = False
    exit(1)

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    print("⚠️ kafka-python not installed. Install with: pip install kafka-python")
    KAFKA_AVAILABLE = False

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    print("⚠️ prometheus-client not installed. Install with: pip install prometheus-client")
    PROMETHEUS_AVAILABLE = False

# Import models for real-time detection
from isolation_forest import IsolationForestModel
from random_forest import RandomForestModel

logger = logging.getLogger(__name__)

class RealMetricsCollector:
    """
    Collects real system metrics from the local machine.
    """
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.kafka_producer = None
        self.prometheus_registry = None
        self.model = None
        self.baseline_metrics = {}
        self.metrics_history = []
        
        self._setup_logging()
        self._setup_kafka()
        self._setup_prometheus()
        self._setup_model()
        self._collect_baseline()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _setup_kafka(self):
        """Setup Kafka producer if available."""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available - will only log data")
            return
        
        try:
            kafka_config = self.config['monitoring']['kafka']
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_config['bootstrap_servers'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            logger.info("✅ Kafka producer connected")
        except Exception as e:
            logger.warning(f"Kafka connection failed: {e}")
            self.kafka_producer = None
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics if available."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available")
            return
        
        try:
            # Start HTTP server for Prometheus scraping
            prometheus_config = self.config['monitoring']['prometheus']
            start_http_server(prometheus_config['metrics_port'])
            
            self.prometheus_registry = CollectorRegistry()
            
            # Create metrics
            self.cpu_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
            self.memory_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage')
            self.disk_gauge = Gauge('system_disk_usage_percent', 'Disk usage percentage')
            self.network_bytes_sent = Gauge('system_network_bytes_sent_total', 'Network bytes sent')
            self.network_bytes_recv = Gauge('system_network_bytes_recv_total', 'Network bytes received')
            self.anomaly_score = Gauge('anomaly_detection_score', 'Anomaly detection score')
            self.is_anomaly = Gauge('anomaly_detection_is_anomaly', 'Is current state anomalous (1=yes, 0=no)')
            
            logger.info(f"✅ Prometheus metrics server started on port {prometheus_config['metrics_port']}")
        except Exception as e:
            logger.warning(f"Prometheus setup failed: {e}")
            self.prometheus_registry = None
    
    def _setup_model(self):
        """Setup anomaly detection model."""
        try:
            # Use a simple model for real-time detection
            self.model = IsolationForestModel(
                contamination=0.1,  # Higher contamination for local testing
                n_estimators=50,    # Faster training
                random_state=42
            )
            logger.info("✅ Anomaly detection model initialized")
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            self.model = None
    
    def _collect_baseline(self):
        """Collect baseline metrics to establish normal behavior."""
        logger.info("📊 Collecting baseline metrics (30 seconds)...")
        
        baseline_samples = []
        for i in range(6):  # 30 seconds / 5-second intervals
            metrics = self.collect_system_metrics()
            baseline_samples.append([
                metrics['cpu_usage'],
                metrics['ram_usage'],
                metrics['disk_usage'],
                metrics['network_bytes_sent'],
                metrics['network_bytes_recv']
            ])
            time.sleep(5)
        
        # Train model with baseline data
        if self.model and baseline_samples:
            import numpy as np
            X_baseline = np.array(baseline_samples)
            self.model.fit(X_baseline)
            logger.info("✅ Model trained with baseline data")
    
    def collect_system_metrics(self) -> Dict:
        """Collect real system metrics using psutil."""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage (root partition)
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (on Unix systems)
        try:
            load_avg = os.getloadavg()[0]  # 1-minute load average
        except (OSError, AttributeError):
            load_avg = 0  # Windows doesn't have load average
        
        # Boot time (uptime)
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        # Temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            cpu_temp = 0
            if temps:
                # Get first available temperature
                for name, entries in temps.items():
                    if entries:
                        cpu_temp = entries[0].current
                        break
        except (AttributeError, OSError):
            cpu_temp = 0
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "host": psutil.os.uname().nodename if hasattr(psutil.os, 'uname') else "localhost",
            "cpu_usage": round(cpu_percent, 2),
            "ram_usage": round(memory_percent, 2),
            "disk_usage": round(disk_percent, 2),
            "disk_io": disk_io.read_bytes + disk_io.write_bytes if disk_io else 0,
            "network_io": network.bytes_sent + network.bytes_recv,
            "network_bytes_sent": network.bytes_sent,
            "network_bytes_recv": network.bytes_recv,
            "process_count": process_count,
            "load_average": round(load_avg, 2),
            "uptime_hours": round(uptime_seconds / 3600, 2),
            "cpu_temperature": round(cpu_temp, 1),
            "response_time": 0,  # Placeholder - could measure actual response time
            "error_rate": 0,     # Placeholder - could collect from logs
            "active_connections": 0,  # Placeholder
            "queue_length": 0    # Placeholder
        }
        
        return metrics
    
    def detect_anomaly(self, metrics: Dict) -> tuple:
        """Detect if current metrics represent an anomaly."""
        if not self.model:
            return False, 0.0
        
        try:
            # Prepare features for model (same format as training)
            features = [
                metrics['cpu_usage'],
                metrics['ram_usage'],
                metrics['disk_usage'],
                metrics['network_bytes_sent'],
                metrics['network_bytes_recv']
            ]
            
            import numpy as np
            X = np.array(features).reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(X)[0]
            
            # Get anomaly score
            scores = self.model.model.decision_function(X)
            anomaly_score = abs(scores[0])
            
            return prediction == -1, anomaly_score  # Isolation Forest returns -1 for anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0
    
    async def send_to_kafka(self, metrics: Dict):
        """Send metrics to Kafka."""
        if not self.kafka_producer:
            return
        
        try:
            topic = self.config['monitoring']['kafka']['topics']['input']
            
            # Send to Kafka
            future = self.kafka_producer.send(topic, value=metrics)
            await asyncio.get_event_loop().run_in_executor(None, future.get, 10)
            
        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")
    
    def update_prometheus_metrics(self, metrics: Dict, is_anomaly: bool, anomaly_score: float):
        """Update Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.cpu_gauge.set(metrics['cpu_usage'])
            self.memory_gauge.set(metrics['ram_usage'])
            self.disk_gauge.set(metrics['disk_usage'])
            self.network_bytes_sent.set(metrics['network_bytes_sent'])
            self.network_bytes_recv.set(metrics['network_bytes_recv'])
            self.anomaly_score.set(anomaly_score)
            self.is_anomaly.set(1 if is_anomaly else 0)
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    async def run_collection(self, duration_minutes: int = 60, interval_seconds: int = 10):
        """Run the real metrics collection."""
        logger.info(f"🚀 Starting real metrics collection for {duration_minutes} minutes")
        logger.info(f"📊 Collecting metrics every {interval_seconds} seconds")
        logger.info(f"🖥️ Monitoring system: {psutil.os.uname().nodename if hasattr(psutil.os, 'uname') else 'localhost'}")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        sample_count = 0
        anomaly_count = 0
        
        while time.time() < end_time:
            try:
                sample_count += 1
                
                # Collect real system metrics
                metrics = self.collect_system_metrics()
                
                # Detect anomalies
                is_anomaly, anomaly_score = self.detect_anomaly(metrics)
                
                if is_anomaly:
                    anomaly_count += 1
                    metrics['anomaly_detected'] = True
                    metrics['anomaly_score'] = anomaly_score
                    
                    logger.warning(f"🚨 ANOMALY DETECTED: CPU: {metrics['cpu_usage']}%, "
                                 f"Memory: {metrics['ram_usage']}%, "
                                 f"Disk: {metrics['disk_usage']}%, "
                                 f"Score: {anomaly_score:.3f}")
                else:
                    metrics['anomaly_detected'] = False
                    metrics['anomaly_score'] = anomaly_score
                
                # Send to monitoring systems
                await self.send_to_kafka(metrics)
                self.update_prometheus_metrics(metrics, is_anomaly, anomaly_score)
                
                # Store in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:  # Keep last 100 samples
                    self.metrics_history.pop(0)
                
                # Log summary
                logger.info(f"📊 Sample {sample_count}: CPU: {metrics['cpu_usage']}%, "
                           f"Memory: {metrics['ram_usage']}%, "
                           f"Disk: {metrics['disk_usage']}%, "
                           f"Anomaly: {'YES' if is_anomaly else 'NO'} (score: {anomaly_score:.3f})")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(interval_seconds)
        
        logger.info(f"✅ Collection completed! Total samples: {sample_count}, Anomalies: {anomaly_count}")
        
        # Save metrics history
        with open('real_metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info("💾 Metrics history saved to real_metrics_history.json")
        
        # Cleanup
        if self.kafka_producer:
            self.kafka_producer.close()

def main():
    parser = argparse.ArgumentParser(description='Real System Metrics Collector')
    parser.add_argument('--config', default='config/local.json', 
                       help='Configuration file path')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Collection duration in minutes')
    parser.add_argument('--interval', type=int, default=15, 
                       help='Interval between samples in seconds')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not PSUTIL_AVAILABLE:
        print("❌ psutil is required. Install with: pip install psutil")
        return 1
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    try:
        collector = RealMetricsCollector(args.config)
        asyncio.run(collector.run_collection(args.duration, args.interval))
        return 0
    except KeyboardInterrupt:
        print("\n⏹️ Collection stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())