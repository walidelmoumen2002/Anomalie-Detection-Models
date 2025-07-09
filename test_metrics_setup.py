#!/usr/bin/env python3
"""
Simple test script to generate metrics for Grafana dashboard testing.
This bypasses the complex Docker networking issues and creates a simple metrics server.
"""

import time
import random
from prometheus_client import start_http_server, Gauge, Counter
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create metrics
cpu_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage')
disk_gauge = Gauge('system_disk_usage_percent', 'Disk usage percentage')
network_bytes_sent = Gauge('system_network_bytes_sent_total', 'Network bytes sent')
network_bytes_recv = Gauge('system_network_bytes_recv_total', 'Network bytes received')
anomaly_score = Gauge('anomaly_detection_score', 'Anomaly detection score')
is_anomaly = Gauge('anomaly_detection_is_anomaly', 'Is current state anomalous (1=yes, 0=no)')

def generate_realistic_metrics():
    """Generate realistic system metrics with occasional anomalies."""
    base_time = time.time()
    
    while True:
        # Generate realistic system metrics
        current_time = time.time()
        
        # CPU usage: 20-80% with some variance
        cpu = 30 + 30 * abs(random.gauss(0, 0.3)) + 10 * random.random()
        cpu = max(5, min(95, cpu))
        
        # Memory usage: 60-90% 
        memory = 70 + 15 * random.gauss(0, 0.5)
        memory = max(40, min(95, memory))
        
        # Disk usage: relatively stable around 50%
        disk = 50 + 5 * random.gauss(0, 0.2)
        disk = max(10, min(90, disk))
        
        # Network I/O: varies throughout the day
        hour_factor = abs(random.gauss(0, 0.3))
        network_sent = 1000000 + 500000 * hour_factor + random.randint(0, 100000)
        network_recv = 1200000 + 600000 * hour_factor + random.randint(0, 120000)
        
        # Anomaly detection: mostly normal, occasional anomalies
        anomaly_probability = 0.05  # 5% chance of anomaly
        is_anomalous = random.random() < anomaly_probability
        
        if is_anomalous:
            # Create anomalous conditions
            anomaly_type = random.choice(['cpu_spike', 'memory_spike', 'network_spike'])
            if anomaly_type == 'cpu_spike':
                cpu = min(95, cpu + random.uniform(20, 40))
            elif anomaly_type == 'memory_spike':
                memory = min(95, memory + random.uniform(15, 25))
            elif anomaly_type == 'network_spike':
                network_sent *= random.uniform(2, 5)
                network_recv *= random.uniform(2, 5)
        
        # Calculate anomaly score (higher when metrics are unusual)
        cpu_anomaly = max(0, abs(cpu - 50) - 20) / 30
        memory_anomaly = max(0, abs(memory - 75) - 15) / 20
        network_anomaly = max(0, (network_sent / 2000000) - 1)
        
        score = min(1.0, max(cpu_anomaly, memory_anomaly, network_anomaly))
        
        # Update Prometheus metrics
        cpu_gauge.set(round(cpu, 1))
        memory_gauge.set(round(memory, 1))
        disk_gauge.set(round(disk, 1))
        network_bytes_sent.set(int(network_sent))
        network_bytes_recv.set(int(network_recv))
        anomaly_score.set(round(score, 3))
        is_anomaly.set(1 if is_anomalous else 0)
        
        # Log current state
        status = "🚨 ANOMALY" if is_anomalous else "✅ Normal"
        logger.info(f"{status} - CPU: {cpu:.1f}%, Memory: {memory:.1f}%, "
                   f"Disk: {disk:.1f}%, Score: {score:.3f}")
        
        # Wait 10 seconds before next update
        time.sleep(10)

def main():
    """Start the metrics server and generator."""
    logger.info("🚀 Starting test metrics server...")
    
    # Start HTTP server for Prometheus scraping
    start_http_server(8000)
    logger.info("✅ Metrics server started on http://localhost:8000/metrics")
    
    # Start metrics generation in background thread
    metrics_thread = threading.Thread(target=generate_realistic_metrics, daemon=True)
    metrics_thread.start()
    logger.info("📊 Metrics generation started")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("⏹️ Stopping metrics server...")

if __name__ == "__main__":
    main()