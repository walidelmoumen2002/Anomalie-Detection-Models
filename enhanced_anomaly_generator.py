#!/usr/bin/env python3
"""
Enhanced Anomaly Generator for Grafana Dashboard Testing
========================================================

This script generates more frequent and dramatic anomalies to properly demonstrate
the anomaly detection dashboard capabilities.

Features:
- Higher anomaly frequency (20% instead of 5%)
- Multiple anomaly patterns (spikes, sustained load, gradual degradation)
- Realistic anomaly scenarios (CPU storms, memory leaks, network attacks)
- Better anomaly scoring that correlates with actual conditions
- Anomaly persistence (anomalies last for multiple cycles)
"""

import time
import random
import math
from prometheus_client import start_http_server, Gauge, Counter, Info
import threading
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create metrics
cpu_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage')
disk_gauge = Gauge('system_disk_usage_percent', 'Disk usage percentage')
network_bytes_sent = Gauge('system_network_bytes_sent_total', 'Network bytes sent')
network_bytes_recv = Gauge('system_network_bytes_recv_total', 'Network bytes received')
anomaly_score = Gauge('anomaly_detection_score', 'Anomaly detection score')
is_anomaly = Gauge('anomaly_detection_is_anomaly', 'Is current state anomalous (1=yes, 0=no)')

# Additional metrics for better visualization
cpu_temperature = Gauge('system_cpu_temperature_celsius', 'CPU temperature')
load_average = Gauge('system_load_average', 'System load average')
active_connections = Gauge('system_active_connections', 'Number of active connections')
error_rate = Gauge('system_error_rate_percent', 'Error rate percentage')

# Info metrics
anomaly_info = Info('anomaly_detection_info', 'Current anomaly information')

class AnomalyState:
    """Tracks the current anomaly state for persistence."""
    def __init__(self):
        self.active = False
        self.type = None
        self.start_time = None
        self.duration = 0
        self.intensity = 0
        self.remaining_cycles = 0

def generate_enhanced_metrics():
    """Generate realistic metrics with frequent, dramatic anomalies."""
    
    anomaly_state = AnomalyState()
    cycle_count = 0
    
    # Base normal values
    base_cpu = 35
    base_memory = 70
    base_disk = 50
    base_network_sent = 1500000
    base_network_recv = 1800000
    
    logger.info("🚀 Starting enhanced anomaly generation...")
    logger.info("📊 Anomaly probability: 20% per cycle")
    logger.info("⏱️ Update interval: 10 seconds")
    
    while True:
        cycle_count += 1
        current_time = datetime.now()
        
        # Check if we should start a new anomaly
        if not anomaly_state.active and random.random() < 0.20:  # 20% chance
            anomaly_state.active = True
            anomaly_state.type = random.choice([
                'cpu_storm', 'memory_leak', 'disk_full', 'network_ddos', 
                'thermal_throttle', 'process_explosion', 'io_storm'
            ])
            anomaly_state.start_time = current_time
            anomaly_state.duration = random.randint(30, 120)  # 30-120 seconds
            anomaly_state.remaining_cycles = anomaly_state.duration // 10
            anomaly_state.intensity = random.uniform(0.6, 1.0)  # 60-100% intensity
            
            logger.warning(f"🚨 NEW ANOMALY: {anomaly_state.type} "
                          f"(duration: {anomaly_state.duration}s, intensity: {anomaly_state.intensity:.1%})")
        
        # Generate base metrics with daily patterns
        hour = current_time.hour
        day_factor = 0.5 + 0.3 * math.sin((hour - 6) * math.pi / 12)  # Peak at 2pm, low at 2am
        
        # Base values with some variance
        cpu = base_cpu + 10 * day_factor + random.gauss(0, 5)
        memory = base_memory + 5 * day_factor + random.gauss(0, 3)
        disk = base_disk + random.gauss(0, 2)
        network_sent = base_network_sent * (0.8 + 0.4 * day_factor) + random.randint(0, 200000)
        network_recv = base_network_recv * (0.8 + 0.4 * day_factor) + random.randint(0, 240000)
        
        # Additional metrics
        temperature = 45 + cpu * 0.3 + random.gauss(0, 2)
        load_avg = cpu / 30 + random.gauss(0, 0.2)
        connections = 100 + int(network_sent / 50000) + random.randint(0, 50)
        error_pct = max(0, random.gauss(1, 0.5))
        
        # Apply anomaly effects
        anomaly_detected = False
        current_anomaly_score = 0
        anomaly_description = "Normal operation"
        
        if anomaly_state.active and anomaly_state.remaining_cycles > 0:
            anomaly_detected = True
            intensity = anomaly_state.intensity
            
            if anomaly_state.type == 'cpu_storm':
                cpu = min(98, 85 + intensity * 10 + random.gauss(0, 2))
                temperature = min(85, 60 + intensity * 20 + random.gauss(0, 3))
                load_avg = intensity * 8 + random.gauss(0, 0.5)
                anomaly_description = f"CPU Storm (intensity: {intensity:.1%})"
                
            elif anomaly_state.type == 'memory_leak':
                leak_progress = 1 - (anomaly_state.remaining_cycles / (anomaly_state.duration // 10))
                memory = min(95, 75 + intensity * 20 * leak_progress + random.gauss(0, 1))
                anomaly_description = f"Memory Leak (progress: {leak_progress:.1%})"
                
            elif anomaly_state.type == 'disk_full':
                disk = min(95, 80 + intensity * 15 + random.gauss(0, 1))
                error_pct = max(error_pct, intensity * 10)
                anomaly_description = f"Disk Space Critical"
                
            elif anomaly_state.type == 'network_ddos':
                network_sent *= (2 + intensity * 8)
                network_recv *= (2 + intensity * 8)
                connections = int(connections * (3 + intensity * 7))
                error_pct = max(error_pct, intensity * 15)
                anomaly_description = f"Network DDoS Attack"
                
            elif anomaly_state.type == 'thermal_throttle':
                temperature = min(95, 70 + intensity * 20 + random.gauss(0, 2))
                cpu = max(10, cpu - intensity * 30)  # CPU throttles down
                anomaly_description = f"Thermal Throttling"
                
            elif anomaly_state.type == 'process_explosion':
                cpu = min(95, 70 + intensity * 25 + random.gauss(0, 3))
                memory = min(90, memory + intensity * 15 + random.gauss(0, 2))
                load_avg = intensity * 12 + random.gauss(0, 1)
                anomaly_description = f"Process Explosion"
                
            elif anomaly_state.type == 'io_storm':
                cpu = min(80, cpu + intensity * 20)  # IO wait
                disk = min(85, disk + intensity * 10)
                load_avg = intensity * 6 + random.gauss(0, 0.5)
                anomaly_description = f"I/O Storm"
            
            anomaly_state.remaining_cycles -= 1
            
            if anomaly_state.remaining_cycles <= 0:
                anomaly_state.active = False
                logger.info(f"✅ Anomaly resolved: {anomaly_state.type}")
        
        # Calculate comprehensive anomaly score
        cpu_score = max(0, (cpu - 70) / 25) if cpu > 70 else 0
        memory_score = max(0, (memory - 85) / 10) if memory > 85 else 0
        disk_score = max(0, (disk - 80) / 15) if disk > 80 else 0
        network_score = max(0, (network_sent / base_network_sent - 3) / 5) if network_sent > base_network_sent * 3 else 0
        temp_score = max(0, (temperature - 70) / 20) if temperature > 70 else 0
        load_score = max(0, (load_avg - 4) / 8) if load_avg > 4 else 0
        error_score = max(0, error_pct / 20) if error_pct > 5 else 0
        
        current_anomaly_score = min(1.0, max(cpu_score, memory_score, disk_score, 
                                            network_score, temp_score, load_score, error_score))
        
        # Ensure bounds
        cpu = max(5, min(100, cpu))
        memory = max(30, min(100, memory))
        disk = max(10, min(100, disk))
        temperature = max(25, min(100, temperature))
        load_avg = max(0, load_avg)
        connections = max(10, connections)
        error_pct = max(0, min(100, error_pct))
        
        # Update Prometheus metrics
        cpu_gauge.set(round(cpu, 1))
        memory_gauge.set(round(memory, 1))
        disk_gauge.set(round(disk, 1))
        network_bytes_sent.set(int(network_sent))
        network_bytes_recv.set(int(network_recv))
        anomaly_score.set(round(current_anomaly_score, 3))
        is_anomaly.set(1 if anomaly_detected else 0)
        
        cpu_temperature.set(round(temperature, 1))
        load_average.set(round(load_avg, 2))
        active_connections.set(int(connections))
        error_rate.set(round(error_pct, 2))
        
        # Update anomaly info
        anomaly_info.info({
            'type': anomaly_state.type if anomaly_state.active else 'none',
            'description': anomaly_description,
            'active': str(anomaly_detected).lower(),
            'score': f"{current_anomaly_score:.3f}",
            'cycle': str(cycle_count)
        })
        
        # Enhanced logging
        status_icon = "🚨" if anomaly_detected else "✅"
        temp_warning = "🔥" if temperature > 70 else ""
        load_warning = "⚡" if load_avg > 4 else ""
        
        logger.info(f"{status_icon} Cycle {cycle_count}: "
                   f"CPU: {cpu:.1f}% {temp_warning}({temperature:.1f}°C), "
                   f"Memory: {memory:.1f}%, "
                   f"Load: {load_avg:.2f} {load_warning}, "
                   f"Score: {current_anomaly_score:.3f}")
        
        if anomaly_detected:
            logger.warning(f"   🔍 {anomaly_description} "
                          f"(remaining: {anomaly_state.remaining_cycles * 10}s)")
        
        # Wait for next cycle
        time.sleep(10)

def main():
    """Start the enhanced metrics server."""
    logger.info("🚀 Starting Enhanced Anomaly Generator...")
    
    # Start HTTP server for Prometheus scraping
    start_http_server(8000)
    logger.info("✅ Metrics server started on http://localhost:8000/metrics")
    
    # Start metrics generation in background thread
    metrics_thread = threading.Thread(target=generate_enhanced_metrics, daemon=True)
    metrics_thread.start()
    logger.info("📊 Enhanced metrics generation started")
    logger.info("🎯 Watch for frequent anomalies in your Grafana dashboard!")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("⏹️ Stopping enhanced anomaly generator...")

if __name__ == "__main__":
    main()