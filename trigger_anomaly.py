#!/usr/bin/env python3
"""
Manual Anomaly Trigger for Testing
==================================

This script allows you to manually trigger specific anomalies to immediately 
test the Grafana dashboard visualization.
"""

import time
import random
from prometheus_client import start_http_server, Gauge, Counter, Info
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create metrics (same as enhanced generator)
cpu_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage')  
disk_gauge = Gauge('system_disk_usage_percent', 'Disk usage percentage')
network_bytes_sent = Gauge('system_network_bytes_sent_total', 'Network bytes sent')
network_bytes_recv = Gauge('system_network_bytes_recv_total', 'Network bytes received')
anomaly_score = Gauge('anomaly_detection_score', 'Anomaly detection score')
is_anomaly = Gauge('anomaly_detection_is_anomaly', 'Is current state anomalous (1=yes, 0=no)')

cpu_temperature = Gauge('system_cpu_temperature_celsius', 'CPU temperature')
load_average = Gauge('system_load_average', 'System load average')
active_connections = Gauge('system_active_connections', 'Number of active connections')
error_rate = Gauge('system_error_rate_percent', 'Error rate percentage')
anomaly_info = Info('anomaly_detection_info', 'Current anomaly information')

def trigger_cpu_storm(duration=60):
    """Trigger a CPU storm anomaly."""
    logger.warning("🚨 TRIGGERING CPU STORM ANOMALY!")
    
    cycles = duration // 10
    for i in range(cycles):
        # High CPU with thermal effects
        cpu = 95 + random.gauss(0, 2)
        memory = 75 + random.gauss(0, 5)
        disk = 52 + random.gauss(0, 2)
        temperature = 85 + random.gauss(0, 3)
        load_avg = 8 + random.gauss(0, 1)
        
        # Network stays normal
        network_sent = 1500000 + random.randint(0, 200000)
        network_recv = 1800000 + random.randint(0, 240000)
        connections = 150 + random.randint(0, 50)
        error_pct = 2 + random.gauss(0, 1)
        
        # High anomaly score
        anomaly_score_val = 0.9 + random.uniform(0, 0.1)
        
        # Update metrics
        cpu_gauge.set(round(cpu, 1))
        memory_gauge.set(round(memory, 1))
        disk_gauge.set(round(disk, 1))
        network_bytes_sent.set(int(network_sent))
        network_bytes_recv.set(int(network_recv))
        anomaly_score.set(round(anomaly_score_val, 3))
        is_anomaly.set(1)
        
        cpu_temperature.set(round(temperature, 1))
        load_average.set(round(load_avg, 2))
        active_connections.set(int(connections))
        error_rate.set(round(error_pct, 2))
        
        anomaly_info.info({
            'type': 'cpu_storm',
            'description': f'CPU Storm Attack (cycle {i+1}/{cycles})',
            'active': 'true',
            'score': f"{anomaly_score_val:.3f}",
            'remaining_seconds': str((cycles - i - 1) * 10)
        })
        
        logger.warning(f"🔥 CPU Storm - Cycle {i+1}/{cycles}: "
                      f"CPU: {cpu:.1f}%, Temp: {temperature:.1f}°C, "
                      f"Load: {load_avg:.2f}, Score: {anomaly_score_val:.3f}")
        
        time.sleep(10)
    
    logger.info("✅ CPU Storm anomaly completed - returning to normal")
    return_to_normal()

def trigger_memory_leak(duration=90):
    """Trigger a progressive memory leak anomaly."""
    logger.warning("🚨 TRIGGERING MEMORY LEAK ANOMALY!")
    
    cycles = duration // 10
    for i in range(cycles):
        progress = i / cycles
        
        # Progressive memory increase
        cpu = 45 + progress * 20 + random.gauss(0, 3)
        memory = 70 + progress * 25 + random.gauss(0, 2)
        disk = 50 + random.gauss(0, 2)
        temperature = 50 + progress * 15 + random.gauss(0, 2)
        load_avg = 2 + progress * 4 + random.gauss(0, 0.5)
        
        network_sent = 1500000 + random.randint(0, 200000)
        network_recv = 1800000 + random.randint(0, 240000)
        connections = 120 + int(progress * 100) + random.randint(0, 30)
        error_pct = 1 + progress * 8 + random.gauss(0, 1)
        
        # Progressive anomaly score
        anomaly_score_val = min(1.0, 0.3 + progress * 0.7)
        
        # Update metrics
        cpu_gauge.set(round(cpu, 1))
        memory_gauge.set(round(memory, 1))
        disk_gauge.set(round(disk, 1))
        network_bytes_sent.set(int(network_sent))
        network_bytes_recv.set(int(network_recv))
        anomaly_score.set(round(anomaly_score_val, 3))
        is_anomaly.set(1)
        
        cpu_temperature.set(round(temperature, 1))
        load_average.set(round(load_avg, 2))
        active_connections.set(int(connections))
        error_rate.set(round(error_pct, 2))
        
        anomaly_info.info({
            'type': 'memory_leak',
            'description': f'Memory Leak ({progress:.1%} progress)',
            'active': 'true',
            'score': f"{anomaly_score_val:.3f}",
            'remaining_seconds': str((cycles - i - 1) * 10)
        })
        
        logger.warning(f"🧠 Memory Leak - Cycle {i+1}/{cycles}: "
                      f"Memory: {memory:.1f}% ({progress:.1%} leaked), "
                      f"CPU: {cpu:.1f}%, Score: {anomaly_score_val:.3f}")
        
        time.sleep(10)
    
    logger.info("✅ Memory leak anomaly completed - returning to normal")
    return_to_normal()

def trigger_network_attack(duration=45):
    """Trigger a network DDoS attack anomaly."""
    logger.warning("🚨 TRIGGERING NETWORK DDOS ANOMALY!")
    
    cycles = duration // 10
    for i in range(cycles):
        # Normal CPU/memory but extreme network
        cpu = 40 + random.gauss(0, 5)
        memory = 75 + random.gauss(0, 3)
        disk = 50 + random.gauss(0, 2)
        temperature = 50 + random.gauss(0, 3)
        load_avg = 1.5 + random.gauss(0, 0.5)
        
        # Extreme network traffic
        network_sent = 15000000 + random.randint(0, 5000000)  # 10x normal
        network_recv = 18000000 + random.randint(0, 6000000)  # 10x normal
        connections = 2000 + random.randint(0, 500)  # 15x normal
        error_pct = 25 + random.gauss(0, 5)  # High error rate
        
        anomaly_score_val = 0.95 + random.uniform(0, 0.05)
        
        # Update metrics
        cpu_gauge.set(round(cpu, 1))
        memory_gauge.set(round(memory, 1))
        disk_gauge.set(round(disk, 1))
        network_bytes_sent.set(int(network_sent))
        network_bytes_recv.set(int(network_recv))
        anomaly_score.set(round(anomaly_score_val, 3))
        is_anomaly.set(1)
        
        cpu_temperature.set(round(temperature, 1))
        load_average.set(round(load_avg, 2))
        active_connections.set(int(connections))
        error_rate.set(round(error_pct, 2))
        
        anomaly_info.info({
            'type': 'network_ddos',
            'description': f'DDoS Attack (cycle {i+1}/{cycles})',
            'active': 'true',
            'score': f"{anomaly_score_val:.3f}",
            'remaining_seconds': str((cycles - i - 1) * 10)
        })
        
        logger.warning(f"🌐 DDoS Attack - Cycle {i+1}/{cycles}: "
                      f"Network: {network_sent/1000000:.1f}MB/s sent, "
                      f"Connections: {connections}, Errors: {error_pct:.1f}%")
        
        time.sleep(10)
    
    logger.info("✅ Network attack anomaly completed - returning to normal")
    return_to_normal()

def return_to_normal():
    """Return all metrics to normal values."""
    logger.info("🔄 Returning to normal operation...")
    
    # Normal values
    cpu_gauge.set(35.0)
    memory_gauge.set(70.0)
    disk_gauge.set(50.0)
    network_bytes_sent.set(1500000)
    network_bytes_recv.set(1800000)
    anomaly_score.set(0.0)
    is_anomaly.set(0)
    
    cpu_temperature.set(45.0)
    load_average.set(1.0)
    active_connections.set(120)
    error_rate.set(1.0)
    
    anomaly_info.info({
        'type': 'none',
        'description': 'Normal operation resumed',
        'active': 'false',
        'score': '0.000',
        'remaining_seconds': '0'
    })

def main():
    """Run anomaly demonstrations."""
    logger.info("🚀 Starting Manual Anomaly Trigger...")
    
    # Start HTTP server
    start_http_server(8000)
    logger.info("✅ Metrics server started on http://localhost:8000/metrics")
    
    # Start with normal values
    return_to_normal()
    
    logger.info("📊 Dashboard should now show normal metrics")
    logger.info("🎯 Starting anomaly demonstration sequence...")
    
    # Wait a moment for normal baseline
    time.sleep(20)
    
    # Demonstrate different anomaly types
    trigger_cpu_storm(60)      # 1 minute CPU storm
    time.sleep(30)             # 30 seconds normal
    
    trigger_memory_leak(90)    # 1.5 minute memory leak  
    time.sleep(30)             # 30 seconds normal
    
    trigger_network_attack(45) # 45 second network attack
    time.sleep(30)             # 30 seconds normal
    
    logger.info("🎉 Anomaly demonstration completed!")
    logger.info("📱 Check your Grafana dashboard at http://localhost:3001")
    
    # Keep normal metrics running
    while True:
        return_to_normal()
        time.sleep(60)

if __name__ == "__main__":
    main()