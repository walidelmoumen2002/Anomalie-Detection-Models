#!/usr/bin/env python3
"""
Auto Generate CPU Load for Testing
==================================
"""

import time
import threading
import multiprocessing

def cpu_intensive_task(duration=30):
    """Generate CPU load for specified duration."""
    end_time = time.time() + duration
    
    while time.time() < end_time:
        # CPU intensive operation
        [x**2 for x in range(10000)]

def generate_high_cpu_load(duration=30, num_threads=None):
    """Generate high CPU load using multiple threads."""
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    
    print(f"🔥 Generating CPU load with {num_threads} threads for {duration} seconds...")
    print("🎯 Watch your Grafana dashboard for anomaly detection!")
    
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=cpu_intensive_task, args=(duration,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("✅ Load generation completed!")

if __name__ == "__main__":
    generate_high_cpu_load(20)  # 20 seconds of high load