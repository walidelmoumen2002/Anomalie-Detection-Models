#!/usr/bin/env python3
"""
Real-time Data Connector for Production MLOps Pipeline
=====================================================

Connects trained models to Kafka, Prometheus, and Grafana for production deployment.
Handles real-time data ingestion, anomaly detection, and monitoring integration.

Author: MLOps Team
Date: June 2025
Version: 1.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import requests

# Add path to core modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from isolation_forest import IsolationForestModel
from random_forest import RandomForestModel
from lstm_model import LSTMModel

# Prometheus metrics
MESSAGES_PROCESSED = Counter('kafka_messages_processed_total', 'Total processed messages')
ANOMALIES_DETECTED = Counter('anomalies_detected_total', 'Total anomalies detected')
PROCESSING_TIME = Histogram('message_processing_seconds', 'Time spent processing messages')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy', ['model_name'])

logger = logging.getLogger(__name__)

class RealTimeAnomalyDetector:
    """
    Production-ready real-time anomaly detection service.
    Integrates with Kafka for data streams and Prometheus for monitoring.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the real-time detector with configuration.
        
        Args:
            config_path: Path to production configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.current_model = None
        self.kafka_consumer = None
        self.kafka_producer = None
        self.metrics_buffer = []
        
        # Initialize components
        self._setup_logging()
        self._load_models()
        self._setup_kafka()
        self._setup_prometheus()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Configure production logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('anomaly_detector.log'),
                logging.StreamHandler()
            ]
        )
    
    def _load_models(self):
        """Load trained models from MLflow registry."""
        logger.info("Loading trained models...")
        
        try:
            # Load best performing model from previous training
            model_config = self.config['models']
            
            # Initialize models with production parameters
            self.models = {
                'isolation_forest': IsolationForestModel(**model_config['isolation_forest']),
                'random_forest': RandomForestModel(**model_config['random_forest']),
                'lstm': LSTMModel(**model_config['lstm'])
            }
            
            # TODO: Load actual trained models from MLflow
            # For now, we'll use the example models
            logger.warning("Using example models - implement MLflow loading for production")
            
            # Set primary model (could be loaded from model registry)
            self.current_model = self.models['random_forest']  # Best performer from examples
            
            logger.info(f"Models loaded successfully. Primary model: {type(self.current_model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _setup_kafka(self):
        """Setup Kafka consumer and producer."""
        kafka_config = self.config['monitoring']['kafka']
        
        try:
            # Consumer for incoming IT metrics
            self.kafka_consumer = KafkaConsumer(
                kafka_config['topics']['input'],
                bootstrap_servers=kafka_config['bootstrap_servers'],
                group_id=kafka_config['consumer_group'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            # Producer for anomaly predictions and alerts
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_config['bootstrap_servers'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            logger.info("Kafka consumer and producer initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Kafka: {e}")
            raise
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics server."""
        prometheus_config = self.config['monitoring']['prometheus']
        
        try:
            # Start Prometheus metrics server
            start_http_server(prometheus_config['metrics_port'])
            logger.info(f"Prometheus metrics server started on port {prometheus_config['metrics_port']}")
            
        except Exception as e:
            logger.error(f"Failed to setup Prometheus: {e}")
            raise
    
    async def run(self):
        """Main execution loop for real-time anomaly detection."""
        logger.info("🚀 Starting real-time anomaly detection service...")
        
        try:
            for message in self.kafka_consumer:
                await self._process_message(message)
                
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise
        finally:
            self._cleanup()
    
    async def _process_message(self, message):
        """Process a single Kafka message."""
        start_time = time.time()
        
        try:
            # Parse message
            metrics_data = message.value
            timestamp = metrics_data.get('timestamp', datetime.utcnow().isoformat())
            
            # Convert to model input format
            features = self._extract_features(metrics_data)
            
            # Detect anomaly
            is_anomaly, confidence = self._detect_anomaly(features)
            
            # Create result
            result = {
                'timestamp': timestamp,
                'host': metrics_data.get('host', 'unknown'),
                'is_anomaly': bool(is_anomaly),
                'confidence': float(confidence),
                'features': features.tolist(),
                'model_used': type(self.current_model).__name__
            }
            
            # Send prediction to output topic
            await self._send_prediction(result)
            
            # Handle anomalies
            if is_anomaly:
                await self._handle_anomaly(result, metrics_data)
            
            # Update metrics
            MESSAGES_PROCESSED.inc()
            if is_anomaly:
                ANOMALIES_DETECTED.inc()
            
            processing_time = time.time() - start_time
            PROCESSING_TIME.observe(processing_time)
            
            logger.debug(f"Processed message in {processing_time:.3f}s - Anomaly: {is_anomaly}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _extract_features(self, metrics_data: Dict) -> np.ndarray:
        """
        Extract features from raw metrics data.
        
        Expected input format:
        {
            "timestamp": "2025-06-20T19:30:00Z",
            "host": "server-01",
            "cpu_usage": 75.2,
            "ram_usage": 82.1,
            "disk_io": 1024,
            "network_io": 2048,
            "response_time": 150,
            "error_rate": 0.02,
            "active_connections": 250,
            "queue_length": 15
        }
        """
        # Extract core metrics (same as training data)
        features = [
            metrics_data.get('cpu_usage', 0),
            metrics_data.get('ram_usage', 0),
            metrics_data.get('disk_io', 0),
            metrics_data.get('network_io', 0),
            metrics_data.get('response_time', 0),
            metrics_data.get('error_rate', 0),
            metrics_data.get('active_connections', 0),
            metrics_data.get('queue_length', 0)
        ]
        
        # Add derived features (same as training)
        cpu_ram_ratio = features[0] / (features[1] + 1)
        io_total = features[2] + features[3]
        load_score = (features[0] + features[1]) / 2
        
        features.extend([cpu_ram_ratio, io_total, load_score])
        
        return np.array(features).reshape(1, -1)
    
    def _detect_anomaly(self, features: np.ndarray) -> tuple:
        """
        Detect anomaly using the current model.
        
        Returns:
            tuple: (is_anomaly, confidence_score)
        """
        try:
            # Get prediction
            prediction = self.current_model.predict(features)[0]
            
            # Get confidence score if available
            confidence = 0.5  # Default
            if hasattr(self.current_model, 'predict_proba'):
                probabilities = self.current_model.predict_proba(features)[0]
                confidence = max(probabilities)
            elif hasattr(self.current_model, 'decision_function'):
                scores = self.current_model.decision_function(features)
                confidence = abs(scores[0])
            
            return prediction == 1, confidence
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0
    
    async def _send_prediction(self, result: Dict):
        """Send prediction result to Kafka output topic."""
        try:
            topic = self.config['monitoring']['kafka']['topics']['output']
            self.kafka_producer.send(topic, value=result)
            
        except Exception as e:
            logger.error(f"Error sending prediction: {e}")
    
    async def _handle_anomaly(self, result: Dict, original_data: Dict):
        """Handle detected anomaly with alerts and notifications."""
        try:
            # Create alert
            alert = {
                'timestamp': result['timestamp'],
                'severity': self._calculate_severity(result['confidence']),
                'host': result['host'],
                'anomaly_type': 'IT_INFRASTRUCTURE',
                'confidence': result['confidence'],
                'original_metrics': original_data,
                'model_prediction': result
            }
            
            # Send to alerts topic
            alerts_topic = self.config['monitoring']['kafka']['topics']['alerts']
            self.kafka_producer.send(alerts_topic, value=alert)
            
            # Log alert
            logger.warning(f"🚨 ANOMALY ALERT: {alert}")
            
            # Send to external systems (Slack, email, etc.)
            await self._send_external_alerts(alert)
            
        except Exception as e:
            logger.error(f"Error handling anomaly: {e}")
    
    def _calculate_severity(self, confidence: float) -> str:
        """Calculate alert severity based on confidence."""
        if confidence > 0.9:
            return "CRITICAL"
        elif confidence > 0.7:
            return "HIGH"
        elif confidence > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def _send_external_alerts(self, alert: Dict):
        """Send alerts to external systems (Slack, email, etc.)."""
        alert_config = self.config.get('alerts', {})
        
        # Example: Send to Slack
        slack_webhook = alert_config.get('channels', {}).get('slack_webhook')
        if slack_webhook:
            try:
                slack_message = {
                    "text": f"🚨 Anomaly Alert - {alert['severity']}",
                    "attachments": [{
                        "color": "danger" if alert['severity'] in ['CRITICAL', 'HIGH'] else "warning",
                        "fields": [
                            {"title": "Host", "value": alert['host'], "short": True},
                            {"title": "Confidence", "value": f"{alert['confidence']:.2%}", "short": True},
                            {"title": "Timestamp", "value": alert['timestamp'], "short": False}
                        ]
                    }]
                }
                
                # Would send to actual Slack webhook in production
                logger.info(f"Would send Slack alert: {slack_message}")
                
            except Exception as e:
                logger.error(f"Error sending Slack alert: {e}")
    
    def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.kafka_consumer:
                self.kafka_consumer.close()
            if self.kafka_producer:
                self.kafka_producer.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point for the real-time anomaly detection service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Anomaly Detection Service')
    parser.add_argument('--config', default='config/production.json', 
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create and run detector
    detector = RealTimeAnomalyDetector(args.config)
    
    try:
        asyncio.run(detector.run())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())