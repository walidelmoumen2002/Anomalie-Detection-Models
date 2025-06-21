# Machine Learning Models Documentation

This document provides comprehensive technical documentation about the three anomaly detection models implemented in this MLOps pipeline.

## 📋 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Isolation Forest](#isolation-forest)
- [Random Forest](#random-forest)
- [LSTM Neural Network](#lstm-neural-network)
- [Model Comparison](#model-comparison)
- [Performance Metrics](#performance-metrics)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

Our anomaly detection pipeline implements three complementary machine learning approaches:

| Model | Type | Use Case | Training Data | Strengths |
|-------|------|----------|---------------|-----------|
| **Isolation Forest** | Unsupervised | Real-time monitoring | Unlabeled data | Fast, scalable, no labels needed |
| **Random Forest** | Supervised | Known anomaly classification | Labeled historical data | High accuracy, interpretable |
| **LSTM** | Supervised | Time-series anomalies | Sequential labeled data | Temporal patterns, complex relationships |

## 🏗️ Model Architecture

### System Architecture Flow

```
Raw Metrics → Feature Engineering → Model Training → Anomaly Detection → Alerts
     ↓              ↓                    ↓               ↓              ↓
  CPU, Memory,   Enhanced Features   ML Models     Binary Prediction  Grafana
  Disk, Network  (ratios, MA, etc.)  (IF/RF/LSTM)  (0=Normal, 1=Anomaly) Dashboard
```

### Feature Engineering Pipeline

```python
Base Features (5):
├── cpu_usage (%)           # CPU utilization percentage
├── memory_usage (%)        # RAM usage percentage  
├── disk_usage (%)          # Disk usage percentage
├── network_io (KB/s)       # Network I/O throughput
└── response_time (ms)      # System response time

Derived Features (4):
├── cpu_ram_ratio           # CPU/Memory ratio
├── io_total                # Combined I/O load
├── load_score              # Overall system load
└── resource_pressure       # Resource contention indicator

Temporal Features (15):     # Generated for time-series analysis
├── feature_ma5             # 5-period moving average
├── feature_std5            # 5-period standard deviation
└── feature_diff_ma         # Difference from moving average
```

## 🌲 Isolation Forest

### Algorithm Overview

Isolation Forest is an unsupervised anomaly detection algorithm that isolates observations by randomly selecting features and split values. Anomalies are data points that are few and different, requiring fewer random splits to isolate them.

### Key Principles

```
Normal Points:     Many similar → Hard to isolate → Deep in tree
Anomalous Points:  Few & different → Easy to isolate → Shallow in tree

Anomaly Score = 2^(-E(h(x))/c(n))
Where:
- E(h(x)) = Average path length of point x
- c(n) = Average path length of unsuccessful search in BST with n points
```

### Implementation Details

```python
class IsolationForestModel(BaseAnomalyModel):
    def __init__(self, contamination=0.05, n_estimators=100, 
                 max_samples="auto", max_features=1.0, 
                 bootstrap=False, random_state=42):
```

**Parameters:**

- **contamination** (0.0-0.5): Expected proportion of anomalies
  - `0.05`: 5% anomalies (standard IT environment)
  - `0.1`: 10% anomalies (unstable environment)
  - `0.02`: 2% anomalies (stable production)

- **n_estimators** (int): Number of trees in the forest
  - `100`: Good balance of accuracy/speed (recommended)
  - `200+`: Higher accuracy, slower training
  - `50`: Faster training, lower accuracy

- **max_samples** (str/int): Samples to draw for each tree
  - `"auto"`: min(256, n_samples) - recommended
  - `256`: Fixed sample size
  - `512`: Larger sample for better accuracy

### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Training Speed | **Fast** | O(n log n) complexity |
| Prediction Speed | **Very Fast** | Tree traversal only |
| Memory Usage | **Low** | Stores only tree structures |
| Scalability | **Excellent** | Handles millions of samples |
| Interpretability | **Medium** | Feature importance available |

### Use Cases

✅ **Ideal For:**
- Real-time monitoring without labeled data
- Initial anomaly detection in new systems
- High-volume data streams
- Cloud infrastructure monitoring
- IoT sensor data analysis

❌ **Not Suitable For:**
- Applications requiring high precision/recall
- Complex temporal patterns
- When labeled data is abundant
- Regulatory compliance requiring explanations

### Example Usage

```python
from core.isolation_forest import IsolationForestModel

# Standard configuration
model = IsolationForestModel(contamination=0.05, n_estimators=100)

# Generate training data
data = model.generate_synthetic_data(n_samples=10000, anomaly_rate=0.05)
features = model.prepare_features(data)
X = features.values
y = data['is_anomaly'].values  # For evaluation only

# Train model (unsupervised)
metrics = model.fit(X)

# Make predictions
predictions = model.predict(new_data)
anomaly_mask = predictions == 1  # Anomalies marked as 1
```

## 🌳 Random Forest

### Algorithm Overview

Random Forest is a supervised ensemble learning method that combines multiple decision trees. Each tree is trained on a bootstrap sample of the data using a random subset of features, making the model robust and highly accurate.

### Key Principles

```
Bootstrap Aggregating (Bagging):
1. Create N bootstrap samples from training data
2. Train decision tree on each sample using random feature subset
3. Combine predictions via majority voting (classification)

Ensemble Benefits:
- Reduces overfitting compared to single trees
- Provides feature importance rankings
- Handles missing values and mixed data types
- Robust to outliers and noise
```

### Implementation Details

```python
class RandomForestModel(BaseAnomalyModel):
    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", bootstrap=True,
                 class_weight="balanced", random_state=42):
```

**Parameters:**

- **n_estimators** (int): Number of trees
  - `100`: Standard choice (recommended)
  - `200+`: Better accuracy, slower training
  - `50`: Faster training, acceptable accuracy

- **max_depth** (int/None): Maximum tree depth
  - `None`: No limit (may overfit)
  - `10-20`: Good balance for IT data
  - `5`: Simple model, prevents overfitting

- **max_features** (str/float): Features per split
  - `"sqrt"`: √(total_features) - recommended
  - `"log2"`: log₂(total_features)
  - `0.5`: Half of all features

- **class_weight** (str/dict): Handle imbalanced classes
  - `"balanced"`: Automatic class balancing
  - `{0: 1, 1: 10}`: Custom weights (normal:anomaly)
  - `None`: No balancing

### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Training Speed | **Medium** | O(n log n × trees × features) |
| Prediction Speed | **Fast** | Parallel tree traversal |
| Memory Usage | **Medium** | Stores all trees |
| Accuracy | **High** | Excellent with labeled data |
| Interpretability | **High** | Feature importance + tree visualization |

### Feature Importance Analysis

Random Forest provides detailed feature importance scores:

```python
# Get feature importance
importance_df = model.get_feature_importance()
print(importance_df.head())

# Output example:
#           feature  importance
# 0      cpu_usage      0.245
# 1   memory_usage      0.198
# 2  response_time      0.156
# 3         io_total     0.142
# 4     load_score      0.089
```

### Use Cases

✅ **Ideal For:**
- Historical anomaly classification
- When labeled training data is available
- High-accuracy requirements
- Feature importance analysis
- Regulatory environments requiring explainability

❌ **Not Suitable For:**
- Real-time unsupervised detection
- When no labeled data exists
- Complex temporal sequence patterns
- Simple linear relationships

### Example Usage

```python
from core.random_forest import RandomForestModel

# High-accuracy configuration
model = RandomForestModel(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced"
)

# Generate labeled training data
data = model.generate_synthetic_data(n_samples=10000, anomaly_rate=0.1)
features = model.prepare_features(data)
X = features.values
y = data['is_anomaly'].values

# Split data
X_train, X_test, y_train, y_test = model.split_data(X, y)

# Train model (supervised)
metrics = model.fit(X_train, y_train)

# Evaluate
test_metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {test_metrics['accuracy']:.3f}")
print(f"F1-Score: {test_metrics['f1_score']:.3f}")

# Feature importance
importance = model.get_feature_importance()
```

## 🧠 LSTM Neural Network

### Algorithm Overview

Long Short-Term Memory (LSTM) is a recurrent neural network architecture designed to learn from sequential data. It uses memory cells and gates to capture long-term dependencies and complex temporal patterns in time-series data.

### Key Principles

```
LSTM Cell Components:
1. Forget Gate: Decides what information to discard
2. Input Gate: Determines what new information to store
3. Output Gate: Controls what parts of cell state to output
4. Cell State: Long-term memory flowing through the network

Memory Flow:
Input Sequence → LSTM Layers → Dense Layers → Binary Classification
[t-n...t-1, t] → Hidden States → Features → [Normal/Anomaly]
```

### Architecture Design

```python
Model Architecture:
Input(sequence_length, n_features)
    ↓
LSTM(units=50, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(units=25, return_sequences=False)
    ↓
Dropout(0.2)
    ↓
Dense(16, activation='relu')
    ↓
Dense(1, activation='sigmoid')  # Binary classification
```

### Implementation Details

```python
class LSTMModel(BaseAnomalyModel):
    def __init__(self, sequence_length=10, lstm_units=[50, 25],
                 dropout_rate=0.2, learning_rate=0.001,
                 epochs=50, batch_size=32, patience=10):
```

**Parameters:**

- **sequence_length** (int): Time steps to look back
  - `10`: Standard window (recommended)
  - `20`: Longer context, more memory needed
  - `5`: Shorter context, faster training

- **lstm_units** (list): Number of units per LSTM layer
  - `[50, 25]`: Two-layer decreasing units
  - `[100]`: Single large layer
  - `[64, 32, 16]`: Three-layer pyramid

- **dropout_rate** (float): Dropout for regularization
  - `0.2`: 20% dropout (recommended)
  - `0.3`: More regularization
  - `0.1`: Less regularization

- **learning_rate** (float): Optimizer learning rate
  - `0.001`: Standard Adam rate
  - `0.01`: Faster learning (may overshoot)
  - `0.0001`: Slower, more stable learning

### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Training Speed | **Slow** | GPU recommended for large datasets |
| Prediction Speed | **Medium** | Sequential computation required |
| Memory Usage | **High** | Stores weights + sequences |
| Temporal Accuracy | **Excellent** | Best for time-series patterns |
| Interpretability | **Low** | Black-box neural network |

### Sequence Preparation

LSTM requires special data preparation for sequential input:

```python
def create_sequences(data, sequence_length=10):
    """
    Transform time-series data into sequences for LSTM.
    
    Input:  [t1, t2, t3, t4, t5, t6, ...]
    Output: [[t1,t2,t3,t4,t5], [t2,t3,t4,t5,t6], ...]
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)
```

### Use Cases

✅ **Ideal For:**
- Time-series anomaly detection
- Complex temporal patterns
- Seasonal behavior analysis
- Sequential event correlation
- Multi-variate time-series analysis

❌ **Not Suitable For:**
- Small datasets (< 1000 samples)
- Static/non-temporal features
- Real-time applications (high latency)
- Interpretability requirements

### Example Usage

```python
from core.lstm_model import LSTMModel

# Time-series configuration
model = LSTMModel(
    sequence_length=20,
    lstm_units=[100, 50],
    dropout_rate=0.3,
    epochs=100
)

# Generate time-series data
data = model.generate_synthetic_data(n_samples=50000, anomaly_rate=0.05)
features = model.prepare_features(data, include_temporal=True)

# Create sequences for LSTM
X_seq, y_seq = model.create_sequences(features.values, sequence_length=20)

# Split data
X_train, X_test, y_train, y_test = model.split_data(X_seq, y_seq)

# Train model (requires GPU for reasonable speed)
metrics = model.fit(X_train, y_train, epochs=100, batch_size=64)

# Evaluate
test_metrics = model.evaluate(X_test, y_test)
print(f"AUC Score: {test_metrics['auc_score']:.3f}")
```

## 📊 Model Comparison

### Performance Matrix

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Speed |
|-------|----------|-----------|--------|----------|---------------|-----------------|
| **Isolation Forest** | 0.85-0.90 | 0.80-0.85 | 0.75-0.82 | 0.77-0.84 | **Fast** | **Very Fast** |
| **Random Forest** | 0.92-0.96 | 0.90-0.94 | 0.88-0.92 | 0.89-0.93 | Medium | Fast |
| **LSTM** | 0.94-0.98 | 0.92-0.96 | 0.90-0.95 | 0.91-0.95 | **Slow** | Medium |

### Resource Requirements

| Model | Memory Usage | CPU Intensive | GPU Required | Disk Storage |
|-------|--------------|---------------|--------------|--------------|
| **Isolation Forest** | **Low** (50-100 MB) | No | No | **Small** (1-5 MB) |
| **Random Forest** | Medium (100-500 MB) | Medium | No | Medium (10-50 MB) |
| **LSTM** | **High** (500+ MB) | **High** | Recommended | Large (50-200 MB) |

### Decision Matrix

Use this decision tree to select the appropriate model:

```
Do you have labeled historical data?
├── NO → Use Isolation Forest
│   ├── Real-time monitoring ✓
│   ├── Unknown anomaly types ✓
│   └── Fast deployment needed ✓
│
└── YES → Do you need high accuracy?
    ├── NO → Use Isolation Forest (faster)
    └── YES → Is temporal pattern important?
        ├── NO → Use Random Forest
        │   ├── Feature interpretability ✓
        │   ├── High accuracy ✓
        │   └── Regulatory compliance ✓
        │
        └── YES → Use LSTM
            ├── Complex time patterns ✓
            ├── Sequential correlations ✓
            └── Highest accuracy ✓
```

## 📈 Performance Metrics

### Evaluation Metrics Explained

**Accuracy**: Overall correctness
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Good for: Balanced datasets
Problem: Misleading for imbalanced data (e.g., 1% anomalies)
```

**Precision**: Anomaly prediction accuracy
```
Precision = TP / (TP + FP)
Meaning: Of all predicted anomalies, how many were actually anomalous?
Important for: Reducing false alarms
```

**Recall (Sensitivity)**: Anomaly detection completeness
```
Recall = TP / (TP + FN)
Meaning: Of all actual anomalies, how many did we detect?
Important for: Not missing critical incidents
```

**F1-Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Best for: Balancing precision and recall
```

**AUC-ROC**: Area under ROC curve
```
ROC plots: True Positive Rate vs False Positive Rate
AUC = 1.0: Perfect classifier
AUC = 0.5: Random classifier
Best for: Ranking and threshold selection
```

### Business Impact Metrics

**Mean Time to Detection (MTTD)**
- Isolation Forest: **< 1 minute**
- Random Forest: **< 30 seconds**  
- LSTM: **2-5 minutes**

**False Positive Rate Impact**
- High FPR → Alert fatigue → Ignored real incidents
- Target: < 5% FPR for production systems
- Random Forest typically achieves lowest FPR

**Critical Incident Detection**
- LSTM: Best for complex, evolving attacks
- Random Forest: Best for known attack patterns
- Isolation Forest: Best for unknown anomalies

## 🎯 Best Practices

### Model Selection Guidelines

1. **Start with Isolation Forest**
   - Deploy quickly for baseline monitoring
   - Requires no labeled data
   - Provides immediate value

2. **Add Random Forest when data available**
   - Use historical incident data for training
   - Focus on high-confidence predictions
   - Leverage feature importance for root cause analysis

3. **Implement LSTM for advanced detection**
   - Requires significant data (months of history)
   - Best for detecting subtle, evolving patterns
   - Consider computational costs

### Hyperparameter Tuning

**Isolation Forest Tuning:**
```python
# Grid search example
param_grid = {
    'contamination': [0.02, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_samples': [256, 512, 'auto']
}
```

**Random Forest Tuning:**
```python
# Grid search example
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', {0: 1, 1: 10}]
}
```

**LSTM Tuning:**
```python
# Manual tuning (expensive to grid search)
experiments = [
    {'sequence_length': 10, 'lstm_units': [50, 25]},
    {'sequence_length': 20, 'lstm_units': [100, 50]},
    {'sequence_length': 15, 'lstm_units': [75, 35]}
]
```

### Data Quality Requirements

**Minimum Data Requirements:**
- Isolation Forest: 1,000+ samples
- Random Forest: 5,000+ samples with labels
- LSTM: 10,000+ sequential samples with labels

**Data Quality Checklist:**
- ✅ No missing values (< 5%)
- ✅ Consistent time intervals
- ✅ Representative anomaly examples
- ✅ Balanced normal/anomaly ratio (adjust class_weight)
- ✅ Feature scaling applied
- ✅ Outliers handled appropriately

### Production Deployment

**Model Refresh Strategy:**
```python
# Isolation Forest: Daily retraining
# Random Forest: Weekly retraining with new labels
# LSTM: Monthly retraining with expanded sequences

def retrain_schedule():
    return {
        'isolation_forest': {'frequency': 'daily', 'window': '7d'},
        'random_forest': {'frequency': 'weekly', 'window': '30d'},
        'lstm': {'frequency': 'monthly', 'window': '90d'}
    }
```

**A/B Testing Framework:**
```python
# Deploy new model alongside existing
# Route 10% traffic to new model
# Compare performance metrics
# Gradually increase traffic if improvement confirmed
```

## 🐛 Troubleshooting

### Common Issues

**Problem**: Isolation Forest detecting too many false positives
```python
# Solution: Adjust contamination parameter
model = IsolationForestModel(contamination=0.02)  # Reduce from 0.05
```

**Problem**: Random Forest overfitting
```python
# Solution: Add regularization
model = RandomForestModel(
    max_depth=10,           # Limit tree depth
    min_samples_split=10,   # Require more samples for splits
    min_samples_leaf=5      # Require more samples in leaves
)
```

**Problem**: LSTM not converging
```python
# Solution: Adjust learning rate and add callbacks
model = LSTMModel(
    learning_rate=0.0001,   # Reduce learning rate
    patience=20,            # Increase early stopping patience
    epochs=200              # Allow more training epochs
)
```

**Problem**: High memory usage
```python
# Solutions:
# 1. Reduce batch size
model = LSTMModel(batch_size=16)  # Reduce from 32

# 2. Use gradient checkpointing
model.compile(optimizer='adam', loss='binary_crossentropy', 
              experimental_gradient_checkpointing=True)

# 3. Process data in chunks
def process_in_chunks(data, chunk_size=10000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]
```

### Model Debugging

**Check model training progress:**
```python
# Plot training/validation loss
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.show()
```

**Validate predictions:**
```python
# Check prediction distribution
predictions = model.predict(X_test)
print(f"Anomaly rate in predictions: {predictions.mean():.3f}")
print(f"Expected anomaly rate: {contamination:.3f}")

# Should be approximately equal for Isolation Forest
```

### Performance Optimization

**Isolation Forest Optimization:**
```python
# Use max_samples for large datasets
model = IsolationForestModel(max_samples=1000)  # Cap sample size

# Parallel processing
import joblib
joblib.Parallel(n_jobs=-1)  # Use all CPU cores
```

**Random Forest Optimization:**
```python
# Enable parallel processing
model = RandomForestModel(n_jobs=-1)  # Use all CPU cores

# Reduce memory usage
model = RandomForestModel(max_samples=0.8)  # Use 80% of data per tree
```

**LSTM Optimization:**
```python
# Use mixed precision training (if GPU available)
import tensorflow as tf
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

---

This documentation provides comprehensive technical details for implementing, tuning, and troubleshooting the anomaly detection models. For additional support, refer to the main README.md or create an issue in the project repository.