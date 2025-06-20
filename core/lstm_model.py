"""
LSTM Model for IT Anomaly Detection
===================================

Modèle de détection d'anomalies basé sur LSTM (Long Short-Term Memory).
Optimal pour l'analyse de séquences temporelles et la détection de patterns complexes.

Avantages du LSTM:
- Capture les dépendances temporelles à long terme
- Détecte les patterns séquentiels complexes
- Excellent pour les séries temporelles
- Peut prédire les anomalies futures
- Gère la mémoire sélective des événements passés

Cas d'usage typiques:
- Analyse de tendances temporelles
- Prédiction d'anomalies futures
- Détection de patterns séquentiels
- Surveillance continue en temps réel

Author: MLOps Team
Date: June 2025
Version: 1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import mlflow
import mlflow.tensorflow
import logging
import matplotlib.pyplot as plt
from base_model import BaseAnomalyModel

# Configuration TensorFlow pour éviter les warnings
tf.get_logger().setLevel('ERROR')
logger = logging.getLogger(__name__)

class LSTMModel(BaseAnomalyModel):
    """
    Modèle de détection d'anomalies utilisant LSTM (Deep Learning).
    
    LSTM est optimal pour:
    - Analyse de séquences temporelles
    - Détection de patterns complexes dans le temps
    - Prédiction d'anomalies futures
    - Capture des dépendances à long terme
    - Gestion de la mémoire des événements passés
    
    Attributes:
        sequence_length (int): Longueur des séquences d'entrée
        lstm_units (list): Nombre d'unités LSTM par couche
        dropout_rate (float): Taux de dropout pour régularisation
        learning_rate (float): Taux d'apprentissage
        batch_size (int): Taille des batches d'entraînement
        epochs (int): Nombre d'époques d'entraînement
        validation_split (float): Proportion pour validation
        early_stopping_patience (int): Patience pour early stopping
    """
    
    def __init__(self, sequence_length: int = 10, lstm_units: list = [50, 50],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 batch_size: int = 32, epochs: int = 50, validation_split: float = 0.2,
                 early_stopping_patience: int = 10, random_state: int = 42):
        """
        Initialise le modèle LSTM.
        
        Args:
            sequence_length (int): Longueur des séquences temporelles
                                 - 10: séquences courtes (recommandé pour IT)
                                 - 20-30: séquences moyennes
                                 - 50+: séquences longues (plus de contexte)
            lstm_units (list): Unités LSTM par couche
                             - [50]: une couche simple
                             - [50, 50]: deux couches (recommandé)
                             - [100, 50, 25]: architecture profonde
            dropout_rate (float): Taux de dropout (0.0 à 0.5)
                                - 0.2: régularisation modérée (recommandé)
                                - 0.3-0.5: régularisation forte
                                - 0.0-0.1: régularisation faible
            learning_rate (float): Taux d'apprentissage
                                 - 0.001: standard (recommandé)
                                 - 0.01: apprentissage rapide
                                 - 0.0001: apprentissage lent mais stable
            batch_size (int): Taille des batches
                            - 32: compromis mémoire/performance (recommandé)
                            - 64-128: plus rapide si assez de mémoire
                            - 16: pour limitations mémoire
            epochs (int): Nombre maximum d'époques
                        - 50: standard (recommandé)
                        - 100+: pour modèles complexes
                        - 20-30: entraînement rapide
            validation_split (float): Proportion pour validation (0.0 à 0.5)
                                     - 0.2: 20% pour validation (recommandé)
            early_stopping_patience (int): Patience avant arrêt prématuré
                                          - 10: patience modérée (recommandé)
                                          - 5: arrêt rapide
                                          - 20: patience élevée
            random_state (int): Graine aléatoire pour reproductibilité
        
        Example:
            # Configuration standard
            model = LSTMModel(sequence_length=10, lstm_units=[50, 50])
            
            # Configuration légère
            model = LSTMModel(sequence_length=5, lstm_units=[25], dropout_rate=0.1)
            
            # Configuration avancée
            model = LSTMModel(sequence_length=20, lstm_units=[100, 50, 25], 
                            dropout_rate=0.3, epochs=100)
        """
        super().__init__(model_name="LSTM", model_type="supervised")
        
        # Paramètres du modèle
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units if isinstance(lstm_units, list) else [lstm_units]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Callbacks et historique
        self.training_history = None
        self.callbacks = None
        
        # Configuration TensorFlow
        tf.random.set_seed(random_state)
        
        # Validation des paramètres
        self._validate_parameters()
        
        logger.info(f"LSTM initialisé: sequence_length={sequence_length}, "
                   f"lstm_units={lstm_units}, dropout_rate={dropout_rate}")
    
    def _validate_parameters(self):
        """Valide les paramètres du modèle."""
        if self.sequence_length < 2:
            raise ValueError("sequence_length doit être >= 2")
        
        if not all(units > 0 for units in self.lstm_units):
            raise ValueError("Toutes les unités LSTM doivent être positives")
        
        if not 0.0 <= self.dropout_rate <= 0.5:
            raise ValueError("dropout_rate doit être entre 0.0 et 0.5")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate doit être positif")
        
        if self.batch_size < 1:
            raise ValueError("batch_size doit être positif")
    
    def _create_model(self, input_shape=None, **kwargs):
        """
        Crée l'architecture LSTM.
        
        Args:
            input_shape (tuple): Forme des données d'entrée (sequence_length, n_features)
            **kwargs: Paramètres additionnels
        
        Returns:
            tf.keras.Model: Modèle LSTM configuré
        """
        if input_shape is None:
            # Forme par défaut (sera mise à jour lors de l'entraînement)
            input_shape = (self.sequence_length, 10)
        
        logger.info(f"Création architecture LSTM: input_shape={input_shape}")
        
        # Construction du modèle séquentiel
        model = Sequential(name="LSTM_AnomalyDetector")
        
        # Couche d'entrée
        model.add(Input(shape=input_shape, name="input_layer"))
        
        # Couches LSTM
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1  # Retourne séquences sauf dernière couche
            
            model.add(LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate * 0.5,  # Dropout récurrent plus faible
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),  # Régularisation L1+L2
                name=f"lstm_layer_{i+1}"
            ))
            
            # Dropout supplémentaire entre les couches LSTM
            if i < len(self.lstm_units) - 1:
                model.add(Dropout(self.dropout_rate, name=f"dropout_{i+1}"))
        
        # Couches denses finales
        model.add(Dense(units=25, activation='relu', name="dense_1",
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(self.dropout_rate, name="final_dropout"))
        
        # Couche de sortie (classification binaire)
        model.add(Dense(units=1, activation='sigmoid', name="output_layer"))
        
        # Compilation du modèle
        optimizer = Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Affichage de l'architecture
        model.summary()
        
        return model
    
    def _prepare_sequences(self, X, y=None):
        """
        Prépare les données en séquences pour LSTM.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels optionnels (n_samples,)
        
        Returns:
            tuple: (X_sequences, y_sequences) ou X_sequences si y=None
        """
        logger.info(f"Préparation des séquences: sequence_length={self.sequence_length}")
        
        if len(X) < self.sequence_length:
            raise ValueError(f"Données insuffisantes: {len(X)} < {self.sequence_length}")
        
        X_sequences = []
        y_sequences = [] if y is not None else None
        
        # Création des séquences glissantes
        for i in range(self.sequence_length, len(X)):
            # Séquence des features
            sequence = X[i-self.sequence_length:i]
            X_sequences.append(sequence)
            
            # Label correspondant à la fin de la séquence
            if y is not None:
                y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        
        if y is not None:
            y_sequences = np.array(y_sequences)
            logger.info(f"Séquences créées: {X_sequences.shape}, labels: {y_sequences.shape}")
            return X_sequences, y_sequences
        else:
            logger.info(f"Séquences créées: {X_sequences.shape}")
            return X_sequences
    
    def _setup_callbacks(self, model_save_path="best_lstm_model.h5"):
        """
        Configure les callbacks pour l'entraînement.
        
        Args:
            model_save_path (str): Chemin pour sauvegarder le meilleur modèle
        
        Returns:
            list: Liste des callbacks configurés
        """
        callbacks = []
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # Réduction du learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.early_stopping_patience // 2,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # Sauvegarde du meilleur modèle
        checkpoint = ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        )
        callbacks.append(checkpoint)
        
        self.callbacks = callbacks
        return callbacks
    
    def _fit_model(self, X_train, y_train):
        """
        Entraîne le modèle LSTM.
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement (requis pour LSTM supervisé)
        
        Returns:
            tf.keras.Model: Modèle entraîné
        """
        if y_train is None:
            raise ValueError("LSTM nécessite des labels d'entraînement (supervisé)")
        
        logger.info(f"Préparation des séquences pour LSTM...")
        
        # Préparation des séquences
        X_sequences, y_sequences = self._prepare_sequences(X_train, y_train)
        
        # Création du modèle avec la bonne forme d'entrée
        input_shape = (X_sequences.shape[1], X_sequences.shape[2])
        self.model = self._create_model(input_shape=input_shape)
        
        # Configuration des callbacks
        callbacks = self._setup_callbacks()
        
        logger.info(f"Entraînement LSTM: {X_sequences.shape[0]} séquences, "
                   f"{X_sequences.shape[1]} pas de temps, {X_sequences.shape[2]} features")
        
        # Entraînement avec validation
        history = self.model.fit(
            X_sequences, y_sequences,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        self.training_history = history
        
        # Log des métriques finales
        final_metrics = {
            'final_loss': history.history['loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'epochs_trained': len(history.history['loss'])
        }
        
        logger.info(f"Entraînement terminé après {final_metrics['epochs_trained']} époques")
        logger.info(f"Accuracy finale: {final_metrics['final_accuracy']:.4f}")
        
        return self.model
    
    def _predict_anomalies(self, X):
        """
        Prédit les anomalies avec LSTM.
        
        Args:
            X: Features normalisées
        
        Returns:
            np.array: Prédictions binaires (1=anomalie, 0=normal)
        """
        # Préparation des séquences
        X_sequences = self._prepare_sequences(X)
        
        # Prédictions probabilistes
        probabilities = self.model.predict(X_sequences, verbose=0)
        
        # Conversion en prédictions binaires
        predictions = (probabilities.flatten() > 0.5).astype(int)
        
        return predictions
    
    def predict_probabilities(self, X):
        """
        Prédit les probabilités d'anomalies.
        
        Args:
            X: Features pour la prédiction
        
        Returns:
            np.array: Probabilités d'anomalies (0.0 à 1.0)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant les prédictions")
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self._prepare_sequences(X_scaled)
        
        probabilities = self.model.predict(X_sequences, verbose=0)
        
        return probabilities.flatten()
    
    def predict_sequences(self, X, return_sequences=False):
        """
        Prédictions avec informations sur les séquences.
        
        Args:
            X: Features pour la prédiction
            return_sequences (bool): Retourner les séquences utilisées
        
        Returns:
            dict: Prédictions avec métadonnées
        """
        X_scaled = self.scaler.transform(X)
        X_sequences = self._prepare_sequences(X_scaled)
        
        probabilities = self.model.predict(X_sequences, verbose=0).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        # Calcul des indices correspondants dans les données originales
        original_indices = np.arange(self.sequence_length, len(X))
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'original_indices': original_indices,
            'n_sequences': len(X_sequences),
            'sequence_length': self.sequence_length
        }
        
        if return_sequences:
            results['sequences'] = X_sequences
        
        return results
    
    def detect_temporal_anomalies(self, X, window_size=5, threshold=0.7):
        """
        Détecte les anomalies avec analyse de fenêtres temporelles.
        
        Args:
            X: Features temporelles
            window_size (int): Taille de la fenêtre d'analyse
            threshold (float): Seuil de probabilité pour anomalies
        
        Returns:
            dict: Détections avec analyse temporelle
        """
        prediction_results = self.predict_sequences(X, return_sequences=True)
        probabilities = prediction_results['probabilities']
        
        # Analyse par fenêtres glissantes
        temporal_analysis = []
        
        for i in range(len(probabilities) - window_size + 1):
            window_probs = probabilities[i:i+window_size]
            
            analysis = {
                'window_start': i,
                'window_end': i + window_size - 1,
                'max_probability': np.max(window_probs),
                'mean_probability': np.mean(window_probs),
                'anomaly_count': np.sum(window_probs > threshold),
                'trend': 'increasing' if window_probs[-1] > window_probs[0] else 'decreasing',
                'volatility': np.std(window_probs)
            }
            
            temporal_analysis.append(analysis)
        
        # Détection d'anomalies persistantes
        persistent_anomalies = []
        consecutive_count = 0
        
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                consecutive_count += 1
            else:
                if consecutive_count >= 3:  # Anomalie persistante si 3+ points consécutifs
                    persistent_anomalies.append({
                        'start_index': i - consecutive_count,
                        'end_index': i - 1,
                        'duration': consecutive_count,
                        'avg_probability': np.mean(probabilities[i-consecutive_count:i])
                    })
                consecutive_count = 0
        
        results = {
            'basic_predictions': prediction_results,
            'temporal_analysis': temporal_analysis,
            'persistent_anomalies': persistent_anomalies,
            'window_size': window_size,
            'threshold': threshold
        }
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Visualise l'historique d'entraînement.
        
        Args:
            save_path (str): Chemin pour sauvegarder le graphique
        """
        if self.training_history is None:
            logger.warning("Aucun historique d'entraînement disponible")
            return
        
        history = self.training_history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Historique d\'entraînement LSTM', fontsize=16)
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Époque')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision')
            axes[1, 0].plot(history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Époque')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall')
            axes[1, 1].plot(history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Époque')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def get_model_summary(self):
        """
        Retourne un résumé détaillé du modèle LSTM.
        
        Returns:
            dict: Informations sur l'architecture et l'entraînement
        """
        if not self.is_fitted:
            return {"error": "Modèle non entraîné"}
        
        # Paramètres du modèle
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        summary = {
            'architecture': {
                'sequence_length': self.sequence_length,
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            },
            'training': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'epochs_configured': self.epochs,
                'validation_split': self.validation_split
            }
        }
        
        # Historique d'entraînement
        if self.training_history:
            history = self.training_history.history
            summary['training_results'] = {
                'epochs_trained': len(history['loss']),
                'final_train_loss': history['loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_accuracy': history['accuracy'][-1],
                'final_val_accuracy': history['val_accuracy'][-1],
                'best_val_loss': min(history['val_loss']),
                'best_val_accuracy': max(history['val_accuracy'])
            }
        
        return summary


def demonstrate_lstm():
    """
    Démonstration complète du modèle LSTM.
    """
    print("🧠 DÉMONSTRATION LSTM")
    print("=" * 50)
    
    # 1. Initialisation
    print("\n1️⃣ Initialisation du modèle")
    model = LSTMModel(
        sequence_length=10,
        lstm_units=[50, 25],
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=20,  # Réduit pour la démo
        random_state=42
    )
    print(f"✅ Modèle initialisé: {model.model_name}")
    
    # 2. Génération des données
    print("\n2️⃣ Génération des données temporelles")
    data = model.generate_synthetic_data(n_samples=2000, anomaly_rate=0.05)
    X = model.prepare_features(data, include_temporal=True)
    y = data['is_anomaly']
    print(f"✅ {len(data)} échantillons temporels")
    
    # 3. Division des données
    X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.2)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Entraînement
    print("\n4️⃣ Entraînement du modèle LSTM")
    training_metrics = model.fit(X_train, y_train)
    print(f"✅ Entraînement terminé")
    
    # 5. Évaluation
    print("\n5️⃣ Évaluation sur séquences")
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"✅ Précision: {eval_metrics['accuracy']:.3f}")
    
    # 6. Prédictions séquentielles
    print("\n6️⃣ Prédictions sur séquences temporelles")
    seq_results = model.predict_sequences(X_test[:100])
    print(f"✅ {seq_results['n_sequences']} séquences analysées")
    print(f"✅ {np.sum(seq_results['predictions'])} anomalies détectées")
    
    # 7. Analyse temporelle
    print("\n7️⃣ Analyse des patterns temporels")
    temporal_results = model.detect_temporal_anomalies(X_test[:200], window_size=5)
    print(f"✅ {len(temporal_results['persistent_anomalies'])} anomalies persistantes")
    
    # 8. Résumé du modèle
    print("\n8️⃣ Résumé du modèle LSTM")
    summary = model.get_model_summary()
    print(f"✅ Paramètres totaux: {summary['architecture']['total_parameters']}")
    print(f"✅ Époques d'entraînement: {summary['training_results']['epochs_trained']}")
    
    # 9. Résumé général
    print("\n9️⃣ Résumé général")
    model.summary()
    
    return model


if __name__ == "__main__":
    # Démonstration complète
    trained_model = demonstrate_lstm()
    
    print("\n🎯 LSTM PRÊT POUR LA PRODUCTION!")
    print("Utilisez ce modèle pour l'analyse de séquences temporelles.")
