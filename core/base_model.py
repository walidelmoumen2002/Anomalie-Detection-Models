"""
Base Anomaly Detection Model Class
==================================

Cette classe de base définit l'interface commune pour tous les modèles de détection d'anomalies.
Elle assure une cohérence dans l'implémentation et facilite l'ajout de nouveaux modèles.

Author: MLOps Team
Date: June 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from abc import ABC, abstractmethod
import logging
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAnomalyModel(ABC):
    """
    Classe de base abstraite pour tous les modèles de détection d'anomalies.
    
    Cette classe définit l'interface standard que tous les modèles doivent implémenter,
    garantissant une cohérence dans l'utilisation et facilitant la maintenance.
    
    Attributes:
        model: Le modèle ML sous-jacent (scikit-learn, tensorflow, etc.)
        scaler: StandardScaler pour la normalisation des features
        is_fitted: Boolean indiquant si le modèle est entraîné
        model_type: Type du modèle (supervisé/non-supervisé)
        feature_names: Noms des features utilisées
        training_metrics: Métriques d'entraînement
    """
    
    def __init__(self, model_name: str, model_type: str = "unsupervised"):
        """
        Initialise le modèle de base.
        
        Args:
            model_name (str): Nom du modèle pour MLflow tracking
            model_type (str): Type de modèle ("supervised" ou "unsupervised")
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = {}
        
        # Configuration MLflow
        mlflow.set_experiment("IT-Anomaly-Detection")
        
        logger.info(f"Initialisation du modèle {model_name} (type: {model_type})")
    
    @abstractmethod
    def _create_model(self, **kwargs):
        """
        Crée l'instance du modèle ML spécifique.
        Doit être implémentée par chaque classe fille.
        
        Args:
            **kwargs: Hyperparamètres spécifiques au modèle
        
        Returns:
            Instance du modèle ML
        """
        pass
    
    @abstractmethod
    def _fit_model(self, X_train, y_train=None):
        """
        Entraîne le modèle avec les données d'entraînement.
        Doit être implémentée par chaque classe fille.
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement (None pour non-supervisé)
        
        Returns:
            Le modèle entraîné
        """
        pass
    
    @abstractmethod
    def _predict_anomalies(self, X):
        """
        Effectue les prédictions d'anomalies.
        Doit être implémentée par chaque classe fille.
        
        Args:
            X: Features pour la prédiction
        
        Returns:
            Prédictions binaires (1=anomalie, 0=normal)
        """
        pass
    
    def generate_synthetic_data(self, n_samples: int = 10000, anomaly_rate: float = 0.05):
        """
        Génère des données synthétiques d'infrastructure IT pour l'entraînement.
        
        Cette méthode crée des données réalistes simulant des métriques système :
        - CPU usage (%)
        - RAM usage (%)  
        - Network I/O (KB/s)
        - Disk I/O (KB/s)
        - Response time (ms)
        
        Args:
            n_samples (int): Nombre d'échantillons à générer
            anomaly_rate (float): Proportion d'anomalies (0.0 à 1.0)
        
        Returns:
            pandas.DataFrame: Dataset avec features et labels
        """
        logger.info(f"Génération de {n_samples} échantillons avec {anomaly_rate*100:.1f}% d'anomalies")
        
        np.random.seed(42)  # Pour la reproductibilité
        
        # Métriques normales (distributions réalistes)
        cpu_usage = np.random.normal(30, 15, n_samples)  # CPU normal ~30%
        ram_usage = np.random.normal(45, 20, n_samples)  # RAM normal ~45%
        network_io = np.random.exponential(1000, n_samples)  # I/O réseau variable
        disk_io = np.random.exponential(500, n_samples)  # I/O disque variable  
        response_time = np.random.gamma(2, 50, n_samples)  # Temps réponse
        
        # Injection d'anomalies
        n_anomalies = int(anomaly_rate * n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        # Types d'anomalies réalistes
        third = len(anomaly_indices) // 3
        
        # Surcharge CPU/RAM (33% des anomalies)
        cpu_anomalies = anomaly_indices[:third]
        cpu_usage[cpu_anomalies] = np.random.normal(90, 5, len(cpu_anomalies))
        ram_usage[cpu_anomalies] = np.random.normal(95, 3, len(cpu_anomalies))
        
        # Saturation réseau (33% des anomalies)
        network_anomalies = anomaly_indices[third:2*third]
        network_io[network_anomalies] = np.random.normal(10000, 2000, len(network_anomalies))
        
        # Latence élevée (34% des anomalies)
        latency_anomalies = anomaly_indices[2*third:]
        response_time[latency_anomalies] = np.random.normal(5000, 1000, len(latency_anomalies))
        
        # Construction du DataFrame
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
            'cpu_usage': np.clip(cpu_usage, 0, 100),
            'ram_usage': np.clip(ram_usage, 0, 100), 
            'network_io': np.abs(network_io),
            'disk_io': np.abs(disk_io),
            'response_time': np.abs(response_time),
            'is_anomaly': 0
        })
        
        # Marquer les anomalies
        data.loc[anomaly_indices, 'is_anomaly'] = 1
        
        logger.info(f"Dataset créé: {len(data)} échantillons, {data['is_anomaly'].sum()} anomalies")
        return data
    
    def prepare_features(self, data: pd.DataFrame, include_temporal: bool = True):
        """
        Prépare et enrichit les features pour l'entraînement ML.
        
        Transformations appliquées :
        - Features de base (métriques brutes)
        - Features dérivées (ratios, combinaisons)
        - Features temporelles (moyennes mobiles, écarts-types)
        
        Args:
            data (pd.DataFrame): Dataset avec les métriques brutes
            include_temporal (bool): Inclure les features temporelles
        
        Returns:
            pd.DataFrame: Features enrichies prêtes pour ML
        """
        logger.info("Préparation des features...")
        
        # Features de base
        base_features = ['cpu_usage', 'ram_usage', 'network_io', 'disk_io', 'response_time']
        
        # Vérification des colonnes requises
        missing_cols = [col for col in base_features if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        feature_data = data[base_features].copy()
        
        # Features dérivées (engineering)
        feature_data['cpu_ram_ratio'] = feature_data['cpu_usage'] / (feature_data['ram_usage'] + 1)
        feature_data['io_total'] = feature_data['network_io'] + feature_data['disk_io']
        feature_data['load_score'] = (feature_data['cpu_usage'] + feature_data['ram_usage']) / 2
        feature_data['resource_pressure'] = (feature_data['cpu_usage'] * feature_data['ram_usage']) / 10000
        
        # Features temporelles (si demandées)
        if include_temporal and len(feature_data) > 5:
            for col in base_features:
                # Moyenne mobile 5 périodes
                feature_data[f'{col}_ma5'] = feature_data[col].rolling(window=5, min_periods=1).mean()
                # Écart-type mobile 5 périodes
                feature_data[f'{col}_std5'] = feature_data[col].rolling(window=5, min_periods=1).std().fillna(0)
                # Différence avec moyenne mobile (détection changements)
                feature_data[f'{col}_diff_ma'] = feature_data[col] - feature_data[f'{col}_ma5']
        
        # Nettoyage des valeurs manquantes
        feature_data = feature_data.fillna(0)
        
        # Sauvegarde des noms de features
        self.feature_names = list(feature_data.columns)
        
        logger.info(f"Features préparées: {len(self.feature_names)} colonnes")
        return feature_data
    
    def split_data(self, X, y=None, test_size: float = 0.2, random_state: int = 42):
        """
        Divise les données en ensembles d'entraînement et de test.
        
        Args:
            X: Features
            y: Labels (None pour non-supervisé)
            test_size (float): Proportion pour le test (0.0 à 1.0)
            random_state (int): Graine aléatoire pour reproductibilité
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) ou (X_train, X_test) si y=None
        """
        if y is not None:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            return train_test_split(X, test_size=test_size, random_state=random_state)
    
    def fit(self, X, y=None, **kwargs):
        """
        Entraîne le modèle avec validation et logging MLflow.
        
        Args:
            X: Features d'entraînement
            y: Labels (None pour non-supervisé)
            **kwargs: Paramètres additionnels pour l'entraînement
        
        Returns:
            dict: Métriques d'entraînement
        """
        logger.info(f"Début entraînement {self.model_name}")
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Création et entraînement du modèle
        self.model = self._create_model(**kwargs)
        
        with mlflow.start_run(run_name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log des paramètres
            mlflow.log_params({
                "model_name": self.model_name,
                "model_type": self.model_type,
                "n_features": X_scaled.shape[1],
                "n_samples": X_scaled.shape[0],
                **kwargs
            })
            
            # Entraînement
            trained_model = self._fit_model(X_scaled, y)
            
            # Calcul des métriques
            metrics = self._calculate_training_metrics(X_scaled, y)
            
            # Log des métriques
            mlflow.log_metrics(metrics)
            
            # Sauvegarde du modèle
            if hasattr(self.model, 'predict'):
                mlflow.sklearn.log_model(self.model, f"{self.model_name}_model")
            
            self.training_metrics = metrics
            self.is_fitted = True
            
        logger.info(f"Entraînement terminé - Score: {metrics.get('score', 'N/A')}")
        return metrics
    
    def predict(self, X):
        """
        Effectue des prédictions sur de nouvelles données.
        
        Args:
            X: Features pour la prédiction
        
        Returns:
            np.array: Prédictions binaires (1=anomalie, 0=normal)
        
        Raises:
            ValueError: Si le modèle n'est pas entraîné
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # Normalisation avec le scaler d'entraînement
        X_scaled = self.scaler.transform(X)
        
        # Prédictions
        predictions = self._predict_anomalies(X_scaled)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Évalue les performances du modèle sur des données de test.
        
        Args:
            X_test: Features de test
            y_test: Labels de test
        
        Returns:
            dict: Métriques de performance
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant l'évaluation")
        
        predictions = self.predict(X_test)
        
        # Calcul des métriques
        metrics = {
            'accuracy': (predictions == y_test).mean(),
            'precision': None,
            'recall': None,
            'f1_score': None
        }
        
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics.update({
                'precision': precision_score(y_test, predictions, zero_division=0),
                'recall': recall_score(y_test, predictions, zero_division=0),
                'f1_score': f1_score(y_test, predictions, zero_division=0)
            })
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(self.scaler.transform(X_test))[:, 1]
                metrics['auc_score'] = roc_auc_score(y_test, proba)
        except Exception as e:
            logger.warning(f"Erreur calcul métriques avancées: {e}")
        
        return metrics
    
    def _calculate_training_metrics(self, X_scaled, y=None):
        """
        Calcule les métriques d'entraînement selon le type de modèle.
        
        Args:
            X_scaled: Features normalisées
            y: Labels (None pour non-supervisé)
        
        Returns:
            dict: Métriques calculées
        """
        metrics = {
            'n_samples': X_scaled.shape[0],
            'n_features': X_scaled.shape[1]
        }
        
        try:
            if self.model_type == "supervised" and y is not None:
                # Prédictions pour métriques supervisées
                predictions = self._predict_anomalies(X_scaled)
                metrics['accuracy'] = (predictions == y).mean()
                
                # Score du modèle si disponible
                if hasattr(self.model, 'score'):
                    metrics['score'] = self.model.score(X_scaled, y)
            else:
                # Métriques non-supervisées
                predictions = self._predict_anomalies(X_scaled)
                metrics['anomaly_rate'] = predictions.mean()
                
                # Score de cohérence si disponible
                if hasattr(self.model, 'score_samples'):
                    scores = self.model.score_samples(X_scaled)
                    metrics['avg_anomaly_score'] = scores.mean()
                    metrics['std_anomaly_score'] = scores.std()
        
        except Exception as e:
            logger.warning(f"Erreur calcul métriques: {e}")
            metrics['score'] = 0.0
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Sauvegarde le modèle entraîné sur disque.
        
        Args:
            filepath (str): Chemin de sauvegarde
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant la sauvegarde")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Charge un modèle depuis le disque.
        
        Args:
            filepath (str): Chemin du fichier modèle
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.is_fitted = True
        
        logger.info(f"Modèle chargé: {filepath}")
    
    def get_feature_importance(self):
        """
        Retourne l'importance des features si disponible.
        
        Returns:
            pd.DataFrame: Importance des features ou None
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.info("Ce modèle ne fournit pas d'importance des features")
            return None
    
    def summary(self):
        """
        Affiche un résumé du modèle et de ses performances.
        """
        print(f"\n{'='*50}")
        print(f"RÉSUMÉ DU MODÈLE: {self.model_name}")
        print(f"{'='*50}")
        print(f"Type: {self.model_type}")
        print(f"État: {'Entraîné' if self.is_fitted else 'Non entraîné'}")
        
        if self.is_fitted:
            print(f"Features: {len(self.feature_names) if self.feature_names else 0}")
            print(f"\nMétriques d'entraînement:")
            for key, value in self.training_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            # Importance des features si disponible
            importance = self.get_feature_importance()
            if importance is not None:
                print(f"\nTop 5 Features importantes:")
                for _, row in importance.head().iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"{'='*50}\n")


# Exemple d'utilisation de la classe de base
if __name__ == "__main__":
    # Cette classe ne peut pas être instanciée directement car elle est abstraite
    print("Classe de base pour les modèles de détection d'anomalies")
    print("Utilisez les classes dérivées: IsolationForestModel, RandomForestModel, LSTMModel")
