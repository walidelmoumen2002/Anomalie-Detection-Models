"""
Isolation Forest Model for IT Anomaly Detection
===============================================

Modèle de détection d'anomalies basé sur l'algorithme Isolation Forest.
Parfait pour la détection d'anomalies non-supervisée dans les infrastructures IT.

Principes de l'Isolation Forest:
- Isole les anomalies en construisant des arbres aléatoires
- Les anomalies sont plus faciles à isoler (moins de splits nécessaires)
- Ne nécessite pas de données étiquetées
- Très efficace sur de gros volumes de données

Cas d'usage typiques:
- Détection d'anomalies système inconnues
- Monitoring continu sans supervision
- Première ligne de défense contre les incidents

Author: MLOps Team
Date: June 2025
Version: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
import mlflow
import logging
from base_model import BaseAnomalyModel

logger = logging.getLogger(__name__)

class IsolationForestModel(BaseAnomalyModel):
    """
    Modèle de détection d'anomalies utilisant Isolation Forest.
    
    L'Isolation Forest est particulièrement adapté pour:
    - La détection d'anomalies sans supervision
    - Le traitement de gros volumes de données
    - L'identification d'outliers multivariés
    - La robustesse face au bruit
    
    Attributes:
        contamination (float): Proportion attendue d'anomalies dans les données
        n_estimators (int): Nombre d'arbres dans la forêt
        max_samples (str/int): Nombre d'échantillons pour construire chaque arbre
        max_features (float): Proportion de features utilisées par arbre
        bootstrap (bool): Utilisation du bootstrap pour l'échantillonnage
        random_state (int): Graine aléatoire pour reproductibilité
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100, 
                 max_samples: str = "auto", max_features: float = 1.0,
                 bootstrap: bool = False, random_state: int = 42):
        """
        Initialise le modèle Isolation Forest.
        
        Args:
            contamination (float): Proportion d'anomalies attendues (0.0 à 0.5)
                                 - 0.05 = 5% d'anomalies (recommandé pour IT)
                                 - 0.1 = 10% d'anomalies (environnement instable)
            n_estimators (int): Nombre d'arbres dans la forêt
                              - 100: bon compromis performance/précision
                              - 200+: meilleure précision, plus lent
            max_samples (str/int): Échantillons par arbre
                                 - "auto": min(256, n_samples)
                                 - int: nombre fixe d'échantillons
            max_features (float): Proportion de features par arbre (0.0 à 1.0)
                                - 1.0: toutes les features (recommandé)
                                - 0.5: la moitié des features (plus rapide)
            bootstrap (bool): Échantillonnage avec remise
                            - False: sans remise (recommandé)
                            - True: avec remise (plus de diversité)
            random_state (int): Graine aléatoire pour reproductibilité
        
        Example:
            # Configuration standard pour IT monitoring
            model = IsolationForestModel(contamination=0.05, n_estimators=100)
            
            # Configuration haute performance
            model = IsolationForestModel(contamination=0.03, n_estimators=200, max_samples=512)
            
            # Configuration rapide pour gros volumes
            model = IsolationForestModel(contamination=0.1, n_estimators=50, max_features=0.8)
        """
        super().__init__(model_name="IsolationForest", model_type="unsupervised")
        
        # Paramètres du modèle
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # Validation des paramètres
        self._validate_parameters()
        
        logger.info(f"IsolationForest initialisé: contamination={contamination}, "
                   f"n_estimators={n_estimators}")
    
    def _validate_parameters(self):
        """Valide les paramètres du modèle."""
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination doit être entre 0.0 et 0.5")
        
        if self.n_estimators < 1:
            raise ValueError("n_estimators doit être positif")
        
        if not 0.0 < self.max_features <= 1.0:
            raise ValueError("max_features doit être entre 0.0 et 1.0")
    
    def _create_model(self, **kwargs):
        """
        Crée l'instance Isolation Forest avec les paramètres configurés.
        
        Args:
            **kwargs: Paramètres additionnels pour surcharger la configuration
        
        Returns:
            IsolationForest: Instance configurée du modèle
        """
        # Fusion des paramètres par défaut et additionnels
        params = {
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'n_jobs': -1,  # Utilise tous les CPUs disponibles
            'verbose': 0   # Pas de sortie verbose
        }
        params.update(kwargs)
        
        logger.info(f"Création IsolationForest avec paramètres: {params}")
        
        return IsolationForest(**params)
    
    def _fit_model(self, X_train, y_train=None):
        """
        Entraîne le modèle Isolation Forest.
        
        Note: y_train est ignoré car Isolation Forest est non-supervisé.
        
        Args:
            X_train: Features d'entraînement normalisées
            y_train: Ignoré (non-supervisé)
        
        Returns:
            IsolationForest: Modèle entraîné
        """
        logger.info(f"Entraînement sur {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
        
        # Entraînement du modèle
        self.model.fit(X_train)
        
        # Log des informations d'entraînement
        logger.info("Entraînement Isolation Forest terminé")
        
        return self.model
    
    def _predict_anomalies(self, X):
        """
        Prédit les anomalies avec Isolation Forest.
        
        Args:
            X: Features normalisées pour la prédiction
        
        Returns:
            np.array: Prédictions binaires (1=anomalie, 0=normal)
        """
        # Isolation Forest retourne -1 pour anomalies, 1 pour normal
        raw_predictions = self.model.predict(X)
        
        # Conversion en format standard (1=anomalie, 0=normal)
        predictions = (raw_predictions == -1).astype(int)
        
        return predictions
    
    def get_anomaly_scores(self, X):
        """
        Calcule les scores d'anomalie pour chaque échantillon.
        
        Plus le score est faible (négatif), plus l'échantillon est anormal.
        Utile pour classer les anomalies par sévérité.
        
        Args:
            X: Features pour le calcul des scores
        
        Returns:
            np.array: Scores d'anomalie (valeurs négatives)
        
        Example:
            scores = model.get_anomaly_scores(X_test)
            # Anomalies les plus sévères
            most_anomalous = np.argsort(scores)[:10]
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant le calcul des scores")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        
        return scores
    
    def get_decision_scores(self, X):
        """
        Calcule les scores de décision pour chaque échantillon.
        
        Returns:
            np.array: Scores de décision (plus proche de 0 = plus anormal)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant le calcul des scores")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        
        return scores
    
    def find_optimal_contamination(self, X, contamination_range=None, cv_folds=3):
        """
        Trouve le taux de contamination optimal par validation croisée.
        
        Cette méthode teste différents taux de contamination et sélectionne
        celui qui maximise la cohérence des prédictions.
        
        Args:
            X: Features d'entraînement
            contamination_range: Liste des taux à tester
            cv_folds: Nombre de folds pour la validation croisée
        
        Returns:
            dict: Résultats de l'optimisation
        
        Example:
            results = model.find_optimal_contamination(X_train)
            print(f"Meilleur taux: {results['best_contamination']}")
        """
        if contamination_range is None:
            contamination_range = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
        
        logger.info(f"Recherche contamination optimale parmi: {contamination_range}")
        
        results = {
            'contamination_tested': contamination_range,
            'scores': [],
            'best_contamination': None,
            'best_score': -np.inf
        }
        
        X_scaled = self.scaler.fit_transform(X)
        
        for contamination in contamination_range:
            # Test du modèle avec ce taux de contamination
            test_model = IsolationForest(
                contamination=contamination,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Validation croisée simplifiée
            fold_scores = []
            fold_size = len(X_scaled) // cv_folds
            
            for fold in range(cv_folds):
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else len(X_scaled)
                
                # Division train/validation
                val_indices = range(start_idx, end_idx)
                train_indices = list(range(0, start_idx)) + list(range(end_idx, len(X_scaled)))
                
                X_fold_train = X_scaled[train_indices]
                X_fold_val = X_scaled[val_indices]
                
                # Entraînement et test
                test_model.fit(X_fold_train)
                scores = test_model.score_samples(X_fold_val)
                
                # Score de cohérence (variance des scores)
                fold_score = -np.var(scores)  # Plus c'est stable, mieux c'est
                fold_scores.append(fold_score)
            
            avg_score = np.mean(fold_scores)
            results['scores'].append(avg_score)
            
            if avg_score > results['best_score']:
                results['best_score'] = avg_score
                results['best_contamination'] = contamination
            
            logger.info(f"Contamination {contamination}: score = {avg_score:.4f}")
        
        logger.info(f"Meilleure contamination: {results['best_contamination']} "
                   f"(score: {results['best_score']:.4f})")
        
        # Mise à jour du modèle avec le meilleur paramètre
        self.contamination = results['best_contamination']
        
        return results
    
    def detect_realtime_anomalies(self, new_data, threshold_percentile=95):
        """
        Détecte les anomalies en temps réel avec seuillage adaptatif.
        
        Args:
            new_data: Nouvelles données à analyser
            threshold_percentile: Percentile pour le seuil d'anomalie
        
        Returns:
            dict: Résultats de détection avec scores et seuils
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant la détection")
        
        # Prédictions et scores
        predictions = self.predict(new_data)
        scores = self.get_anomaly_scores(new_data)
        decision_scores = self.get_decision_scores(new_data)
        
        # Seuil adaptatif basé sur la distribution des scores
        threshold = np.percentile(scores, threshold_percentile)
        
        # Anomalies sévères (en dessous du seuil)
        severe_anomalies = scores < threshold
        
        results = {
            'predictions': predictions,
            'anomaly_scores': scores,
            'decision_scores': decision_scores,
            'threshold': threshold,
            'severe_anomalies': severe_anomalies,
            'n_anomalies': np.sum(predictions),
            'n_severe_anomalies': np.sum(severe_anomalies),
            'anomaly_rate': np.mean(predictions),
            'severe_anomaly_rate': np.mean(severe_anomalies)
        }
        
        return results
    
    def explain_anomalies(self, X_anomalies, feature_names=None):
        """
        Analyse les caractéristiques des anomalies détectées.
        
        Args:
            X_anomalies: Données identifiées comme anomalies
            feature_names: Noms des features
        
        Returns:
            dict: Analyse des anomalies
        """
        if feature_names is None:
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X_anomalies.shape[1])]
        
        # Statistiques des anomalies
        anomaly_stats = pd.DataFrame(X_anomalies, columns=feature_names).describe()
        
        # Features les plus déviantes (z-score élevé)
        z_scores = np.abs((X_anomalies - np.mean(X_anomalies, axis=0)) / (np.std(X_anomalies, axis=0) + 1e-8))
        avg_z_scores = np.mean(z_scores, axis=0)
        
        top_features = pd.DataFrame({
            'feature': feature_names,
            'avg_z_score': avg_z_scores
        }).sort_values('avg_z_score', ascending=False)
        
        analysis = {
            'n_anomalies': len(X_anomalies),
            'anomaly_statistics': anomaly_stats,
            'top_anomalous_features': top_features.head(10),
            'feature_correlations': pd.DataFrame(X_anomalies, columns=feature_names).corr()
        }
        
        return analysis


def demonstrate_isolation_forest():
    """
    Démonstration complète du modèle Isolation Forest.
    
    Cette fonction montre toutes les fonctionnalités du modèle avec des exemples pratiques.
    """
    print("🌲 DÉMONSTRATION ISOLATION FOREST")
    print("=" * 50)
    
    # 1. Initialisation du modèle
    print("\n1️⃣ Initialisation du modèle")
    model = IsolationForestModel(
        contamination=0.05,  # 5% d'anomalies attendues
        n_estimators=100,    # 100 arbres dans la forêt
        random_state=42      # Pour reproductibilité
    )
    print(f"✅ Modèle initialisé: {model.model_name}")
    
    # 2. Génération des données
    print("\n2️⃣ Génération des données d'entraînement")
    data = model.generate_synthetic_data(n_samples=5000, anomaly_rate=0.05)
    print(f"✅ {len(data)} échantillons générés avec {data['is_anomaly'].sum()} anomalies")
    
    # 3. Préparation des features
    print("\n3️⃣ Préparation des features")
    X = model.prepare_features(data, include_temporal=True)
    y = data['is_anomaly']
    print(f"✅ {X.shape[1]} features préparées")
    
    # 4. Division des données
    print("\n4️⃣ Division train/test")
    X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.2)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 5. Optimisation des hyperparamètres
    print("\n5️⃣ Optimisation du taux de contamination")
    optimization_results = model.find_optimal_contamination(X_train)
    print(f"✅ Meilleur taux: {optimization_results['best_contamination']}")
    
    # 6. Entraînement
    print("\n6️⃣ Entraînement du modèle")
    training_metrics = model.fit(X_train)
    print(f"✅ Entraînement terminé - Score: {training_metrics.get('score', 'N/A')}")
    
    # 7. Évaluation
    print("\n7️⃣ Évaluation sur données de test")
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"✅ Précision: {eval_metrics['accuracy']:.3f}")
    print(f"✅ Rappel: {eval_metrics.get('recall', 'N/A')}")
    
    # 8. Détection en temps réel
    print("\n8️⃣ Détection d'anomalies en temps réel")
    rt_results = model.detect_realtime_anomalies(X_test[:100])
    print(f"✅ {rt_results['n_anomalies']} anomalies détectées sur 100 échantillons")
    print(f"✅ {rt_results['n_severe_anomalies']} anomalies sévères")
    
    # 9. Analyse des anomalies
    print("\n9️⃣ Analyse des anomalies détectées")
    anomaly_indices = np.where(rt_results['predictions'] == 1)[0]
    if len(anomaly_indices) > 0:
        X_anomalies = X_test.iloc[anomaly_indices]
        analysis = model.explain_anomalies(X_anomalies.values, X_anomalies.columns)
        print(f"✅ {analysis['n_anomalies']} anomalies analysées")
        print("Top 3 features les plus anomales:")
        for _, row in analysis['top_anomalous_features'].head(3).iterrows():
            print(f"   - {row['feature']}: score {row['avg_z_score']:.2f}")
    
    # 10. Résumé du modèle
    print("\n🔟 Résumé du modèle")
    model.summary()
    
    return model


# Exemple d'usage avancé
def advanced_isolation_forest_usage():
    """
    Exemples d'usage avancés du modèle Isolation Forest.
    """
    print("\n🚀 USAGE AVANCÉ ISOLATION FOREST")
    print("=" * 50)
    
    # Configuration pour différents scénarios
    scenarios = {
        "Production stable": {
            "contamination": 0.02,
            "n_estimators": 150,
            "max_samples": "auto"
        },
        "Environnement test": {
            "contamination": 0.1,
            "n_estimators": 50,
            "max_samples": 256
        },
        "Haute précision": {
            "contamination": 0.03,
            "n_estimators": 300,
            "max_samples": 512
        }
    }
    
    for scenario_name, params in scenarios.items():
        print(f"\n📊 Scénario: {scenario_name}")
        model = IsolationForestModel(**params)
        print(f"   Paramètres: {params}")
        print(f"   Modèle prêt pour: {scenario_name}")


if __name__ == "__main__":
    # Démonstration complète
    trained_model = demonstrate_isolation_forest()
    
    # Usage avancé
    advanced_isolation_forest_usage()
    
    print("\n🎯 ISOLATION FOREST PRÊT POUR LA PRODUCTION!")
    print("Utilisez ce modèle pour la détection d'anomalies non-supervisée.")
