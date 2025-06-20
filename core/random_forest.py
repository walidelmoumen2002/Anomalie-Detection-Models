"""
Random Forest Model for IT Anomaly Detection
============================================

Modèle de détection d'anomalies basé sur Random Forest (supervisé).
Excellent pour la classification d'anomalies lorsque des données étiquetées sont disponibles.

Avantages du Random Forest:
- Haute précision sur données étiquetées
- Résistant au surapprentissage
- Fournit l'importance des features
- Gère bien les données déséquilibrées
- Rapide en inférence

Cas d'usage typiques:
- Classification d'anomalies connues
- Validation des détections d'autres modèles
- Analyse d'importance des features
- Production avec données historiques étiquetées

Author: MLOps Team
Date: June 2025
Version: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import mlflow
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from base_model import BaseAnomalyModel

logger = logging.getLogger(__name__)

class RandomForestModel(BaseAnomalyModel):
    """
    Modèle de détection d'anomalies utilisant Random Forest (supervisé).
    
    Random Forest est optimal pour:
    - Classification d'anomalies avec données étiquetées
    - Haute précision et rappel
    - Interprétabilité via importance des features
    - Robustesse face au bruit et aux outliers
    - Gestion des classes déséquilibrées
    
    Attributes:
        n_estimators (int): Nombre d'arbres dans la forêt
        max_depth (int): Profondeur maximale des arbres
        min_samples_split (int): Minimum d'échantillons pour diviser un nœud
        min_samples_leaf (int): Minimum d'échantillons dans une feuille
        max_features (str/float): Nombre de features à considérer pour chaque split
        bootstrap (bool): Utilisation du bootstrap
        class_weight (str/dict): Pondération des classes
        random_state (int): Graine aléatoire
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = "sqrt", bootstrap: bool = True,
                 class_weight: str = "balanced", random_state: int = 42):
        """
        Initialise le modèle Random Forest.
        
        Args:
            n_estimators (int): Nombre d'arbres dans la forêt
                              - 100: bon compromis (recommandé)
                              - 200+: meilleure précision, plus lent
                              - 50: plus rapide, moins précis
            max_depth (int): Profondeur maximale des arbres
                           - None: pas de limite (peut surappendre)
                           - 10-20: bon compromis pour IT
                           - 5: modèle plus simple
            min_samples_split (int): Échantillons minimum pour diviser
                                   - 2: maximum de splits (par défaut)
                                   - 5-10: évite le surapprentissage
            min_samples_leaf (int): Échantillons minimum dans une feuille
                                  - 1: maximum de détail (par défaut)
                                  - 2-5: évite le surapprentissage
            max_features (str): Features considérées pour chaque split
                              - "sqrt": racine du nombre total (recommandé)
                              - "log2": log2 du nombre total
                              - "auto": equivalent à "sqrt"
                              - float: proportion des features
            bootstrap (bool): Échantillonnage bootstrap
                            - True: avec remise (recommandé)
                            - False: sans remise
            class_weight (str/dict): Pondération des classes
                                   - "balanced": équilibre automatique (recommandé)
                                   - "balanced_subsample": équilibre par échantillon
                                   - dict: poids personnalisés {0: 1, 1: 10}
                                   - None: pas de pondération
            random_state (int): Graine aléatoire pour reproductibilité
        
        Example:
            # Configuration standard pour IT
            model = RandomForestModel(n_estimators=100, max_depth=15)
            
            # Configuration haute performance
            model = RandomForestModel(n_estimators=200, max_depth=20, min_samples_split=5)
            
            # Configuration rapide
            model = RandomForestModel(n_estimators=50, max_depth=10, max_features=0.8)
        """
        super().__init__(model_name="RandomForest", model_type="supervised")
        
        # Paramètres du modèle
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.random_state = random_state
        
        # Métriques avancées
        self.feature_importance_df = None
        self.cv_scores = None
        self.confusion_matrix = None
        
        # Validation des paramètres
        self._validate_parameters()
        
        logger.info(f"RandomForest initialisé: n_estimators={n_estimators}, "
                   f"max_depth={max_depth}, class_weight={class_weight}")
    
    def _validate_parameters(self):
        """Valide les paramètres du modèle."""
        if self.n_estimators < 1:
            raise ValueError("n_estimators doit être positif")
        
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth doit être positif ou None")
        
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split doit être >= 2")
        
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf doit être >= 1")
    
    def _create_model(self, **kwargs):
        """
        Crée l'instance Random Forest avec les paramètres configurés.
        
        Args:
            **kwargs: Paramètres additionnels pour surcharger la configuration
        
        Returns:
            RandomForestClassifier: Instance configurée du modèle
        """
        # Fusion des paramètres par défaut et additionnels
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'n_jobs': -1,  # Utilise tous les CPUs
            'verbose': 0,  # Pas de sortie verbose
            'oob_score': True if self.bootstrap else False  # Score out-of-bag
        }
        params.update(kwargs)
        
        logger.info(f"Création RandomForest avec paramètres: {params}")
        
        return RandomForestClassifier(**params)
    
    def _fit_model(self, X_train, y_train):
        """
        Entraîne le modèle Random Forest.
        
        Args:
            X_train: Features d'entraînement normalisées
            y_train: Labels d'entraînement (requis pour supervisé)
        
        Returns:
            RandomForestClassifier: Modèle entraîné
        
        Raises:
            ValueError: Si y_train est None (supervisé nécessite des labels)
        """
        if y_train is None:
            raise ValueError("Random Forest nécessite des labels d'entraînement (supervisé)")
        
        logger.info(f"Entraînement sur {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
        logger.info(f"Distribution des classes: {np.bincount(y_train)}")
        
        # Entraînement du modèle
        self.model.fit(X_train, y_train)
        
        # Calcul des métriques d'entraînement
        if self.bootstrap and hasattr(self.model, 'oob_score_'):
            logger.info(f"Score OOB: {self.model.oob_score_:.4f}")
        
        # Sauvegarde de l'importance des features
        self._calculate_feature_importance()
        
        logger.info("Entraînement Random Forest terminé")
        
        return self.model
    
    def _predict_anomalies(self, X):
        """
        Prédit les anomalies avec Random Forest.
        
        Args:
            X: Features normalisées pour la prédiction
        
        Returns:
            np.array: Prédictions binaires (1=anomalie, 0=normal)
        """
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        """
        Prédit les probabilités d'anomalies.
        
        Args:
            X: Features pour la prédiction
        
        Returns:
            np.array: Probabilités [prob_normal, prob_anomalie]
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant les prédictions")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def get_anomaly_probabilities(self, X):
        """
        Retourne uniquement les probabilités d'anomalies (classe 1).
        
        Args:
            X: Features pour la prédiction
        
        Returns:
            np.array: Probabilités d'anomalies (0.0 à 1.0)
        """
        probas = self.predict_proba(X)
        return probas[:, 1]  # Probabilité de la classe anomalie
    
    def _calculate_feature_importance(self):
        """Calcule et sauvegarde l'importance des features."""
        if self.model and hasattr(self.model, 'feature_importances_'):
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
            self.feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_,
                'importance_std': np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Importance des features calculée pour {len(self.feature_importance_df)} features")
    
    def cross_validate(self, X, y, cv_folds=5, scoring='f1'):
        """
        Effectue une validation croisée approfondie.
        
        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
            cv_folds (int): Nombre de folds pour la validation croisée
            scoring (str): Métrique de scoring ('accuracy', 'f1', 'roc_auc', 'precision', 'recall')
        
        Returns:
            dict: Résultats de la validation croisée
        """
        logger.info(f"Validation croisée avec {cv_folds} folds, métrique: {scoring}")
        
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X)
        
        # Validation croisée stratifiée (préserve la distribution des classes)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Modèle temporaire pour la validation
        temp_model = self._create_model()
        
        # Calcul des scores
        cv_scores = cross_val_score(temp_model, X_scaled, y, cv=skf, scoring=scoring, n_jobs=-1)
        
        # Métriques multiples
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        detailed_results = {}
        
        for metric in metrics:
            try:
                scores = cross_val_score(temp_model, X_scaled, y, cv=skf, scoring=metric, n_jobs=-1)
                detailed_results[metric] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max()
                }
            except Exception as e:
                logger.warning(f"Impossible de calculer {metric}: {e}")
        
        self.cv_scores = detailed_results
        
        results = {
            'primary_metric': scoring,
            'primary_scores': cv_scores,
            'primary_mean': cv_scores.mean(),
            'primary_std': cv_scores.std(),
            'detailed_metrics': detailed_results
        }
        
        logger.info(f"Validation croisée terminée - {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv_folds=3, scoring='f1'):
        """
        Optimise les hyperparamètres par Grid Search.
        
        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
            param_grid (dict): Grille des paramètres à tester
            cv_folds (int): Nombre de folds pour la validation
            scoring (str): Métrique d'optimisation
        
        Returns:
            dict: Résultats de l'optimisation
        """
        if param_grid is None:
            # Grille par défaut pour optimisation
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8]
            }
        
        logger.info(f"Optimisation hyperparamètres avec grille: {param_grid}")
        
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X)
        
        # Configuration du Grid Search
        base_model = RandomForestClassifier(
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Exécution de la recherche
        grid_search.fit(X_scaled, y)
        
        # Mise à jour des paramètres avec les meilleurs trouvés
        best_params = grid_search.best_params_
        for param, value in best_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        results = {
            'best_params': best_params,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_,
            'score_improvement': grid_search.best_score_ - np.mean(grid_search.cv_results_['mean_test_score'])
        }
        
        logger.info(f"Optimisation terminée - Meilleur score: {grid_search.best_score_:.4f}")
        logger.info(f"Meilleurs paramètres: {best_params}")
        
        return results
    
    def analyze_predictions(self, X_test, y_test, save_plots=False):
        """
        Analyse approfondie des prédictions du modèle.
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            save_plots (bool): Sauvegarder les graphiques
        
        Returns:
            dict: Analyse complète des performances
        """
        logger.info("Analyse des prédictions...")
        
        # Prédictions et probabilités
        predictions = self.predict(X_test)
        probabilities = self.get_anomaly_probabilities(X_test)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, predictions)
        self.confusion_matrix = cm
        
        # Métriques détaillées
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': (predictions == y_test).mean(),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1_score': f1_score(y_test, predictions, zero_division=0),
            'auc_score': roc_auc_score(y_test, probabilities),
            'confusion_matrix': cm
        }
        
        # Analyse par seuil
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_analysis = []
        
        for threshold in thresholds:
            thresh_predictions = (probabilities >= threshold).astype(int)
            thresh_metrics = {
                'threshold': threshold,
                'accuracy': (thresh_predictions == y_test).mean(),
                'precision': precision_score(y_test, thresh_predictions, zero_division=0),
                'recall': recall_score(y_test, thresh_predictions, zero_division=0),
                'f1_score': f1_score(y_test, thresh_predictions, zero_division=0)
            }
            threshold_analysis.append(thresh_metrics)
        
        # Identification des erreurs
        false_positives = np.where((predictions == 1) & (y_test == 0))[0]
        false_negatives = np.where((predictions == 0) & (y_test == 1))[0]
        
        analysis = {
            'basic_metrics': metrics,
            'threshold_analysis': pd.DataFrame(threshold_analysis),
            'false_positives': {
                'count': len(false_positives),
                'indices': false_positives,
                'avg_probability': probabilities[false_positives].mean() if len(false_positives) > 0 else 0
            },
            'false_negatives': {
                'count': len(false_negatives),
                'indices': false_negatives,
                'avg_probability': probabilities[false_negatives].mean() if len(false_negatives) > 0 else 0
            }
        }
        
        # Rapport de classification
        classification_rep = classification_report(y_test, predictions, output_dict=True)
        analysis['classification_report'] = classification_rep
        
        logger.info(f"Analyse terminée - Précision: {metrics['precision']:.3f}, "
                   f"Rappel: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        return analysis
    
    def detect_anomalies_with_confidence(self, X, confidence_threshold=0.8):
        """
        Détecte les anomalies avec seuil de confiance.
        
        Args:
            X: Features pour la détection
            confidence_threshold (float): Seuil de confiance minimum (0.5 à 1.0)
        
        Returns:
            dict: Détections avec niveaux de confiance
        """
        probabilities = self.get_anomaly_probabilities(X)
        
        # Classifications avec seuils
        high_confidence_anomalies = probabilities >= confidence_threshold
        medium_confidence_anomalies = (probabilities >= 0.6) & (probabilities < confidence_threshold)
        low_confidence_anomalies = (probabilities >= 0.5) & (probabilities < 0.6)
        
        results = {
            'probabilities': probabilities,
            'high_confidence': {
                'indices': np.where(high_confidence_anomalies)[0],
                'count': np.sum(high_confidence_anomalies),
                'avg_probability': probabilities[high_confidence_anomalies].mean() if np.any(high_confidence_anomalies) else 0
            },
            'medium_confidence': {
                'indices': np.where(medium_confidence_anomalies)[0],
                'count': np.sum(medium_confidence_anomalies),
                'avg_probability': probabilities[medium_confidence_anomalies].mean() if np.any(medium_confidence_anomalies) else 0
            },
            'low_confidence': {
                'indices': np.where(low_confidence_anomalies)[0],
                'count': np.sum(low_confidence_anomalies),
                'avg_probability': probabilities[low_confidence_anomalies].mean() if np.any(low_confidence_anomalies) else 0
            }
        }
        
        return results
    
    def explain_predictions(self, X_sample, feature_names=None, top_n=5):
        """
        Explique les prédictions en montrant l'influence des features.
        
        Args:
            X_sample: Échantillon à expliquer (single row ou multiple rows)
            feature_names: Noms des features
            top_n (int): Nombre de top features à montrer
        
        Returns:
            dict: Explication des prédictions
        """
        if feature_names is None:
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X_sample.shape[1])]
        
        # Assurer que c'est un array 2D
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        predictions = self.predict(X_sample)
        probabilities = self.get_anomaly_probabilities(X_sample)
        
        explanations = []
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Importance des features pour cet échantillon
            sample_features = X_sample[i]
            
            # Calcul de l'influence relative (feature_value * feature_importance)
            if self.feature_importance_df is not None:
                feature_influences = sample_features * self.feature_importance_df['importance'].values
                
                # Top features influentes
                top_indices = np.argsort(np.abs(feature_influences))[-top_n:][::-1]
                
                top_features = []
                for idx in top_indices:
                    top_features.append({
                        'feature_name': feature_names[idx],
                        'feature_value': sample_features[idx],
                        'feature_importance': self.feature_importance_df.iloc[idx]['importance'],
                        'influence': feature_influences[idx]
                    })
                
                explanation = {
                    'sample_index': i,
                    'prediction': pred,
                    'probability': prob,
                    'confidence': 'High' if prob > 0.8 else 'Medium' if prob > 0.6 else 'Low',
                    'top_influencing_features': top_features
                }
                
                explanations.append(explanation)
        
        return explanations


def demonstrate_random_forest():
    """
    Démonstration complète du modèle Random Forest.
    """
    print("🌳 DÉMONSTRATION RANDOM FOREST")
    print("=" * 50)
    
    # 1. Initialisation
    print("\n1️⃣ Initialisation du modèle")
    model = RandomForestModel(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        random_state=42
    )
    print(f"✅ Modèle initialisé: {model.model_name}")
    
    # 2. Génération des données
    print("\n2️⃣ Génération des données")
    data = model.generate_synthetic_data(n_samples=5000, anomaly_rate=0.05)
    X = model.prepare_features(data, include_temporal=True)
    y = data['is_anomaly']
    print(f"✅ {len(data)} échantillons, {X.shape[1]} features")
    
    # 3. Division des données
    X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.2)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Validation croisée
    print("\n4️⃣ Validation croisée")
    cv_results = model.cross_validate(X_train, y_train, cv_folds=5, scoring='f1')
    print(f"✅ F1-Score moyen: {cv_results['primary_mean']:.3f} ± {cv_results['primary_std']:.3f}")
    
    # 5. Entraînement
    print("\n5️⃣ Entraînement du modèle")
    training_metrics = model.fit(X_train, y_train)
    print(f"✅ Entraînement terminé")
    
    # 6. Évaluation
    print("\n6️⃣ Évaluation détaillée")
    analysis = model.analyze_predictions(X_test, y_test)
    metrics = analysis['basic_metrics']
    print(f"✅ Précision: {metrics['precision']:.3f}")
    print(f"✅ Rappel: {metrics['recall']:.3f}")
    print(f"✅ F1-Score: {metrics['f1_score']:.3f}")
    print(f"✅ AUC: {metrics['auc_score']:.3f}")
    
    # 7. Importance des features
    print("\n7️⃣ Importance des features")
    importance = model.get_feature_importance()
    if importance is not None:
        print("Top 5 features importantes:")
        for _, row in importance.head().iterrows():
            print(f"   - {row['feature']}: {row['importance']:.4f}")
    
    # 8. Détection avec confiance
    print("\n8️⃣ Détection avec niveaux de confiance")
    confidence_results = model.detect_anomalies_with_confidence(X_test, confidence_threshold=0.8)
    print(f"✅ Haute confiance: {confidence_results['high_confidence']['count']} anomalies")
    print(f"✅ Confiance moyenne: {confidence_results['medium_confidence']['count']} anomalies")
    
    # 9. Résumé
    print("\n9️⃣ Résumé du modèle")
    model.summary()
    
    return model


if __name__ == "__main__":
    # Démonstration complète
    trained_model = demonstrate_random_forest()
    
    print("\n🎯 RANDOM FOREST PRÊT POUR LA PRODUCTION!")
    print("Utilisez ce modèle pour la classification supervisée d'anomalies.")
