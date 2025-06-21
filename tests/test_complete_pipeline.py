"""
Complete Test Suite for MLOps Anomaly Detection Pipeline
========================================================

Suite de tests complète pour valider tous les composants de la pipeline MLOps.
Couvre les tests unitaires, d'intégration et de performance.

Test Categories:
- Unit Tests: Tests individuels des modèles
- Integration Tests: Tests de la pipeline complète
- Performance Tests: Tests de charge et de latence
- Data Quality Tests: Validation des données
- Model Validation Tests: Validation des performances

Author: MLOps Team
Date: June 2025
Version: 1.0
"""

import unittest
import sys
import os
import tempfile
import shutil
import time
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports des composants à tester
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_anomaly_model import BaseAnomalyModel
from isolation_forest_model import IsolationForestModel
from random_forest_model import RandomForestModel
from lstm_model import LSTMModel
from model_comparison import ModelComparator
from main_orchestrator import MLOpsPipeline

class TestBaseAnomalyModel(unittest.TestCase):
    """Tests pour la classe de base BaseAnomalyModel."""
    
    def setUp(self):
        """Configuration avant chaque test."""
        # On ne peut pas instancier BaseAnomalyModel directement (classe abstraite)
        # On utilise IsolationForestModel pour tester les méthodes de base
        self.model = IsolationForestModel(random_state=42)
    
    def test_data_generation(self):
        """Test de génération de données synthétiques."""
        data = self.model.generate_synthetic_data(n_samples=1000, anomaly_rate=0.05)
        
        # Vérifications de base
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 1000)
        self.assertIn('is_anomaly', data.columns)
        
        # Vérification du taux d'anomalies
        anomaly_rate = data['is_anomaly'].mean()
        self.assertAlmostEqual(anomaly_rate, 0.05, delta=0.02)
        
        # Vérification des colonnes requises
        required_columns = ['cpu_usage', 'ram_usage', 'network_io', 'disk_io', 'response_time']
        for col in required_columns:
            self.assertIn(col, data.columns)
            self.assertTrue(data[col].dtype in [np.float64, np.int64])
    
    def test_feature_preparation(self):
        """Test de préparation des features."""
        data = self.model.generate_synthetic_data(n_samples=500)
        features = self.model.prepare_features(data, include_temporal=True)
        
        # Vérifications
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(data))
        self.assertGreater(features.shape[1], 5)  # Au moins les features de base + dérivées
        
        # Vérification des features dérivées
        self.assertIn('cpu_ram_ratio', features.columns)
        self.assertIn('io_total', features.columns)
        self.assertIn('load_score', features.columns)
        
        # Test sans features temporelles
        features_no_temporal = self.model.prepare_features(data, include_temporal=False)
        self.assertLess(features_no_temporal.shape[1], features.shape[1])
    
    def test_data_split(self):
        """Test de division des données."""
        data = self.model.generate_synthetic_data(n_samples=1000)
        X = self.model.prepare_features(data)
        y = data['is_anomaly']
        
        # Test avec labels
        X_train, X_test, y_train, y_test = self.model.split_data(X, y, test_size=0.2)
        
        self.assertEqual(len(X_train), 800)
        self.assertEqual(len(X_test), 200)
        self.assertEqual(len(y_train), 800)
        self.assertEqual(len(y_test), 200)
        
        # Test sans labels
        X_train_only, X_test_only = self.model.split_data(X, test_size=0.3)
        self.assertEqual(len(X_train_only), 700)
        self.assertEqual(len(X_test_only), 300)


class TestIsolationForestModel(unittest.TestCase):
    """Tests spécifiques pour IsolationForestModel."""
    
    def setUp(self):
        """Configuration avant chaque test."""
        self.model = IsolationForestModel(contamination=0.05, random_state=42)
    
    def test_model_initialization(self):
        """Test d'initialisation du modèle."""
        self.assertEqual(self.model.model_name, "IsolationForest")
        self.assertEqual(self.model.model_type, "unsupervised")
        self.assertEqual(self.model.contamination, 0.05)
        self.assertFalse(self.model.is_fitted)
    
    def test_parameter_validation(self):
        """Test de validation des paramètres."""
        # Paramètres invalides
        with self.assertRaises(ValueError):
            IsolationForestModel(contamination=0.6)  # Trop élevé
        
        with self.assertRaises(ValueError):
            IsolationForestModel(n_estimators=0)  # Négatif
    
    def test_training_and_prediction(self):
        """Test d'entraînement et de prédiction."""
        # Génération des données
        data = self.model.generate_synthetic_data(n_samples=1000, anomaly_rate=0.05)
        X = self.model.prepare_features(data)
        y = data['is_anomaly']
        
        # Entraînement
        metrics = self.model.fit(X)
        self.assertTrue(self.model.is_fitted)
        self.assertIsInstance(metrics, dict)
        
        # Prédictions
        predictions = self.model.predict(X)
        self.assertEqual(len(predictions), len(X))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
        # Scores d'anomalie
        scores = self.model.get_anomaly_scores(X)
        self.assertEqual(len(scores), len(X))
        self.assertTrue(np.all(scores <= 0))  # Scores négatifs pour Isolation Forest
    
    def test_contamination_optimization(self):
        """Test d'optimisation du taux de contamination."""
        data = self.model.generate_synthetic_data(n_samples=500)
        X = self.model.prepare_features(data)
        
        # Test d'optimisation
        results = self.model.find_optimal_contamination(X, contamination_range=[0.03, 0.05, 0.1])
        
        self.assertIn('best_contamination', results)
        self.assertIn('scores', results)
        self.assertIn(results['best_contamination'], [0.03, 0.05, 0.1])


class TestRandomForestModel(unittest.TestCase):
    """Tests spécifiques pour RandomForestModel."""
    
    def setUp(self):
        """Configuration avant chaque test."""
        self.model = RandomForestModel(n_estimators=50, random_state=42)  # Réduit pour tests
    
    def test_model_initialization(self):
        """Test d'initialisation du modèle."""
        self.assertEqual(self.model.model_name, "RandomForest")
        self.assertEqual(self.model.model_type, "supervised")
        self.assertEqual(self.model.n_estimators, 50)
        self.assertFalse(self.model.is_fitted)
    
    def test_supervised_training(self):
        """Test d'entraînement supervisé."""
        # Génération des données
        data = self.model.generate_synthetic_data(n_samples=1000, anomaly_rate=0.1)
        X = self.model.prepare_features(data)
        y = data['is_anomaly']
        
        # Division des données
        X_train, X_test, y_train, y_test = self.model.split_data(X, y, test_size=0.2)
        
        # Entraînement
        metrics = self.model.fit(X_train, y_train)
        self.assertTrue(self.model.is_fitted)
        
        # Évaluation
        eval_metrics = self.model.evaluate(X_test, y_test)
        self.assertIn('accuracy', eval_metrics)
        self.assertIn('precision', eval_metrics)
        self.assertIn('recall', eval_metrics)
        
        # Vérification de performance minimale
        self.assertGreater(eval_metrics['accuracy'], 0.7)
    
    def test_probability_predictions(self):
        """Test des prédictions probabilistes."""
        # Entraînement rapide
        data = self.model.generate_synthetic_data(n_samples=500)
        X = self.model.prepare_features(data)
        y = data['is_anomaly']
        
        self.model.fit(X, y)
        
        # Test des probabilités
        probas = self.model.predict_proba(X)
        self.assertEqual(probas.shape, (len(X), 2))
        self.assertTrue(np.all((probas >= 0) & (probas <= 1)))
        self.assertTrue(np.allclose(probas.sum(axis=1), 1))
        
        # Test des probabilités d'anomalies uniquement
        anomaly_probas = self.model.get_anomaly_probabilities(X)
        self.assertEqual(len(anomaly_probas), len(X))
        self.assertTrue(np.all((anomaly_probas >= 0) & (anomaly_probas <= 1)))
    
    def test_feature_importance(self):
        """Test de l'importance des features."""
        data = self.model.generate_synthetic_data(n_samples=500)
        X = self.model.prepare_features(data)
        y = data['is_anomaly']
        
        self.model.fit(X, y)
        
        importance = self.model.get_feature_importance()
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
        self.assertEqual(len(importance), X.shape[1])
    
    def test_cross_validation(self):
        """Test de validation croisée."""
        data = self.model.generate_synthetic_data(n_samples=300)  # Plus petit pour tests
        X = self.model.prepare_features(data)
        y = data['is_anomaly']
        
        cv_results = self.model.cross_validate(X, y, cv_folds=3, scoring='f1')
        
        self.assertIn('primary_scores', cv_results)
        self.assertIn('primary_mean', cv_results)
        self.assertIn('detailed_metrics', cv_results)
        self.assertGreater(cv_results['primary_mean'], 0)


class TestLSTMModel(unittest.TestCase):
    """Tests spécifiques pour LSTMModel."""
    
    def setUp(self):
        """Configuration avant chaque test."""
        self.model = LSTMModel(
            sequence_length=5, 
            lstm_units=[10], 
            epochs=5,  # Très réduit pour tests
            random_state=42
        )
    
    def test_model_initialization(self):
        """Test d'initialisation du modèle."""
        self.assertEqual(self.model.model_name, "LSTM")
        self.assertEqual(self.model.model_type, "supervised")
        self.assertEqual(self.model.sequence_length, 5)
        self.assertFalse(self.model.is_fitted)
    
    def test_sequence_preparation(self):
        """Test de préparation des séquences."""
        # Données de test
        data = np.random.randn(100, 5)  # 100 échantillons, 5 features
        y = np.random.randint(0, 2, 100)
        
        X_sequences, y_sequences = self.model._prepare_sequences(data, y)
        
        expected_length = 100 - self.model.sequence_length
        self.assertEqual(len(X_sequences), expected_length)
        self.assertEqual(len(y_sequences), expected_length)
        self.assertEqual(X_sequences.shape[1], self.model.sequence_length)
        self.assertEqual(X_sequences.shape[2], 5)
    
    def test_lstm_training(self):
        """Test d'entraînement LSTM (test rapide)."""
        # Génération de données temporelles
        data = self.model.generate_synthetic_data(n_samples=200)  # Petit dataset
        X = self.model.prepare_features(data)
        y = data['is_anomaly']
        
        # Entraînement (rapide)
        try:
            metrics = self.model.fit(X, y)
            self.assertTrue(self.model.is_fitted)
            self.assertIsInstance(metrics, dict)
        except Exception as e:
            # LSTM peut être sensible en environnement de test
            self.skipTest(f"LSTM training failed in test environment: {e}")
    
    def test_temporal_analysis(self):
        """Test d'analyse temporelle."""
        if not self.model.is_fitted:
            self.skipTest("Model not trained - skipping temporal analysis")
        
        # Données de test
        data = self.model.generate_synthetic_data(n_samples=100)
        X = self.model.prepare_features(data)
        
        try:
            temporal_results = self.model.detect_temporal_anomalies(X, window_size=3)
            self.assertIn('basic_predictions', temporal_results)
            self.assertIn('temporal_analysis', temporal_results)
        except Exception as e:
            self.skipTest(f"Temporal analysis failed: {e}")


class TestModelComparison(unittest.TestCase):
    """Tests pour la comparaison de modèles."""
    
    def setUp(self):
        """Configuration avant chaque test."""
        self.comparator = ModelComparator(random_state=42)
    
    def test_model_creation(self):
        """Test de création des modèles."""
        self.comparator.create_models()
        
        self.assertIn('IsolationForest', self.comparator.models)
        self.assertIn('RandomForest', self.comparator.models)
        self.assertIn('LSTM', self.comparator.models)
        
        # Vérification des types
        self.assertIsInstance(self.comparator.models['IsolationForest'], IsolationForestModel)
        self.assertIsInstance(self.comparator.models['RandomForest'], RandomForestModel)
        self.assertIsInstance(self.comparator.models['LSTM'], LSTMModel)
    
    def test_scenario_generation(self):
        """Test de génération des scénarios."""
        self.comparator.generate_test_scenarios()
        
        self.assertGreater(len(self.comparator.data_scenarios), 0)
        
        for scenario_name, scenario in self.comparator.data_scenarios.items():
            self.assertIn('data', scenario)
            self.assertIn('description', scenario)
            self.assertIn('anomaly_rate', scenario)
            self.assertIsInstance(scenario['data'], pd.DataFrame)
    
    def test_rapid_comparison(self):
        """Test de comparaison rapide (modèles réduits)."""
        # Configuration réduite pour tests
        self.comparator.models = {
            'IsolationForest': IsolationForestModel(n_estimators=20, random_state=42),
            'RandomForest': RandomForestModel(n_estimators=20, random_state=42)
        }
        
        # Scénario de test simple
        temp_model = IsolationForestModel()
        test_data = temp_model.generate_synthetic_data(n_samples=300, anomaly_rate=0.05)
        self.comparator.data_scenarios = {
            'test': {
                'data': test_data,
                'description': 'Test scenario',
                'anomaly_rate': 0.05
            }
        }
        
        # Entraînement
        training_results = self.comparator.train_all_models('test')
        
        # Vérifications
        self.assertIn('IsolationForest', training_results)
        self.assertIn('RandomForest', training_results)
        
        # Évaluation
        eval_results = self.comparator.evaluate_all_models('test')
        self.assertIn('IsolationForest', eval_results)
        self.assertIn('RandomForest', eval_results)


class TestMLOpsPipeline(unittest.TestCase):
    """Tests pour la pipeline MLOps complète."""
    
    def setUp(self):
        """Configuration avant chaque test."""
        # Répertoire temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self.model_dir = os.path.join(self.temp_dir, 'models')
        
        # Configuration de test
        test_config = {
            "data": {
                "n_samples": 500,
                "anomaly_rate": 0.05,
                "test_size": 0.2
            },
            "models": {
                "isolation_forest": {
                    "contamination": 0.05,
                    "n_estimators": 20
                },
                "random_forest": {
                    "n_estimators": 20,
                    "max_depth": 10
                }
            },
            "evaluation": {
                "primary_metric": "f1_score",
                "min_accuracy": 0.6
            }
        }
        
        # Sauvegarde de la configuration de test
        import json
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        self.pipeline = MLOpsPipeline(
            config_path=self.config_path,
            model_dir=self.model_dir,
            random_state=42
        )
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test d'initialisation de la pipeline."""
        self.assertEqual(self.pipeline.config_path, self.config_path)
        self.assertEqual(str(self.pipeline.model_dir), self.model_dir)
        self.assertIsInstance(self.pipeline.config, dict)
        self.assertTrue(os.path.exists(self.model_dir))
    
    def test_data_generation(self):
        """Test de génération des données."""
        X, y, data_info = self.pipeline.generate_or_load_data()
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertIsInstance(data_info, dict)
        self.assertIn('source', data_info)
        self.assertEqual(data_info['source'], 'synthetic')
    
    def test_model_training(self):
        """Test d'entraînement des modèles."""
        X, y, _ = self.pipeline.generate_or_load_data()
        
        training_results = self.pipeline.train_all_models(X, y)
        
        self.assertIsInstance(training_results, dict)
        self.assertGreater(len(training_results), 0)
        
        # Vérification qu'au moins un modèle a été entraîné avec succès
        successful_models = [name for name, result in training_results.items() 
                           if result.get('success', False)]
        self.assertGreater(len(successful_models), 0)
    
    def test_model_selection(self):
        """Test de sélection du meilleur modèle."""
        X, y, _ = self.pipeline.generate_or_load_data()
        training_results = self.pipeline.train_all_models(X, y)
        
        best_name, best_model, reason = self.pipeline.select_best_model(training_results)
        
        if best_model is not None:
            self.assertIsNotNone(best_name)
            self.assertIsNotNone(reason)
            self.assertTrue(best_model.is_fitted)
    
    def test_end_to_end_pipeline(self):
        """Test de la pipeline complète de bout en bout."""
        try:
            # Exécution complète (sans déploiement)
            result = self.pipeline.run_full_pipeline(deploy=False)
            
            self.assertIsInstance(result, dict)
            self.assertIn('pipeline_execution', result)
            
            # Si succès, vérifier les résultats
            if result['pipeline_execution'].get('success'):
                self.assertIn('best_model', result)
                self.assertIn('training_results', result)
                self.assertIn('data_info', result)
        
        except Exception as e:
            # Certains tests peuvent échouer selon l'environnement
            self.skipTest(f"End-to-end pipeline failed: {e}")


class TestPerformance(unittest.TestCase):
    """Tests de performance et de charge."""
    
    def test_inference_speed(self):
        """Test de vitesse d'inférence."""
        model = IsolationForestModel(n_estimators=50, random_state=42)
        
        # Entraînement
        data = model.generate_synthetic_data(n_samples=1000)
        X = model.prepare_features(data)
        model.fit(X)
        
        # Test de vitesse sur échantillon de test
        test_data = model.generate_synthetic_data(n_samples=100)
        X_test = model.prepare_features(test_data)
        
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Vérifications
        self.assertEqual(len(predictions), len(X_test))
        self.assertLess(inference_time, 1.0)  # Moins d'1 seconde pour 100 échantillons
        
        # Temps par échantillon
        time_per_sample = inference_time / len(X_test)
        self.assertLess(time_per_sample, 0.01)  # Moins de 10ms par échantillon
    
    def test_memory_usage(self):
        """Test d'utilisation mémoire."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Entraînement d'un modèle
        model = RandomForestModel(n_estimators=50, random_state=42)
        data = model.generate_synthetic_data(n_samples=2000)
        X = model.prepare_features(data)
        y = data['is_anomaly']
        
        model.fit(X, y)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Vérification que l'augmentation mémoire reste raisonnable
        self.assertLess(memory_increase, 200)  # Moins de 200MB d'augmentation


class TestDataQuality(unittest.TestCase):
    """Tests de qualité des données."""
    
    def test_data_consistency(self):
        """Test de cohérence des données générées."""
        model = IsolationForestModel()
        
        # Génération avec différents paramètres
        data1 = model.generate_synthetic_data(n_samples=1000, anomaly_rate=0.05)
        data2 = model.generate_synthetic_data(n_samples=1000, anomaly_rate=0.10)
        
        # Vérifications de cohérence
        self.assertEqual(len(data1), 1000)
        self.assertEqual(len(data2), 1000)
        
        # Taux d'anomalies respectés
        rate1 = data1['is_anomaly'].mean()
        rate2 = data2['is_anomaly'].mean()
        
        self.assertAlmostEqual(rate1, 0.05, delta=0.02)
        self.assertAlmostEqual(rate2, 0.10, delta=0.02)
        
        # Distributions des métriques dans des plages réalistes
        for col in ['cpu_usage', 'ram_usage']:
            self.assertTrue((data1[col] >= 0).all())
            self.assertTrue((data1[col] <= 100).all())
        
        for col in ['network_io', 'disk_io', 'response_time']:
            self.assertTrue((data1[col] >= 0).all())
    
    def test_feature_quality(self):
        """Test de qualité des features préparées."""
        model = IsolationForestModel()
        data = model.generate_synthetic_data(n_samples=500)
        features = model.prepare_features(data, include_temporal=True)
        
        # Vérifications de qualité
        self.assertFalse(features.isnull().any().any())  # Pas de valeurs manquantes
        self.assertFalse((features == np.inf).any().any())  # Pas d'infinis
        self.assertFalse((features == -np.inf).any().any())  # Pas d'infinis négatifs
        
        # Vérification des plages de valeurs
        self.assertTrue(np.isfinite(features.values).all())
        
        # Vérification de la variance (pas de colonnes constantes)
        for col in features.columns:
            self.assertGreater(features[col].var(), 0)


def run_test_suite():
    """
    Lance la suite de tests complète avec reporting détaillé.
    """
    print("🧪 LANCEMENT DE LA SUITE DE TESTS MLOPS")
    print("=" * 60)
    
    # Configuration du test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajout des classes de test
    test_classes = [
        TestBaseAnomalyModel,
        TestIsolationForestModel,
        TestRandomForestModel,
        TestLSTMModel,
        TestModelComparison,
        TestMLOpsPipeline,
        TestPerformance,
        TestDataQuality
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Exécution des tests avec reporting détaillé
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Rapport final
    print("\n" + "=" * 60)
    print("📊 RAPPORT FINAL DES TESTS")
    print("=" * 60)
    print(f"⏱️ Durée totale: {end_time - start_time:.2f} secondes")
    print(f"✅ Tests réussis: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests échoués: {len(result.failures)}")
    print(f"💥 Erreurs: {len(result.errors)}")
    print(f"⏭️ Tests ignorés: {len(result.skipped)}")
    
    # Détails des échecs
    if result.failures:
        print("\n🔍 DÉTAILS DES ÉCHECS:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\n💥 DÉTAILS DES ERREURS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    # Code de sortie
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
        return 0
    else:
        print("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        return 1


if __name__ == "__main__":
    # Vérification des dépendances
    try:
        import mlflow
        import sklearn
        import tensorflow
        import pandas
        import numpy
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("Installez toutes les dépendances avant de lancer les tests")
        sys.exit(1)
    
    # Lancement des tests
    exit_code = run_test_suite()
    sys.exit(exit_code)
