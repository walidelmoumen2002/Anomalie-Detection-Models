"""
Main Orchestrator for MLOps Anomaly Detection Pipeline
======================================================

Script principal pour orchestrer l'ensemble de la pipeline MLOps.
Intègre tous les modèles et automatise le processus complet de bout en bout.

Fonctionnalités:
- Orchestration complète des modèles
- Entraînement automatisé
- Sélection du meilleur modèle
- Déploiement en production
- Monitoring continu
- Interface CLI simple

Author: MLOps Team
Date: June 2025
Version: 1.0
"""

import os
import sys
import argparse
import json
import pickle
from datetime import datetime
import logging
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

# Add path to core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

# Imports des modèles
from isolation_forest import IsolationForestModel
from random_forest import RandomForestModel
from lstm_model import LSTMModel
from model_comparison import ModelComparator

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MLOpsPipeline:
    """
    Pipeline MLOps complète pour la détection d'anomalies.
    
    Cette classe orchestrate l'ensemble du processus :
    - Préparation des données
    - Entraînement des modèles
    - Sélection du meilleur modèle
    - Déploiement
    - Monitoring
    """
    
    def __init__(self, config_path="config.json", model_dir="models", random_state=42):
        """
        Initialise la pipeline MLOps.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration
            model_dir (str): Répertoire pour sauvegarder les modèles
            random_state (int): Graine aléatoire
        """
        self.config_path = config_path
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        
        # Création du répertoire des modèles
        self.model_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = self._load_config()
        
        # Modèles disponibles
        self.available_models = {
            'isolation_forest': IsolationForestModel,
            'random_forest': RandomForestModel,
            'lstm': LSTMModel
        }
        
        # État de la pipeline
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.performance_metrics = {}
        
        # Configuration MLflow
        mlflow.set_experiment(self.config.get('mlflow_experiment', 'MLOps-Production'))
        
        logger.info(f"Pipeline MLOps initialisée - Répertoire modèles: {self.model_dir}")
    
    def _load_config(self):
        """
        Charge la configuration depuis le fichier JSON.
        
        Returns:
            dict: Configuration chargée
        """
        default_config = {
            "data": {
                "n_samples": 10000,
                "anomaly_rate": 0.05,
                "test_size": 0.2,
                "include_temporal": True
            },
            "models": {
                "isolation_forest": {
                    "contamination": 0.05,
                    "n_estimators": 100
                },
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 15,
                    "class_weight": "balanced"
                },
                "lstm": {
                    "sequence_length": 10,
                    "lstm_units": [50, 25],
                    "epochs": 30,
                    "dropout_rate": 0.2
                }
            },
            "evaluation": {
                "primary_metric": "f1_score",
                "min_accuracy": 0.8,
                "min_precision": 0.75,
                "min_recall": 0.75
            },
            "deployment": {
                "model_registry": "mlflow",
                "stage": "staging",
                "auto_promote": False
            },
            "mlflow_experiment": "MLOps-Production",
            "random_state": 42
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Fusion avec la configuration par défaut
                default_config.update(loaded_config)
                logger.info(f"Configuration chargée depuis {self.config_path}")
            except Exception as e:
                logger.warning(f"Erreur chargement config: {e}. Utilisation config par défaut.")
        else:
            # Création du fichier de configuration par défaut
            self._save_config(default_config)
            logger.info(f"Configuration par défaut créée: {self.config_path}")
        
        return default_config
    
    def _save_config(self, config):
        """
        Sauvegarde la configuration dans un fichier JSON.
        
        Args:
            config (dict): Configuration à sauvegarder
        """
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def generate_or_load_data(self, data_path=None):
        """
        Génère ou charge les données pour l'entraînement.
        
        Args:
            data_path (str): Chemin vers un fichier de données existant
        
        Returns:
            tuple: (X, y, data_info)
        """
        logger.info("🔄 Préparation des données...")
        
        if data_path and os.path.exists(data_path):
            logger.info(f"Chargement des données depuis {data_path}")
            data = pd.read_csv(data_path)
            
            # Validation des colonnes requises
            required_cols = ['cpu_usage', 'ram_usage', 'network_io', 'disk_io', 'response_time', 'is_anomaly']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans les données: {missing_cols}")
            
            data_info = {
                'source': 'file',
                'path': data_path,
                'n_samples': len(data),
                'anomaly_rate': data['is_anomaly'].mean()
            }
        else:
            logger.info("Génération de données synthétiques...")
            # Génération de données synthétiques
            temp_model = IsolationForestModel()
            data = temp_model.generate_synthetic_data(
                n_samples=self.config['data']['n_samples'],
                anomaly_rate=self.config['data']['anomaly_rate']
            )
            
            data_info = {
                'source': 'synthetic',
                'n_samples': len(data),
                'anomaly_rate': data['is_anomaly'].mean()
            }
        
        # Préparation des features
        base_model = IsolationForestModel()
        X = base_model.prepare_features(
            data, 
            include_temporal=self.config['data']['include_temporal']
        )
        y = data['is_anomaly']
        
        logger.info(f"✅ Données préparées: {X.shape[0]} échantillons, {X.shape[1]} features")
        logger.info(f"📊 Taux d'anomalies: {data_info['anomaly_rate']:.1%}")
        
        return X, y, data_info
    
    def train_all_models(self, X, y):
        """
        Entraîne tous les modèles configurés.
        
        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
        
        Returns:
            dict: Résultats d'entraînement
        """
        logger.info("🚀 Entraînement de tous les modèles...")
        
        # Division des données
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.random_state,
            stratify=y
        )
        
        training_results = {}
        
        for model_name, model_class in self.available_models.items():
            if model_name in self.config['models']:
                logger.info(f"📦 Entraînement {model_name}...")
                
                try:
                    # Création du modèle avec configuration
                    model_config = self.config['models'][model_name].copy()
                    model_config['random_state'] = self.random_state
                    
                    model = model_class(**model_config)
                    
                    # Entraînement
                    start_time = datetime.now()
                    
                    if model_name == 'isolation_forest':
                        # Non-supervisé
                        metrics = model.fit(X_train)
                    else:
                        # Supervisé
                        metrics = model.fit(X_train, y_train)
                    
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Évaluation
                    eval_metrics = model.evaluate(X_test, y_test)
                    
                    # Sauvegarde du modèle
                    model_path = self.model_dir / f"{model_name}_model.pkl"
                    model.save_model(str(model_path))
                    
                    # Stockage des résultats
                    self.trained_models[model_name] = model
                    training_results[model_name] = {
                        'model': model,
                        'training_metrics': metrics,
                        'evaluation_metrics': eval_metrics,
                        'training_time': training_time,
                        'model_path': str(model_path),
                        'success': True
                    }
                    
                    logger.info(f"✅ {model_name} entraîné - "
                               f"Précision: {eval_metrics.get('precision', 'N/A'):.3f}, "
                               f"Temps: {training_time:.1f}s")
                
                except Exception as e:
                    logger.error(f"❌ Erreur entraînement {model_name}: {e}")
                    training_results[model_name] = {
                        'success': False,
                        'error': str(e)
                    }
        
        # Sauvegarde des données de test pour évaluation ultérieure
        test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'X_train': X_train,
            'y_train': y_train
        }
        
        test_data_path = self.model_dir / "test_data.pkl"
        with open(test_data_path, 'wb') as f:
            pickle.dump(test_data, f)
        
        logger.info(f"📊 Entraînement terminé - {len([r for r in training_results.values() if r.get('success')])} modèles réussis")
        
        return training_results
    
    def select_best_model(self, training_results):
        """
        Sélectionne le meilleur modèle basé sur les métriques.
        
        Args:
            training_results (dict): Résultats d'entraînement
        
        Returns:
            tuple: (best_model_name, best_model, selection_reason)
        """
        logger.info("🏆 Sélection du meilleur modèle...")
        
        # Extraction des métriques de performance
        model_scores = {}
        primary_metric = self.config['evaluation']['primary_metric']
        
        for model_name, results in training_results.items():
            if results.get('success'):
                eval_metrics = results['evaluation_metrics']
                
                # Score composite basé sur plusieurs métriques
                accuracy = eval_metrics.get('accuracy', 0)
                precision = eval_metrics.get('precision', 0)
                recall = eval_metrics.get('recall', 0)
                f1_score = eval_metrics.get('f1_score', 0)
                
                # Score composite pondéré
                composite_score = (
                    0.3 * accuracy +
                    0.25 * precision +
                    0.25 * recall +
                    0.2 * f1_score
                )
                
                model_scores[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'composite_score': composite_score,
                    'primary_metric_value': eval_metrics.get(primary_metric, 0)
                }
        
        if not model_scores:
            logger.error("❌ Aucun modèle disponible pour la sélection")
            return None, None, "Aucun modèle entraîné avec succès"
        
        # Sélection basée sur la métrique primaire
        best_model_name = max(model_scores.keys(), 
                             key=lambda x: model_scores[x]['primary_metric_value'])
        
        best_model = training_results[best_model_name]['model']
        best_metrics = model_scores[best_model_name]
        
        # Vérification des seuils minimaux
        min_accuracy = self.config['evaluation']['min_accuracy']
        min_precision = self.config['evaluation']['min_precision']
        min_recall = self.config['evaluation']['min_recall']
        
        meets_requirements = (
            best_metrics['accuracy'] >= min_accuracy and
            best_metrics['precision'] >= min_precision and
            best_metrics['recall'] >= min_recall
        )
        
        selection_reason = f"Meilleur {primary_metric}: {best_metrics['primary_metric_value']:.3f}"
        
        if not meets_requirements:
            selection_reason += " (⚠️ Ne respecte pas tous les seuils minimaux)"
            logger.warning("⚠️ Le meilleur modèle ne respecte pas tous les seuils minimaux")
        
        # Stockage du meilleur modèle
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.performance_metrics = best_metrics
        
        logger.info(f"🏆 Meilleur modèle sélectionné: {best_model_name}")
        logger.info(f"📊 {selection_reason}")
        logger.info(f"🎯 Métriques: Précision={best_metrics['precision']:.3f}, "
                   f"Rappel={best_metrics['recall']:.3f}, F1={best_metrics['f1_score']:.3f}")
        
        return best_model_name, best_model, selection_reason
    
    def deploy_model(self, model_name, model, stage="staging"):
        """
        Déploie le modèle sélectionné vers MLflow Model Registry.
        
        Args:
            model_name (str): Nom du modèle
            model (BaseAnomalyModel): Instance du modèle
            stage (str): Stage de déploiement ("staging" ou "production")
        
        Returns:
            dict: Informations de déploiement
        """
        logger.info(f"🚀 Déploiement du modèle {model_name} en {stage}...")
        
        try:
            # Enregistrement dans MLflow
            with mlflow.start_run(run_name=f"Production_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log des paramètres du modèle
                model_params = {
                    'model_type': model.model_type,
                    'model_name': model.model_name,
                    'is_fitted': model.is_fitted
                }
                
                if hasattr(model, 'n_estimators'):
                    model_params['n_estimators'] = model.n_estimators
                if hasattr(model, 'contamination'):
                    model_params['contamination'] = model.contamination
                if hasattr(model, 'sequence_length'):
                    model_params['sequence_length'] = model.sequence_length
                
                mlflow.log_params(model_params)
                
                # Log des métriques de performance
                mlflow.log_metrics(self.performance_metrics)
                
                # Enregistrement du modèle
                if model_name == 'lstm':
                    mlflow.tensorflow.log_model(
                        model.model, 
                        f"{model_name}_model",
                        registered_model_name=f"AnomalyDetector_{model_name}"
                    )
                else:
                    mlflow.sklearn.log_model(
                        model.model,
                        f"{model_name}_model", 
                        registered_model_name=f"AnomalyDetector_{model_name}"
                    )
                
                # Log des artefacts additionnels
                if model.scaler:
                    scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(model.scaler, f)
                    mlflow.log_artifact(str(scaler_path))
                
                run_id = mlflow.active_run().info.run_id
            
            deployment_info = {
                'model_name': model_name,
                'stage': stage,
                'run_id': run_id,
                'deployment_time': datetime.now().isoformat(),
                'performance_metrics': self.performance_metrics,
                'success': True
            }
            
            logger.info(f"✅ Modèle déployé avec succès - Run ID: {run_id}")
            
            return deployment_info
        
        except Exception as e:
            logger.error(f"❌ Erreur déploiement: {e}")
            return {
                'model_name': model_name,
                'stage': stage,
                'success': False,
                'error': str(e)
            }
    
    def run_full_pipeline(self, data_path=None, deploy=False):
        """
        Exécute la pipeline complète de bout en bout.
        
        Args:
            data_path (str): Chemin vers les données (optionnel)
            deploy (bool): Déployer le meilleur modèle
        
        Returns:
            dict: Résultats complets de la pipeline
        """
        logger.info("🌟 DÉMARRAGE DE LA PIPELINE MLOPS COMPLÈTE")
        logger.info("=" * 60)
        
        pipeline_start = datetime.now()
        
        try:
            # 1. Préparation des données
            X, y, data_info = self.generate_or_load_data(data_path)
            
            # 2. Entraînement des modèles
            training_results = self.train_all_models(X, y)
            
            # 3. Sélection du meilleur modèle
            best_model_name, best_model, selection_reason = self.select_best_model(training_results)
            
            if not best_model:
                raise Exception("Aucun modèle valide trouvé")
            
            # 4. Déploiement (optionnel)
            deployment_info = None
            if deploy:
                deployment_info = self.deploy_model(
                    best_model_name, 
                    best_model,
                    stage=self.config['deployment']['stage']
                )
            
            # 5. Génération du rapport final
            pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
            
            final_report = {
                'pipeline_execution': {
                    'start_time': pipeline_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': pipeline_duration,
                    'success': True
                },
                'data_info': data_info,
                'training_results': {
                    name: {
                        'success': result['success'],
                        'training_time': result.get('training_time', 0),
                        'evaluation_metrics': result.get('evaluation_metrics', {})
                    }
                    for name, result in training_results.items()
                },
                'best_model': {
                    'name': best_model_name,
                    'selection_reason': selection_reason,
                    'performance_metrics': self.performance_metrics
                },
                'deployment': deployment_info,
                'config': self.config
            }
            
            # Sauvegarde du rapport
            report_path = self.model_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            logger.info("🎉 PIPELINE TERMINÉE AVEC SUCCÈS!")
            logger.info(f"⏱️ Durée totale: {pipeline_duration:.1f} secondes")
            logger.info(f"🏆 Meilleur modèle: {best_model_name}")
            logger.info(f"📊 Performance: {self.performance_metrics['f1_score']:.3f} F1-Score")
            logger.info(f"📄 Rapport sauvegardé: {report_path}")
            
            return final_report
        
        except Exception as e:
            logger.error(f"❌ ERREUR PIPELINE: {e}")
            
            error_report = {
                'pipeline_execution': {
                    'start_time': pipeline_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - pipeline_start).total_seconds(),
                    'success': False,
                    'error': str(e)
                }
            }
            
            return error_report
    
    def predict_anomalies(self, new_data, model_name=None):
        """
        Utilise le meilleur modèle pour prédire des anomalies.
        
        Args:
            new_data: Nouvelles données à analyser
            model_name (str): Nom du modèle à utiliser (par défaut: meilleur)
        
        Returns:
            dict: Prédictions et métadonnées
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("Aucun modèle entraîné disponible")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Modèle {model_name} non disponible")
            model = self.trained_models[model_name]
        
        logger.info(f"🔍 Prédiction d'anomalies avec {model_name}")
        
        # Préparation des features (même processus que l'entraînement)
        if isinstance(new_data, pd.DataFrame):
            X = model.prepare_features(new_data, include_temporal=self.config['data']['include_temporal'])
        else:
            X = new_data
        
        # Prédictions
        predictions = model.predict(X)
        
        # Métriques additionnelles si disponibles
        probabilities = None
        if hasattr(model, 'predict_probabilities'):
            probabilities = model.predict_probabilities(X)
        elif hasattr(model, 'get_anomaly_probabilities'):
            probabilities = model.get_anomaly_probabilities(X)
        
        results = {
            'model_used': model_name,
            'predictions': predictions,
            'probabilities': probabilities,
            'n_samples': len(predictions),
            'n_anomalies': np.sum(predictions),
            'anomaly_rate': np.mean(predictions),
            'prediction_time': datetime.now().isoformat()
        }
        
        logger.info(f"✅ {results['n_anomalies']} anomalies détectées sur {results['n_samples']} échantillons")
        
        return results


def create_cli():
    """
    Crée l'interface en ligne de commande.
    
    Returns:
        argparse.ArgumentParser: Parser CLI configuré
    """
    parser = argparse.ArgumentParser(
        description="Pipeline MLOps pour la détection d'anomalies IT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
  python main_orchestrator.py train --data data.csv --deploy
  python main_orchestrator.py compare --models all
  python main_orchestrator.py predict --input new_data.csv --model random_forest
  python main_orchestrator.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande train
    train_parser = subparsers.add_parser('train', help='Entraîner les modèles')
    train_parser.add_argument('--data', type=str, help='Chemin vers les données')
    train_parser.add_argument('--config', type=str, default='config.json', help='Fichier de configuration')
    train_parser.add_argument('--deploy', action='store_true', help='Déployer le meilleur modèle')
    train_parser.add_argument('--model-dir', type=str, default='models', help='Répertoire des modèles')
    
    # Commande compare
    compare_parser = subparsers.add_parser('compare', help='Comparer les modèles')
    compare_parser.add_argument('--models', type=str, default='all', help='Modèles à comparer')
    compare_parser.add_argument('--scenarios', type=str, nargs='+', help='Scénarios de test')
    
    # Commande predict
    predict_parser = subparsers.add_parser('predict', help='Prédire des anomalies')
    predict_parser.add_argument('--input', type=str, required=True, help='Fichier de données')
    predict_parser.add_argument('--model', type=str, help='Modèle à utiliser')
    predict_parser.add_argument('--output', type=str, help='Fichier de sortie')
    
    # Commande status
    status_parser = subparsers.add_parser('status', help='Statut de la pipeline')
    status_parser.add_argument('--model-dir', type=str, default='models', help='Répertoire des modèles')
    
    return parser


def main():
    """
    Fonction principale avec interface CLI.
    """
    parser = create_cli()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Configuration des chemins
    config_path = getattr(args, 'config', 'config.json')
    model_dir = getattr(args, 'model_dir', 'models')
    
    if args.command == 'train':
        # Entraînement complet
        pipeline = MLOpsPipeline(config_path=config_path, model_dir=model_dir)
        
        result = pipeline.run_full_pipeline(
            data_path=args.data,
            deploy=args.deploy
        )
        
        if result['pipeline_execution']['success']:
            print("🎉 Entraînement terminé avec succès!")
            print(f"🏆 Meilleur modèle: {result['best_model']['name']}")
        else:
            print("❌ Erreur lors de l'entraînement")
            sys.exit(1)
    
    elif args.command == 'compare':
        # Comparaison des modèles
        print("🔬 Comparaison des modèles...")
        comparator = ModelComparator()
        comparator.create_models()
        comparator.generate_test_scenarios()
        
        scenarios = args.scenarios if args.scenarios else None
        results = comparator.compare_models(scenarios)
        
        comparator.create_comparison_report()
        print("📊 Rapport de comparaison généré: model_comparison_report.html")
    
    elif args.command == 'predict':
        # Prédiction sur nouvelles données
        pipeline = MLOpsPipeline(config_path=config_path, model_dir=model_dir)
        
        # Chargement des modèles existants
        model_files = list(Path(model_dir).glob("*_model.pkl"))
        if not model_files:
            print("❌ Aucun modèle entraîné trouvé. Lancez d'abord l'entraînement.")
            sys.exit(1)
        
        # Chargement des données
        data = pd.read_csv(args.input)
        
        # Prédictions
        results = pipeline.predict_anomalies(data, model_name=args.model)
        
        print(f"🔍 {results['n_anomalies']} anomalies détectées")
        
        # Sauvegarde des résultats
        if args.output:
            output_data = data.copy()
            output_data['predicted_anomaly'] = results['predictions']
            if results['probabilities'] is not None:
                output_data['anomaly_probability'] = results['probabilities']
            
            output_data.to_csv(args.output, index=False)
            print(f"💾 Résultats sauvegardés: {args.output}")
    
    elif args.command == 'status':
        # Statut de la pipeline
        model_dir_path = Path(model_dir)
        
        print("📊 STATUT DE LA PIPELINE MLOPS")
        print("=" * 40)
        
        if model_dir_path.exists():
            model_files = list(model_dir_path.glob("*_model.pkl"))
            report_files = list(model_dir_path.glob("pipeline_report_*.json"))
            
            print(f"📁 Répertoire modèles: {model_dir_path}")
            print(f"🤖 Modèles entraînés: {len(model_files)}")
            
            for model_file in model_files:
                print(f"   - {model_file.name}")
            
            print(f"📄 Rapports disponibles: {len(report_files)}")
            
            if report_files:
                # Dernier rapport
                latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
                with open(latest_report) as f:
                    report = json.load(f)
                
                print(f"\n🏆 Dernier entraînement:")
                print(f"   Date: {report['pipeline_execution']['start_time']}")
                print(f"   Meilleur modèle: {report['best_model']['name']}")
                print(f"   F1-Score: {report['best_model']['performance_metrics']['f1_score']:.3f}")
        else:
            print("❌ Aucun modèle trouvé. Lancez l'entraînement d'abord.")


if __name__ == "__main__":
    main()
