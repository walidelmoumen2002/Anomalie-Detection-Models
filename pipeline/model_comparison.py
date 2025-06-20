"""
Model Comparison and Selection Tool
===================================

Script de comparaison complète des modèles de détection d'anomalies.
Compare Isolation Forest, Random Forest et LSTM sur différents critères.

Fonctionnalités:
- Comparaison des performances
- Analyse des temps d'exécution
- Évaluation sur différents types de données
- Recommandations d'usage
- Génération de rapports détaillés

Author: MLOps Team
Date: June 2025
Version: 1.0
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import mlflow
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add path to core modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

# Import des modèles
from isolation_forest import IsolationForestModel
from random_forest import RandomForestModel
from lstm_model import LSTMModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Classe pour comparer les différents modèles de détection d'anomalies.

    Cette classe permet de:
    - Entraîner tous les modèles sur les mêmes données
    - Comparer leurs performances
    - Analyser leurs caractéristiques
    - Générer des recommandations d'usage
    """

    def __init__(self, random_state=42):
        """
        Initialise le comparateur de modèles.

        Args:
            random_state (int): Graine aléatoire pour reproductibilité
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.data_scenarios = {}

        # Configuration MLflow
        mlflow.set_experiment("Model-Comparison")

        logger.info("Comparateur de modèles initialisé")

    def create_models(self):
        """
        Crée les instances de tous les modèles à comparer.
        """
        logger.info("Création des modèles...")

        # Isolation Forest (non-supervisé)
        self.models["IsolationForest"] = IsolationForestModel(
            contamination=0.05, n_estimators=100, random_state=self.random_state
        )

        # Random Forest (supervisé)
        self.models["RandomForest"] = RandomForestModel(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=self.random_state,
        )

        # LSTM (supervisé)
        self.models["LSTM"] = LSTMModel(
            sequence_length=10,
            lstm_units=[50, 25],
            dropout_rate=0.2,
            epochs=30,
            random_state=self.random_state,
        )

        logger.info(f"✅ {len(self.models)} modèles créés")

    def generate_test_scenarios(self):
        """
        Génère différents scénarios de données pour tester les modèles.
        """
        logger.info("Génération des scénarios de test...")

        # Scénario 1: Dataset standard
        model_temp = IsolationForestModel()
        data_standard = model_temp.generate_synthetic_data(
            n_samples=3000, anomaly_rate=0.05
        )
        self.data_scenarios["standard"] = {
            "data": data_standard,
            "description": "Dataset standard (5% anomalies)",
            "anomaly_rate": 0.05,
        }

        # Scénario 2: Haute charge d'anomalies
        data_high_anomaly = model_temp.generate_synthetic_data(
            n_samples=3000, anomaly_rate=0.15
        )
        self.data_scenarios["high_anomaly"] = {
            "data": data_high_anomaly,
            "description": "Haute charge d'anomalies (15%)",
            "anomaly_rate": 0.15,
        }

        # Scénario 3: Faible charge d'anomalies
        data_low_anomaly = model_temp.generate_synthetic_data(
            n_samples=3000, anomaly_rate=0.02
        )
        self.data_scenarios["low_anomaly"] = {
            "data": data_low_anomaly,
            "description": "Faible charge d'anomalies (2%)",
            "anomaly_rate": 0.02,
        }

        # Scénario 4: Dataset volumineux
        data_large = model_temp.generate_synthetic_data(
            n_samples=10000, anomaly_rate=0.05
        )
        self.data_scenarios["large_dataset"] = {
            "data": data_large,
            "description": "Dataset volumineux (10k échantillons)",
            "anomaly_rate": 0.05,
        }

        logger.info(f"✅ {len(self.data_scenarios)} scénarios générés")

    def train_all_models(self, scenario_name="standard"):
        """
        Entraîne tous les modèles sur un scénario donné.

        Args:
            scenario_name (str): Nom du scénario à utiliser
        """
        if scenario_name not in self.data_scenarios:
            raise ValueError(f"Scénario '{scenario_name}' non disponible")

        logger.info(f"Entraînement des modèles sur scénario: {scenario_name}")

        scenario = self.data_scenarios[scenario_name]
        data = scenario["data"]

        # Préparation des données communes
        base_model = IsolationForestModel()
        X = base_model.prepare_features(data, include_temporal=True)
        y = data["is_anomaly"]

        # Division train/test
        X_train, X_test, y_train, y_test = base_model.split_data(X, y, test_size=0.2)

        # Entraînement de chaque modèle
        training_results = {}

        for model_name, model in self.models.items():
            logger.info(f"Entraînement {model_name}...")

            start_time = time.time()

            try:
                if model_name == "IsolationForest":
                    # Isolation Forest (non-supervisé)
                    metrics = model.fit(X_train)
                else:
                    # Modèles supervisés
                    metrics = model.fit(X_train, y_train)

                training_time = time.time() - start_time

                training_results[model_name] = {
                    "success": True,
                    "training_time": training_time,
                    "metrics": metrics,
                }

                logger.info(f"✅ {model_name} entraîné en {training_time:.2f}s")

            except Exception as e:
                logger.error(f"❌ Erreur entraînement {model_name}: {e}")
                training_results[model_name] = {
                    "success": False,
                    "error": str(e),
                    "training_time": 0,
                }

        # Sauvegarde des résultats d'entraînement
        if scenario_name not in self.results:
            self.results[scenario_name] = {}

        self.results[scenario_name]["training"] = training_results
        self.results[scenario_name]["data_split"] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        return training_results

    def evaluate_all_models(self, scenario_name="standard"):
        """
        Évalue les performances de tous les modèles.

        Args:
            scenario_name (str): Nom du scénario à évaluer
        """
        if scenario_name not in self.results:
            raise ValueError(
                f"Modèles non entraînés pour le scénario '{scenario_name}'"
            )

        logger.info(f"Évaluation des modèles sur scénario: {scenario_name}")

        data_split = self.results[scenario_name]["data_split"]
        X_test, y_test = data_split["X_test"], data_split["y_test"]

        evaluation_results = {}

        for model_name, model in self.models.items():
            if not self.results[scenario_name]["training"][model_name]["success"]:
                logger.warning(f"Modèle {model_name} non disponible pour évaluation")
                continue

            logger.info(f"Évaluation {model_name}...")

            try:
                # Mesure du temps d'inférence
                start_time = time.time()
                predictions = model.predict(X_test)
                inference_time = time.time() - start_time

                # Calcul des métriques
                accuracy = (predictions == y_test).mean()

                # Métriques détaillées
                from sklearn.metrics import precision_score, recall_score, f1_score

                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                f1 = f1_score(y_test, predictions, zero_division=0)

                # AUC si probabilités disponibles
                auc = None
                if hasattr(model, "predict_proba") or hasattr(
                    model, "get_anomaly_probabilities"
                ):
                    try:
                        if model_name == "RandomForest":
                            probas = model.get_anomaly_probabilities(X_test)
                        elif model_name == "LSTM":
                            probas = model.predict_probabilities(X_test)
                        elif model_name == "IsolationForest":
                            probas = -model.get_anomaly_scores(
                                X_test
                            )  # Scores inversés

                        auc = roc_auc_score(y_test, probas)
                    except Exception as e:
                        logger.warning(
                            f"Impossible de calculer AUC pour {model_name}: {e}"
                        )

                # Matrice de confusion
                cm = confusion_matrix(y_test, predictions)

                evaluation_results[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auc_score": auc,
                    "inference_time": inference_time,
                    "inference_time_per_sample": inference_time / len(X_test),
                    "confusion_matrix": cm,
                    "n_test_samples": len(X_test),
                    "n_anomalies_predicted": np.sum(predictions),
                    "n_anomalies_actual": np.sum(y_test),
                }

                logger.info(
                    f"✅ {model_name} - Précision: {precision:.3f}, "
                    f"Rappel: {recall:.3f}, F1: {f1:.3f}"
                )

            except Exception as e:
                logger.error(f"❌ Erreur évaluation {model_name}: {e}")
                evaluation_results[model_name] = {"error": str(e)}

        self.results[scenario_name]["evaluation"] = evaluation_results
        return evaluation_results

    def compare_models(self, scenarios=None):
        """
        Compare tous les modèles sur tous les scénarios.

        Args:
            scenarios (list): Liste des scénarios à comparer (None = tous)
        """
        if scenarios is None:
            scenarios = list(self.data_scenarios.keys())

        logger.info(f"Comparaison complète sur {len(scenarios)} scénarios")

        comparison_results = {}

        for scenario in scenarios:
            logger.info(f"\n📊 Scénario: {scenario}")
            logger.info("-" * 40)

            # Entraînement
            training_results = self.train_all_models(scenario)

            # Évaluation
            evaluation_results = self.evaluate_all_models(scenario)

            comparison_results[scenario] = {
                "training": training_results,
                "evaluation": evaluation_results,
                "scenario_info": self.data_scenarios[scenario],
            }

        # Analyse globale
        self._analyze_global_performance(comparison_results)

        return comparison_results

    def _analyze_global_performance(self, comparison_results):
        """
        Analyse les performances globales et génère des recommandations.

        Args:
            comparison_results (dict): Résultats de comparaison
        """
        logger.info("\n🔍 ANALYSE GLOBALE DES PERFORMANCES")
        logger.info("=" * 50)

        # Agrégation des métriques
        model_stats = {}

        for model_name in self.models.keys():
            stats = {
                "scenarios_tested": 0,
                "avg_accuracy": [],
                "avg_precision": [],
                "avg_recall": [],
                "avg_f1": [],
                "avg_auc": [],
                "avg_training_time": [],
                "avg_inference_time": [],
            }

            for scenario, results in comparison_results.items():
                if model_name in results["evaluation"]:
                    eval_data = results["evaluation"][model_name]

                    if "error" not in eval_data:
                        stats["scenarios_tested"] += 1
                        stats["avg_accuracy"].append(eval_data["accuracy"])
                        stats["avg_precision"].append(eval_data["precision"])
                        stats["avg_recall"].append(eval_data["recall"])
                        stats["avg_f1"].append(eval_data["f1_score"])

                        if eval_data["auc_score"] is not None:
                            stats["avg_auc"].append(eval_data["auc_score"])

                        stats["avg_training_time"].append(
                            results["training"][model_name]["training_time"]
                        )
                        stats["avg_inference_time"].append(eval_data["inference_time"])

            # Calcul des moyennes
            for metric in [
                "avg_accuracy",
                "avg_precision",
                "avg_recall",
                "avg_f1",
                "avg_auc",
                "avg_training_time",
                "avg_inference_time",
            ]:
                if stats[metric]:
                    stats[metric] = np.mean(stats[metric])
                else:
                    stats[metric] = 0

            model_stats[model_name] = stats

        # Affichage des résultats
        print("\n📈 RÉSUMÉ DES PERFORMANCES MOYENNES:")
        print("-" * 60)

        metrics_to_show = [
            "avg_accuracy",
            "avg_precision",
            "avg_recall",
            "avg_f1",
            "avg_auc",
        ]

        for model_name, stats in model_stats.items():
            print(f"\n🤖 {model_name}:")
            for metric in metrics_to_show:
                value = stats[metric]
                if value > 0:
                    print(f"   {metric.replace('avg_', '').title()}: {value:.3f}")
            print(f"   Temps entraînement moyen: {stats['avg_training_time']:.2f}s")
            print(f"   Temps inférence moyen: {stats['avg_inference_time']:.4f}s")

        # Recommandations
        self._generate_recommendations(model_stats)

    def _generate_recommendations(self, model_stats):
        """
        Génère des recommandations d'usage basées sur les performances.

        Args:
            model_stats (dict): Statistiques des modèles
        """
        logger.info("\n💡 RECOMMANDATIONS D'USAGE")
        logger.info("=" * 40)

        # Classement par métrique
        rankings = {}

        for metric in [
            "avg_accuracy",
            "avg_precision",
            "avg_recall",
            "avg_f1",
            "avg_auc",
        ]:
            rankings[metric] = sorted(
                [(name, stats[metric]) for name, stats in model_stats.items()],
                key=lambda x: x[1],
                reverse=True,
            )

        # Recommandations spécifiques
        recommendations = {
            "Haute précision": rankings["avg_precision"][0][0],
            "Haute sensibilité": rankings["avg_recall"][0][0],
            "Équilibre F1": rankings["avg_f1"][0][0],
            "Performance globale": rankings["avg_accuracy"][0][0],
        }

        # Recommandations par cas d'usage
        print("\n🎯 RECOMMANDATIONS PAR CAS D'USAGE:")
        print("-" * 40)

        for use_case, best_model in recommendations.items():
            print(f"• {use_case}: {best_model}")

        # Recommandations techniques
        print("\n⚙️ RECOMMANDATIONS TECHNIQUES:")
        print("-" * 40)

        # Vitesse d'entraînement
        fastest_training = min(
            model_stats.items(), key=lambda x: x[1]["avg_training_time"]
        )
        print(
            f"• Entraînement le plus rapide: {fastest_training[0]} "
            f"({fastest_training[1]['avg_training_time']:.2f}s)"
        )

        # Vitesse d'inférence
        fastest_inference = min(
            model_stats.items(), key=lambda x: x[1]["avg_inference_time"]
        )
        print(
            f"• Inférence la plus rapide: {fastest_inference[0]} "
            f"({fastest_inference[1]['avg_inference_time']:.4f}s)"
        )

        # Recommandations contextuelles
        print("\n🔄 RECOMMANDATIONS CONTEXTUELLES:")
        print("-" * 40)
        print("• Production temps réel: RandomForest (équilibre vitesse/précision)")
        print("• Exploration de données: IsolationForest (non-supervisé)")
        print("• Analyse séquentielle: LSTM (patterns temporels)")
        print("• Environnement mixte: Ensemble des 3 modèles")

    def create_comparison_report(self, save_path="model_comparison_report.html"):
        """
        Crée un rapport HTML de comparaison des modèles.

        Args:
            save_path (str): Chemin de sauvegarde du rapport
        """
        logger.info("Génération du rapport de comparaison...")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport de Comparaison des Modèles</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .best {{ background-color: #d4edda; }}
                .good {{ background-color: #fff3cd; }}
                .poor {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Rapport de Comparaison des Modèles MLOps</h1>
                <p>Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Résumé Exécutif</h2>
                <p>Comparaison de 3 modèles de détection d'anomalies sur {len(self.data_scenarios)} scénarios.</p>
            </div>
        """

        # Ajout des résultats détaillés
        for scenario_name, scenario_info in self.data_scenarios.items():
            if scenario_name in self.results:
                html_content += f"""
                <div class="section">
                    <h3>🎯 Scénario: {scenario_name}</h3>
                    <p><strong>Description:</strong> {scenario_info['description']}</p>
                    <table>
                        <tr>
                            <th>Modèle</th>
                            <th>Précision</th>
                            <th>Rappel</th>
                            <th>F1-Score</th>
                            <th>Temps Entraînement</th>
                            <th>Temps Inférence</th>
                        </tr>
                """

                eval_results = self.results[scenario_name].get("evaluation", {})
                training_results = self.results[scenario_name].get("training", {})

                for model_name in self.models.keys():
                    if (
                        model_name in eval_results
                        and "error" not in eval_results[model_name]
                    ):
                        eval_data = eval_results[model_name]
                        train_data = training_results[model_name]

                        html_content += f"""
                        <tr>
                            <td class="metric">{model_name}</td>
                            <td>{eval_data['precision']:.3f}</td>
                            <td>{eval_data['recall']:.3f}</td>
                            <td>{eval_data['f1_score']:.3f}</td>
                            <td>{train_data['training_time']:.2f}s</td>
                            <td>{eval_data['inference_time']:.4f}s</td>
                        </tr>
                        """

                html_content += "</table></div>"

        html_content += """
            <div class="section">
                <h2>💡 Recommandations</h2>
                <ul>
                    <li><strong>Production temps réel:</strong> Random Forest (équilibre performance/vitesse)</li>
                    <li><strong>Exploration non-supervisée:</strong> Isolation Forest</li>
                    <li><strong>Analyse temporelle:</strong> LSTM (patterns séquentiels)</li>
                    <li><strong>Environnement hybride:</strong> Ensemble des modèles</li>
                </ul>
            </div>
        </body>
        </html>
        """

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"✅ Rapport sauvegardé: {save_path}")

    def plot_comparison_charts(self, save_path="model_comparison_charts.png"):
        """
        Crée des graphiques de comparaison des modèles.

        Args:
            save_path (str): Chemin de sauvegarde des graphiques
        """
        logger.info("Génération des graphiques de comparaison...")

        # Préparation des données pour les graphiques
        models_data = []

        for scenario_name, results in self.results.items():
            eval_results = results.get("evaluation", {})
            training_results = results.get("training", {})

            for model_name in self.models.keys():
                if (
                    model_name in eval_results
                    and "error" not in eval_results[model_name]
                ):
                    eval_data = eval_results[model_name]
                    train_data = training_results[model_name]

                    models_data.append(
                        {
                            "Model": model_name,
                            "Scenario": scenario_name,
                            "Accuracy": eval_data["accuracy"],
                            "Precision": eval_data["precision"],
                            "Recall": eval_data["recall"],
                            "F1_Score": eval_data["f1_score"],
                            "Training_Time": train_data["training_time"],
                            "Inference_Time": eval_data["inference_time"],
                        }
                    )

        if not models_data:
            logger.warning("Pas de données disponibles pour les graphiques")
            return

        df = pd.DataFrame(models_data)
        
        # Debug: Print data available for plotting
        logger.info(f"Données disponibles pour les graphiques: {len(models_data)} points")
        logger.info(f"Colonnes: {df.columns.tolist()}")
        logger.info(f"Modèles: {df['Model'].unique()}")
        logger.info(f"Exemple de données:\n{df.head()}")

        # Création des graphiques
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Comparaison des Modèles de Détection d'Anomalies", fontsize=16)

        # Use bar plots instead of box plots for small datasets
        plot_type = "bar" if len(df) <= 5 else "box"
        
        # Graphique 1: Précision par modèle
        if plot_type == "bar":
            sns.barplot(data=df, x="Model", y="Precision", ax=axes[0, 0], ci=None)
        else:
            sns.boxplot(data=df, x="Model", y="Precision", ax=axes[0, 0])
        axes[0, 0].set_title("Précision par Modèle")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].set_ylim(0, 1.1)

        # Graphique 2: Rappel par modèle
        if plot_type == "bar":
            sns.barplot(data=df, x="Model", y="Recall", ax=axes[0, 1], ci=None)
        else:
            sns.boxplot(data=df, x="Model", y="Recall", ax=axes[0, 1])
        axes[0, 1].set_title("Rappel par Modèle")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].set_ylim(0, 1.1)

        # Graphique 3: F1-Score par modèle
        if plot_type == "bar":
            sns.barplot(data=df, x="Model", y="F1_Score", ax=axes[0, 2], ci=None)
        else:
            sns.boxplot(data=df, x="Model", y="F1_Score", ax=axes[0, 2])
        axes[0, 2].set_title("F1-Score par Modèle")
        axes[0, 2].tick_params(axis="x", rotation=45)
        axes[0, 2].set_ylim(0, 1.1)

        # Graphique 4: Temps d'entraînement
        sns.barplot(data=df, x="Model", y="Training_Time", ax=axes[1, 0], ci=None)
        axes[1, 0].set_title("Temps d'Entraînement (s)")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].set_ylabel("Temps (secondes)")

        # Graphique 5: Temps d'inférence
        sns.barplot(data=df, x="Model", y="Inference_Time", ax=axes[1, 1], ci=None)
        axes[1, 1].set_title("Temps d'Inférence (s)")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].set_ylabel("Temps (secondes)")

        # Graphique 6: Performance globale (radar chart approximé)
        avg_metrics = df.groupby("Model")[
            ["Accuracy", "Precision", "Recall", "F1_Score"]
        ].mean()
        avg_metrics.plot(kind="bar", ax=axes[1, 2])
        axes[1, 2].set_title("Performance Globale Moyenne")
        axes[1, 2].tick_params(axis="x", rotation=45)
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Graphiques sauvegardés: {save_path}")
        plt.close()  # Close instead of show to avoid display issues


def main():
    """
    Fonction principale pour exécuter la comparaison complète.
    """
    print("🚀 COMPARAISON COMPLÈTE DES MODÈLES MLOPS")
    print("=" * 60)

    # Initialisation du comparateur
    comparator = ModelComparator(random_state=42)

    # Création des modèles
    comparator.create_models()

    # Génération des scénarios de test
    comparator.generate_test_scenarios()

    # Comparaison complète
    print("\n🔬 Démarrage de la comparaison...")
    results = comparator.compare_models()

    # Génération des rapports
    print("\n📊 Génération des rapports...")
    comparator.create_comparison_report()
    comparator.plot_comparison_charts()

    print("\n🎉 COMPARAISON TERMINÉE!")
    print("Consultez les fichiers générés:")
    print("- model_comparison_report.html")
    print("- model_comparison_charts.png")

    return comparator, results


if __name__ == "__main__":
    comparator, results = main()
