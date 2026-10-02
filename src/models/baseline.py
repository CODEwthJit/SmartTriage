"""
SmartTriage Classical Machine Learning Baseline Module.
Responsible for TF-IDF feature extraction, training benchmark classifiers
(Multinomial Naive Bayes & Logistic Regression), evaluating performance,
and persisting the best baseline model to models/registry/baseline_pipeline.joblib.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"
MODEL_REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"
BASELINE_MODEL_FILE = MODEL_REGISTRY_DIR / "baseline_pipeline.joblib"
BASELINE_METRICS_FILE = MODEL_REGISTRY_DIR / "baseline_metrics.json"


def load_data():
    """Loads processed train and validation datasets and metadata contract."""
    print("Loading processed datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    X_train = train_df["text"].astype(str)
    y_train = train_df["category"].astype(str)

    X_val = val_df["text"].astype(str)
    y_val = val_df["category"].astype(str)

    return X_train, y_train, X_val, y_val, metadata


def train_and_evaluate_baselines():
    """
    Trains and compares Multinomial Naive Bayes and Logistic Regression
    on TF-IDF representations of the training data.
    """
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val, metadata = load_data()

    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")
    target_names = list(metadata["category_to_id"].keys())

    # Candidate 1: Multinomial Naive Bayes Pipeline
    nb_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=1.0)),
        ]
    )

    # Candidate 2: Balanced Multinomial Logistic Regression Pipeline
    lr_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )

    candidates = {"Multinomial_Naive_Bayes": nb_pipeline, "Logistic_Regression_Balanced": lr_pipeline}

    results = {}
    best_model_name = None
    best_macro_f1 = -1.0
    best_pipeline = None

    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING & EVALUATION BENCHMARK")
    print("=" * 60)

    for name, pipeline in candidates.items():
        print(f"\n--- Fitting {name} ---")
        pipeline.fit(X_train, y_train)

        # Generate predictions on unseen validation data
        y_val_pred = pipeline.predict(X_val)

        # Compute core evaluation metrics
        acc = float(accuracy_score(y_val, y_val_pred))
        macro_f1 = float(f1_score(y_val, y_val_pred, average="macro"))
        weighted_f1 = float(f1_score(y_val, y_val_pred, average="weighted"))
        report = classification_report(y_val, y_val_pred, target_names=target_names, output_dict=True)
        conf_matrix = confusion_matrix(y_val, y_val_pred, labels=target_names).tolist()

        print(f"Accuracy:    {acc:.4f}")
        print(f"Macro-F1:    {macro_f1:.4f}  (Equal weight to all classes)")
        print(f"Weighted-F1: {weighted_f1:.4f}  (Weighted by class prevalence)")
        print("\nPer-class Metrics:")
        for label in target_names:
            p = report[label]["precision"]
            r = report[label]["recall"]
            f = report[label]["f1-score"]
            supp = int(report[label]["support"])
            print(f"  [{label:<13}] Prec: {p:.3f} | Rec: {r:.3f} | F1: {f:.3f} | Support: {supp}")

        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": conf_matrix,
            "classification_report": report,
        }

        # Track the model with the highest Macro-F1 score
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name
            best_pipeline = pipeline

    # Persist the winning pipeline and metrics
    print("\n" + "=" * 60)
    print(f"WINNING BASELINE: {best_model_name} (Macro-F1: {best_macro_f1:.4f})")
    print("=" * 60)

    print(f"Persisting fitted pipeline to: {BASELINE_MODEL_FILE}")
    joblib.dump(best_pipeline, BASELINE_MODEL_FILE)

    metrics_payload = {
        "best_model": best_model_name,
        "best_macro_f1": best_macro_f1,
        "vocabulary_size": len(best_pipeline.named_steps["tfidf"].vocabulary_),
        "benchmark_results": results,
    }

    with open(BASELINE_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved benchmark metrics to: {BASELINE_METRICS_FILE}")
    return metrics_payload


def main():
    train_and_evaluate_baselines()


if __name__ == "__main__":
    main()
