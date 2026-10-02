"""
SmartTriage Production Drift Monitoring Engine.
Monitors incoming inference traffic for:
1. Token length distribution shift (Kolmogorov-Smirnov test)
2. Out-Of-Vocabulary (OOV) token drift rate
3. Target class distribution shift
"""

from pathlib import Path
import json
import pandas as pd
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
MODEL_REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"
DRIFT_REPORT_FILE = MODEL_REGISTRY_DIR / "drift_report.json"


class DriftDetector:
    def __init__(self, reference_data_path: Path = TRAIN_FILE):
        self.ref_df = pd.read_csv(reference_data_path)
        self.ref_tokens = set(" ".join(self.ref_df["text"]).lower().split())
        self.ref_lengths = self.ref_df["text"].apply(lambda t: len(t.split())).values
        self.ref_class_dist = self.ref_df["category"].value_counts(normalize=True).to_dict()

    def check_drift(self, current_texts: list[str], current_predictions: list[str]) -> dict:
        """
        Runs statistical tests to detect data and prediction drift.
        """
        current_lengths = [len(t.split()) for t in current_texts]
        current_tokens = set(" ".join(current_texts).lower().split())

        # 1. Two-sample Kolmogorov-Smirnov test on sequence lengths
        ks_stat, ks_pvalue = stats.ks_2samp(self.ref_lengths, current_lengths)
        length_drift = bool(ks_pvalue < 0.05)

        # 2. Out-of-Vocabulary (OOV) Token Ratio
        unseen_tokens = current_tokens - self.ref_tokens
        oov_rate = len(unseen_tokens) / max(len(current_tokens), 1)
        oov_drift = bool(oov_rate > 0.20)  # Alert if >20% tokens are brand new

        # 3. Prediction Distribution Comparison
        curr_series = pd.Series(current_predictions)
        curr_dist = curr_series.value_counts(normalize=True).to_dict()

        # Overall Status
        status = "NO_DRIFT"
        if length_drift or oov_drift:
            status = "MODERATE_DRIFT"
        if length_drift and oov_drift:
            status = "CRITICAL_DRIFT"

        report = {
            "overall_status": status,
            "sample_size": len(current_texts),
            "length_drift": {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(ks_pvalue), 4),
                "drift_detected": length_drift
            },
            "vocabulary_drift": {
                "oov_rate": round(float(oov_rate), 4),
                "unseen_token_count": len(unseen_tokens),
                "drift_detected": oov_drift
            },
            "class_distribution_comparison": {
                "reference": {k: round(v, 4) for k, v in self.ref_class_dist.items()},
                "current": {k: round(v, 4) for k, v in curr_dist.items()}
            }
        }

        # Persist report
        MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(DRIFT_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


def main():
    detector = DriftDetector()

    # Simulated production batch: issues are normal
    normal_batch_texts = [
        "NullPointerException on cart checkout when user session expires",
        "Add support for export to CSV format on billing transactions",
        "Slow SQL query in user notifications table taking over 3 seconds"
    ]
    normal_batch_preds = ["bug", "feature", "performance"]

    print("--- Running Drift Check on Production Batch ---")
    report = detector.check_drift(normal_batch_texts, normal_batch_preds)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Length Drift (p-value): {report['length_drift']['p_value']}")
    print(f"OOV Token Rate: {report['vocabulary_drift']['oov_rate'] * 100:.1f}%")
    print(f"Report written to: {DRIFT_REPORT_FILE}")


if __name__ == "__main__":
    main()
