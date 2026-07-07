from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.config import MODEL_PATH, PREPROCESSOR_PATH, PLOTS_DIR, REPORT_PATH, SPLITS_PATH


sns.set_theme(style="whitegrid")


def evaluate_saved_model() -> dict:
    """Load the stored split and model, then generate evaluation outputs."""
    bundle = joblib.load(SPLITS_PATH)
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    x_test = bundle["x_test"]
    y_test = bundle["y_test"]

    x_test_matrix = preprocessor.transform(x_test)
    predictions = model.predict(x_test_matrix)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    report = classification_report(y_test, predictions, target_names=["ham", "spam"])
    matrix = confusion_matrix(y_test, predictions)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    REPORT_PATH.write_text(
        "Evaluation metrics\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1 score: {metrics['f1']:.4f}\n\n"
        f"Classification report:\n{report}\n",
        encoding="utf-8",
    )

    print("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.title()}: {metric_value:.4f}")
    print("\nClassification report:\n", report)
    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
