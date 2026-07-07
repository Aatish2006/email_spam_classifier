from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import CLEANED_DATA_PATH, MODEL_PATH, REPORT_PATH
from src.data_cleaning_exploration import load_raw_dataset, clean_dataset
from src.feature_engineering import engineer_features


def train_random_forest():
    """Train a Random Forest model on the engineered features."""
    if CLEANED_DATA_PATH.exists():
        import pandas as pd

        cleaned_frame = pd.read_csv(CLEANED_DATA_PATH)
    else:
        cleaned_frame = clean_dataset(load_raw_dataset())
        CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned_frame.to_csv(CLEANED_DATA_PATH, index=False)

    features = engineer_features(cleaned_frame)
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(features.x_train, features.y_train)

    predictions = model.predict(features.x_test)
    report = classification_report(features.y_test, predictions, target_names=["ham", "spam"])
    accuracy = accuracy_score(features.y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        f"Training accuracy: {accuracy:.4f}\n\nClassification report:\n{report}\n",
        encoding="utf-8",
    )

    print(f"Model saved to {MODEL_PATH}")
    print(f"Training accuracy: {accuracy:.4f}")
    print(report)
    return model


if __name__ == "__main__":
    train_random_forest()
