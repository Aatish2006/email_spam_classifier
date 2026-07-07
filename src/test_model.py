from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODEL_PATH, PREPROCESSOR_PATH, SAMPLES_PATH


SAMPLE_MESSAGES = [
    "Congratulations, you have won a free prize. Call now to claim.",
    "Can we meet for lunch tomorrow?",
    "Urgent! Your account has been suspended, click this link now.",
    "I will send the report before 5 pm today.",
]


def run_sample_predictions() -> pd.DataFrame:
    """Run a few example predictions so the submission has visible test output."""
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    sample_frame = pd.DataFrame({"message": SAMPLE_MESSAGES})
    sample_matrix = preprocessor.transform(sample_frame)
    predictions = model.predict(sample_matrix)
    probabilities = model.predict_proba(sample_matrix)[:, 1]

    output_frame = pd.DataFrame(
        {
            "message": SAMPLE_MESSAGES,
            "predicted_label": ["spam" if prediction == 1 else "ham" for prediction in predictions],
            "spam_probability": probabilities,
        }
    )

    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(SAMPLES_PATH, index=False)
    print(output_frame.to_string(index=False))
    return output_frame


if __name__ == "__main__":
    run_sample_predictions()
