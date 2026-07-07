from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PREPROCESSOR_PATH, SPLITS_PATH


class TextStatisticsTransformer(BaseEstimator, TransformerMixin):
    """Create lightweight numeric features from each message."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            messages = X.iloc[:, 0].astype(str).tolist()
        elif isinstance(X, pd.Series):
            messages = X.astype(str).tolist()
        else:
            messages = pd.Series(X).astype(str).tolist()

        features = []
        for message in messages:
            words = message.split()
            word_count = len(words)
            char_count = len(message)
            avg_word_length = (sum(len(word) for word in words) / word_count) if word_count else 0.0
            digit_count = sum(character.isdigit() for character in message)
            punctuation_count = sum(character in "!?.,;:" for character in message)
            features.append([
                char_count,
                word_count,
                avg_word_length,
                digit_count,
                punctuation_count,
            ])

        return np.asarray(features, dtype=float)


@dataclass
class FeatureBundle:
    """Store the matrices used by model training and evaluation."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def build_preprocessor() -> ColumnTransformer:
    """Create the reusable feature pipeline for the message column."""
    text_features = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words="english")),
        ("svd", TruncatedSVD(n_components=100, random_state=42)),
    ])

    numeric_features = Pipeline([
        ("statistics", TextStatisticsTransformer()),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer([
        ("text", text_features, "message"),
        ("stats", numeric_features, "message"),
    ])


def split_dataset(cleaned_frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split the cleaned dataset once so every stage uses the same test set."""
    label_map = {"ham": 0, "spam": 1}
    x = cleaned_frame[["message"]].copy()
    y = cleaned_frame["label"].map(label_map).astype(int).to_numpy()
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def engineer_features(cleaned_frame: pd.DataFrame) -> FeatureBundle:
    """Fit the preprocessing stack and convert text into model-ready arrays."""
    x_train, x_test, y_train, y_test = split_dataset(cleaned_frame)
    preprocessor = build_preprocessor()
    x_train_matrix = preprocessor.fit_transform(x_train)
    x_test_matrix = preprocessor.transform(x_test)

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }, SPLITS_PATH)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    return FeatureBundle(x_train_matrix, x_test_matrix, y_train, y_test)


def transform_messages(messages: list[str]) -> np.ndarray:
    """Load the saved preprocessor and transform new messages for inference."""
    import joblib

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    frame = pd.DataFrame({"message": messages})
    return preprocessor.transform(frame)
