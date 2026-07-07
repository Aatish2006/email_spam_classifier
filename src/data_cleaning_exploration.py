from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CLEANED_DATA_PATH, PLOTS_DIR, RAW_DATA_PATH


sns.set_theme(style="whitegrid")


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw SMS spam dataset and keep only the meaningful columns."""
    raw_frame = pd.read_csv(path, encoding="latin-1")
    raw_frame = raw_frame.iloc[:, :2].copy()
    raw_frame.columns = ["label", "message"]
    return raw_frame


def clean_text(value: str) -> str:
    """Normalize a single message so the later feature pipeline sees consistent text."""
    text = str(value).lower().strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text


def clean_dataset(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, drop missing rows, and standardize the text column."""
    cleaned_frame = raw_frame.dropna(subset=["label", "message"]).copy()
    cleaned_frame["label"] = cleaned_frame["label"].astype(str).str.strip().str.lower()
    cleaned_frame["message"] = cleaned_frame["message"].map(clean_text)
    cleaned_frame = cleaned_frame[cleaned_frame["label"].isin(["ham", "spam"])]
    cleaned_frame = cleaned_frame.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
    return cleaned_frame


def run_exploration(cleaned_frame: pd.DataFrame) -> None:
    """Create quick EDA outputs for the submission evidence."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = cleaned_frame["label"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar(class_counts.index, class_counts.values, color=["#2c7fb8", "#7fcdbb"])
    plt.title("Spam vs Ham Distribution")
    plt.xlabel("Class")
    plt.ylabel("Message Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=200)
    plt.close()

    text_lengths = cleaned_frame["message"].str.len()
    plt.figure(figsize=(7, 4))
    sns.histplot(text_lengths, bins=30, kde=True, color="#1f77b4")
    plt.title("Message Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "message_length_distribution.png", dpi=200)
    plt.close()

    print("Class counts:\n", class_counts.to_string())
    print("\nMessage length summary:\n", text_lengths.describe().to_string())


def main() -> None:
    raw_frame = load_raw_dataset()
    cleaned_frame = clean_dataset(raw_frame)
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_frame.to_csv(CLEANED_DATA_PATH, index=False)
    run_exploration(cleaned_frame)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")


if __name__ == "__main__":
    main()
