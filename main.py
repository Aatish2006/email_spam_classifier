from src.data_cleaning_exploration import main as clean_and_explore
from src.evaluate_model import evaluate_saved_model
from src.test_model import run_sample_predictions
from src.train_model import train_random_forest


def run_pipeline() -> None:
    """Execute the project in the same order a report would describe it."""
    clean_and_explore()
    train_random_forest()
    evaluate_saved_model()
    run_sample_predictions()


if __name__ == "__main__":
    run_pipeline()
