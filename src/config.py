from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
RAW_DATA_PATH = DATA_DIR / "spam.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEANED_DATA_PATH = ARTIFACT_DIR / "cleaned" / "spam_cleaned.csv"
MODEL_PATH = ARTIFACT_DIR / "models" / "random_forest_spam_model.joblib"
PREPROCESSOR_PATH = ARTIFACT_DIR / "models" / "spam_preprocessor.joblib"
SPLITS_PATH = ARTIFACT_DIR / "splits" / "spam_split.joblib"
REPORT_PATH = ARTIFACT_DIR / "reports" / "evaluation_report.txt"
PLOTS_DIR = OUTPUT_DIR / "plots"
SAMPLES_PATH = OUTPUT_DIR / "sample_predictions.csv"
