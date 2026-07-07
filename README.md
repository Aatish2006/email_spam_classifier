# Email Spam Classification Using Random Forest

## Project Details

| Field | Value |
| --- | --- |
| Intern ID | CITS4384_ |
| Full Name | Aatish Ayyapath |
| No. of days | 5 |
| Project Name | Email Spam Classification Using Random Forest |
| Project Scope | Build a complete email spam classifier with data cleaning, exploration, feature engineering, model training, evaluation, and testing. |

## Project Overview

This project classifies emails or SMS messages into two classes: `ham` and `spam`. The workflow is separated file by file so each stage stays clear and easy to review.

## GitHub Repository

https://github.com/Aatish2006/email_spam_classifier.git

## Folder Structure

```text
email_spam_classification/
├── Data/
│   └── spam.csv
├── main.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data_cleaning_exploration.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── test_model.py
├── artifacts/
│   ├── cleaned/
│   ├── models/
│   ├── reports/
│   └── splits/
└── outputs/
    ├── plots/
    └── sample_predictions.csv
```

## File-by-File Workflow

### 1. Data Cleaning and Exploration
Run `src/data_cleaning_exploration.py` to:
- Load the raw dataset
- Keep only the useful columns
- Remove missing values and duplicates
- Normalize the message text
- Save the cleaned dataset
- Generate basic exploratory plots

### 2. Feature Engineering
Run `src/feature_engineering.py` through the training pipeline to:
- Split the cleaned dataset into train and test sets
- Convert text into TF-IDF features
- Reduce text dimensionality with Truncated SVD
- Create extra message statistics such as character count and word count
- Save the fitted preprocessing pipeline

### 3. Training
Run `src/train_model.py` to:
- Train a Random Forest classifier
- Save the model artifact
- Write a training report

### 4. Evaluation
Run `src/evaluate_model.py` to:
- Load the saved test split
- Predict on the test set
- Print accuracy, precision, recall, and F1 score
- Save the confusion matrix image

### 5. Testing
Run `src/test_model.py` to:
- Predict on sample messages
- Save sample prediction output for submission evidence

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

### Full pipeline
```bash
python main.py
```

### Individual stages
```bash
python src/data_cleaning_exploration.py
python src/train_model.py
python src/evaluate_model.py
python src/test_model.py
```

## Output Files

The scripts generate these submission-friendly outputs:
- [outputs/plots/class_distribution.png](outputs/plots/class_distribution.png)
- [outputs/plots/message_length_distribution.png](outputs/plots/message_length_distribution.png)
- [outputs/plots/confusion_matrix.png](outputs/plots/confusion_matrix.png)
- [outputs/sample_predictions.csv](outputs/sample_predictions.csv)
- [artifacts/reports/evaluation_report.txt](artifacts/reports/evaluation_report.txt)

## Source Code

The complete source code is in the `src/` folder and `main.py`.

## README File

This README contains the project details, workflow, setup instructions, and output locations required for submission.

## Screenshots

The generated output images below can be used as submission screenshots:

![Class distribution](outputs/plots/class_distribution.png)

![Message length distribution](outputs/plots/message_length_distribution.png)

![Confusion matrix](outputs/plots/confusion_matrix.png)

## Output Images

The project already generates the required output images in `outputs/plots/`.

- [Class distribution](outputs/plots/class_distribution.png)
- [Message length distribution](outputs/plots/message_length_distribution.png)
- [Confusion matrix](outputs/plots/confusion_matrix.png)

## Documentation

This project is organized so each major step is isolated in its own file, making the pipeline easier to understand, grade, and maintain.


