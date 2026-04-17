# Machine Failure Prediction with Random Forest

This project applies machine learning to industrial sensor data in order to predict machine failures.

## Goal

The goal of this project is to build a small and interpretable machine learning workflow for failure prediction based on industrial sensor measurements.

## Dataset

Source: AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

The dataset contains industrial machine sensor data with the following features:

- air temperature
- process temperature
- rotational speed
- torque
- tool wear
- product type

Target variable:

- machine failure

## Workflow

The project includes the following steps:

- loading the dataset from CSV
- preprocessing and cleaning the data
- encoding the categorical feature `type`
- splitting features and target
- training a Random Forest classifier
- evaluating the model with classification metrics
- exporting predictions and visual outputs

## Model

A Random Forest classifier is used because it works well on tabular data, is easy to apply, and provides interpretable feature importance values.

## Results

The model achieved the following results on the test set:

- Accuracy: 0.9815
- Precision: 0.8780
- Recall: 0.5294
- F1 Score: 0.6606

The model achieved strong precision and high overall accuracy, but recall remained moderate.  
This means that predicted failures are often correct, but a relevant share of actual failures is still missed.

## Why Accuracy Alone Is Not Enough

The dataset is imbalanced, which means machine failures are much rarer than normal cases.  
Because of that, accuracy alone is not a sufficient metric. Precision and recall are more important for understanding how well the model detects failures.

## Feature Importance

The most important predictors in this project were:

- torque
- rotational speed
- tool wear

This suggests that mechanical load related features contribute most strongly to failure prediction in this dataset.

## Outputs

The project generates the following outputs:

- `outputs/predictions.csv`
- `outputs/metrics.txt`
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png`

## Project Structure

```text
machine-failure-prediction/
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   └── create_data.py
│
├── outputs/
│   ├── metrics.txt
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── predictions.csv
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md


## How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python data/create_data.py
python main.py