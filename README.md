# Machine Failure Prediction

A machine learning pipeline that predicts industrial machine failures from sensor data using a Random Forest classifier.

Built to explore whether failure prediction on highly imbalanced data is reliable — and what metrics actually tell you the truth when 96.6% of samples are non-failures.

---

## Prerequisites — Data Validation

Before training, sensor data quality is validated using
[data-quality-pipeline](https://github.com/iTzRyuujin/data-quality-pipeline):

```bash
python data-quality-pipeline/analyze.py \
  --input data/raw/data.csv \
  --config data-quality-pipeline/config/ai4i_rules.yaml

# Only proceed if exit code is 0 (no critical issues)
python main.py
```

The AI4I dataset passes all quality checks — no missing values, all sensor readings within physically documented bounds.

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository
([doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C))

10,000 rows of industrial sensor data with 6 features:

| Feature | Unit | Distribution |
|---|---|---|
| Air temperature | K | N(300, 2) |
| Process temperature | K | Air temp + 10 K |
| Rotational speed | rpm | Derived from 2860 W |
| Torque | Nm | N(40, 10) |
| Tool wear | min | 0 – 253 |
| Product type | L/M/H | Categorical |

**Class imbalance:** 96.6% non-failure / 3.4% failure — accuracy alone is not a useful metric here.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | 98.15% |
| Precision | 87.8% |
| Recall | 52.94% |
| F1 Score | 66.06% |
| **AUC** | **96.69%** |

### Why AUC, not Accuracy?

A model that always predicts "no failure" achieves 96.6% accuracy — without learning anything. AUC measures whether the model can actually **distinguish failures from normal operation**, independent of any threshold choice. AUC = 0.967 means the model has strong separability even where the default threshold hurts recall.

The recall gap (52.94%) reflects a deliberate threshold tradeoff: at 0.5, the model is conservative. Lowering the threshold increases recall at the cost of more false alarms (unnecessary maintenance triggers) — a decision that belongs to the domain engineer, not the model.

---

## Feature Importance

Top predictors for machine failure:

1. **Torque** — mechanical overload is the dominant failure signal
2. **Rotational speed** — power-related stress indicator
3. **Tool wear** — cumulative degradation over time

---

## Outputs

```
outputs/
├── predictions.csv         # actual, predicted, predicted_probability
├── metrics.txt             # all metrics + classification report
├── confusion_matrix.png
├── feature_importance.png
└── roc_curve.png           # ROC curve with AUC score
```

---

## Project Structure

```
machine_failure_prediction/
├── data/
│   ├── raw/
│   │   └── data.csv
│   └── create_data.py
├── outputs/
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python data/create_data.py
python main.py
```
