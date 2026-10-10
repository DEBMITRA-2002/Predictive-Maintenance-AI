# AI-Based Predictive Maintenance System

A machine learning application for estimating industrial machine failure from sensor measurements, with an interactive Streamlit dashboard.

## Overview

Unexpected machine failures can lead to downtime, maintenance costs, and operational disruption. This project explores how machine learning can help identify potential machine failures using sensor readings.

The system uses the AI4I 2020 Predictive Maintenance dataset and provides a web interface where users can enter machine parameters and obtain a model prediction with an estimated failure probability.

## Key Features

- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- Prevention of target leakage by excluding failure-mode flags from model inputs
- Stratified train-test split
- Preprocessing pipeline for numerical and categorical features
- Baseline classification models
- Advanced gradient-boosting models
- Hyperparameter tuning using RandomizedSearchCV
- Model evaluation using classification metrics
- Saved model pipeline using Joblib
- Interactive Streamlit application
- Feature importance visualization where supported by the selected model

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

The dataset contains 10,000 records of synthetic industrial machine data.

Input features used by the model:

- Type
- Air temperature (K)
- Process temperature (K)
- Rotational speed (rpm)
- Torque (Nm)
- Tool wear (min)

**Target:** `Machine failure`

Identifier columns and failure-mode flags are excluded from the main model inputs to reduce identifier noise and avoid target leakage.

Dataset source: [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## Models Explored

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Tuned Random Forest

The final model should be selected based on measured validation results, with particular attention to PR-AUC, Recall, and F1 Score because machine failures are relatively uncommon in this dataset.

## Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC (Average Precision)
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

Actual measured scores should be reported from the generated evaluation files rather than assumed in advance.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
Predictive-Maintenance-AI/
├── app.py
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── final_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── 01_data_understanding.ipynb
├── reports/
│   ├── evaluation_metrics.csv
│   ├── evaluation_metrics.json
│   ├── classification_report.csv
│   └── figures/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── precision_recall_curve.png
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

Some files or directories may be absent until the corresponding project steps have been completed.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Predictive-Maintenance-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

For macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset and trained model

Place the downloaded dataset at:

```text
data/raw/ai4i2020.csv
```

Run the notebook cells to train and evaluate the models, then save the selected fitted pipeline to:

```text
models/final_model.joblib
```

The Streamlit application requires this model file to make predictions.

### 5. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Usage

1. Launch the Streamlit application.
2. Enter the machine type and sensor readings.
3. Click **Predict Machine Failure**.
4. Review the predicted class and estimated failure probability.
5. Explore the Model Insights and About tabs.

## Outputs

The completed project can generate:

- A saved, fitted machine learning pipeline
- Model metadata
- Evaluation metrics in CSV and JSON formats
- A classification report
- Confusion matrix and evaluation curves
- An interactive prediction dashboard

## Limitations

- The AI4I 2020 dataset is synthetic and may not represent real industrial operating conditions.
- Performance on this dataset does not guarantee performance on actual machinery.
- Predicted probabilities should not be interpreted as calibrated real-world failure risks unless calibration has been independently evaluated.
- This application is an academic prototype and must not replace industrial inspection, engineering judgment, or safety systems.

## Future Improvements

- Evaluate on real industrial sensor data.
- Perform cross-validation and systematic model comparison.
- Calibrate predicted probabilities.
- Add explainability methods such as permutation importance or SHAP.
- Introduce time-series monitoring and maintenance alerts.
- Deploy the application to a suitable cloud platform.

## Author

**Debmitra Chatterjee**

Machine Learning Project — AI-Based Predictive Maintenance System

