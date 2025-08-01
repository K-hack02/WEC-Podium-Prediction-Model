# WEC Podium Prediction Model

Predict podium finishers in FIA World Endurance Championship (WEC) races using machine learning.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [References](#references)

---

## Introduction

This project predicts whether a team will finish on the podium (top 3) in FIA WEC races, given practice or qualifying data. The tool can help racing teams, manufacturers, and sponsors estimate race outcomes and optimize their strategies.

The model uses lap-level data from 2012–2022, incorporates extensive data cleaning and feature engineering, and applies machine learning classification to predict podium finishes.

---

## Dataset

- **Source:** [FIA WEC Lap Data 2012-2022 (Kaggle)](https://www.kaggle.com/datasets/tristenterracciano/fia-wec-lap-data-20122022)
- **Size:** 503,680 entries across 10 years, 20 circuits, and 322 unique teams/cars.
- **Features:** The dataset contains lap-by-lap timing and car information for FIA WEC from 2012–2022, including lap times, driver, car number, team, class, circuit, sector times, speeds, and race positions.

---

## Features

- **Starting Position**: Qualifying position at race start
- **Sector Times (s1, s2, s3)**: Average time in each circuit sector
- **Average Speed (kph)**
- **Top Speed**
- **Team Stint Number**: Maximum number of driver switches per team
- **Speed Efficiency**: Average speed divided by top speed
- **Normalized Starting Position**: Relative starting position within each class/race
- **Circuit, Class, Manufacturer**: Encoded categorical data

---

## Installation

1. **Clone the repository** (or copy project files).
2. **Install dependencies**:
    ```bash
    pip install pandas matplotlib scikit-learn kagglehub
    ```
3. **Download the dataset** via Kaggle (Kaggle API key required):
    ```python
    import kagglehub
    path = kagglehub.dataset_download("tristenterracciano/fia-wec-lap-data-20122022")
    ```

---

## Usage

1. **Load and preprocess data**: Cleans N/A values, drops unnecessary columns, and aggregates laps.
2. **Feature engineering**: Generates podium label, speed efficiency, normalized starting positions, etc.
3. **Model training**: Trains a `RandomForestClassifier` on 2012–2021 races, tests on 2022 data.
4. **Model evaluation**: Outputs accuracy, classification report, feature importances, and ROC curve.

To run the main workflow, execute:

run all cells in `main.ipynb`.

---

## Model Details

- **Algorithm:** Random Forest Classifier
- **Hyperparameters:**  
  - `n_estimators`: 150  
  - `max_depth`: 30  
  - `criterion`: 'log_loss'
- **Preprocessing:**  
  - Label Encoding for categorical features  
  - Standard Scaling for numerical features

**Features used:**
- circuit, class, manufacturer
- s1, s2, s3 (sector times)
- podium, kph, top_speed, team_stint_no, normalized_starting_position, speed_efficiency

---

## Results

- **Accuracy:** ~79% on 2022 test data
- **Key findings:**  
  - Starting position and sector 1 time are highly predictive  
  - Manufacturer, stint number, and class have less impact

**Sample Output:**
```text
              precision    recall  f1-score   support

           0       0.82      0.92      0.87       136
           1       0.63      0.39      0.48        33

    accuracy                           0.79       169
   macro avg       0.73      0.66      0.67       169
weighted avg       0.78      0.79      0.77       169
```
Feature importance and ROC curves are plotted and saved automatically.

---

## Examples

Example: Predicting 2022 Podium Finish

```python
# Assuming test data is processed as in main.ipynb
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Troubleshooting

- **Kaggle API errors:** Ensure your Kaggle API key is set up properly.
- **Missing values:** The code fills missing sector/top speed data by group mean, but check your preprocessed data for unhandled NaNs.
- **Encoding mismatch:** If adding new circuits/classes/manufacturers, retrain the LabelEncoder with the new data.

---

## Contributors

- Kavin Vasudevan
- Sidharth Bansal
- Murtaza Shiyaji

---

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/tristenterracciano/fia-wec-lap-data-20122022
