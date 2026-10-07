# Earthquake Building Damage Prediction Using Machine Learning

## Overview
This project predicts the level of building damage caused by the **2015 Nepal Earthquake** using machine learning.
The project focuses on comparing different models and studying the impact of **feature selection** and **class-imbalance handling** on prediction performance.

## Problem Statement
The task is a multiclass classification problem in which buildings are classified into three damage grades:
* **1** – Low damage
* **2** – Medium damage
* **3** – Almost complete destruction

The dataset contains **260,601 training records**, **86,868 test records**, and **38 original features**.

## Approach
The project follows these main steps:
* Data exploration and preprocessing
* Numerical feature scaling and categorical feature encoding
* Stratified train-validation split
* Feature selection using `SelectKBest`
* Class-imbalance handling using balanced weights
* Model training and comparison
* Feature importance analysis

Three machine learning models were evaluated:
* Logistic Regression
* Random Forest
* XGBoost

Two feature sets were considered:
* **68 processed features**
* **20 selected features**

The experiments combined different feature sets with two class-handling strategies:

| Configuration | Features | Class handling |
|---|---|---|
| **Baseline** | 68 processed | Original |
| **Experiment 1** | 20 selected | Original |
| **Experiment 2** | 68 processed | Balanced |
| **Experiment 3** | 20 selected | Balanced |

Each configuration was evaluated using Linear Regression, Random Forest and XGBoost.
Each setting was evaluated using Random Forest and XGBoost, resulting in **12 model configurations** in the final comparison.

## Results

**Best Accuracy:** XGBoost (baseline) – 72.15%

**Best Macro-F1:** Random Forest (baseline) – 0.6576

Since the dataset is imbalanced, **Macro-F1 was prioritized when selecting the final model**, making the baseline Random Forest the selected model.

## Key Findings
* Using all 68 processed features performed better than reducing the feature set to 20 features.
* Class balancing did not improve the performance of the best baseline models.
* Random Forest achieved the best Macro-F1.
* XGBoost achieved the highest Accuracy.
* Feature importance analysis was used to identify the most influential features for the final Random Forest model.

## Project Structure

```text
Earthquake-Building-Damage-Prediction/
│
├── Data_Preprocessing_Feature_Selection.ipynb
├── Model_Training_and_Evaluation.ipynb
└── earthquake_damage_submission.csv
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Google Colab / Jupyter Notebook

## How to Run

1. Obtain the Nepal Earthquake Building Damage dataset.
2. Run `Data_Preprocessing_Feature_Selection.ipynb`.
3. Run `Model_Training_and_Evaluation.ipynb`.
4. Review the model comparison, evaluation metrics, and feature importance results.

## Conclusion
The project shows that **using all processed features with a baseline Random Forest provided the best Macro-F1 performance** among the evaluated configurations.
The experiments also demonstrate that feature selection and class balancing do not necessarily improve performance and should be evaluated experimentally for a given dataset.
