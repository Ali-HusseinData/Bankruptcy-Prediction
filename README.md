# Bankruptcy Prediction Using Machine Learning

This project aims to build a predictive system for identifying companies at risk of bankruptcy using financial ratios and modern machine learning techniques.

## Dataset

The **Taiwanese Bankruptcy Prediction dataset** was used in this project, which contains **6819 companies' financial data over a period of 10 years (1999–2009)**. The dataset includes **64 financial ratios**, with the target variable indicating whether the company went bankrupt.

- **Class Imbalance:** Only **3%** of companies in the dataset are labeled as bankrupt, which required handling techniques such as **SMOTE (Synthetic Minority Oversampling Technique)**.
- **Target Variable:** `Bankrupt?` (1 = Bankrupt, 0 = Not bankrupt)

## Objective

The main goals of this project are:

1. Identify the financial ratios that are most effective in predicting corporate bankruptcy.
2. Evaluate the performance of various machine learning models under **realistic class-imbalance conditions**.

## Methods and Models

Applied a range of machine learning models, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN)
  
To improve performance and interpretability, the following techniques were applied:

- **Feature selection** via `SelectKBest` and `PCA`
- **Oversampling** using `SMOTE`
- **Hyperparameter tuning** via GridSearchCV

## Evaluation Metrics

Due to the severe class imbalance, **accuracy alone is misleading** (even though some models reached up to **96% accuracy**). Therefore, the main focus was on:

- **Precision** for the bankrupt class (minimizing false positives)
- **Recall** for the bankrupt class (minimizing false negatives)
- **F1-score** to balance both
- ROC-AUC and Precision-Recall curves for further insights

## Key Findings

The top-performing models in predicting bankrupt companies were:

**SVM**: F1-score ≈ **41%**
**XGBoost**: F1-score ≈ **40%**
**Random Forest**: F1-score ≈ **40%**

These results were obtained after applying **SelectKBest** for feature selection, **SMOTE** for handling class imbalance, and **hyperparameter tuning** to optimize model performance.

## ROC Curve Comparison

The ROC curves below show each model's ability to distinguish between bankrupt and non-bankrupt companies. Most models performed well, with AUC scores over 0.9.

![ROC Curves](https://github.com/user-attachments/assets/515c1a4f-6411-45b8-b1f0-5ab851a50a4d)


---

## Model Performance Summary

While overall accuracy was high, it does not reflect the model’s ability to detect bankruptcies. The F1-score for class 1 (bankrupt) shows more realistic performance.

![F1 accuracy](https://github.com/user-attachments/assets/c9ae6694-6d5a-4e75-bd90-89b72c5973e9)

## References

- [Taiwanese Bankruptcy Dataset](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data) – UCI Machine Learning Repository

## Future Work

- Validate on international datasets
- Experiment with deep learning techniques
- Integrate alternative data sources
---

## How to Run

1. Open the notebook on [Google Colab](https://colab.research.google.com/drive/1N_GH2NXbV3RfaR26YIRP4wdZvXPp125l#scrollTo=zI6rIxPX0fZH)
2. Run all cells and explore the visualizations and model results
