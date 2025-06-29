# Bankruptcy-Prediction
Predicting corporate bankruptcy using financial ratios and machine learning.
# 🏦 Bankruptcy Prediction Using Machine Learning

This project aims to build a predictive system for identifying companies at risk of bankruptcy using financial ratios and modern machine learning techniques.

## 📊 Dataset

We use the **Taiwanese Bankruptcy Prediction dataset**, which contains **6819 companies' financial data over a period of 10 years (1999–2009)**. The dataset includes **64 financial ratios**, with the target variable indicating whether the company went bankrupt.

- **Class Imbalance:** Only **3%** of companies in the dataset are labeled as bankrupt, which required handling techniques such as **SMOTE (Synthetic Minority Oversampling Technique)**.
- **Target Variable:** `Bankrupt?` (1 = Bankrupt, 0 = Not bankrupt)

## 🧠 Objective

The main goals of this project are:

1. Identify the financial ratios that are most effective in predicting corporate bankruptcy.
2. Evaluate the performance of various machine learning models under **realistic class-imbalance conditions**.

## ⚙️ Methods and Models

We applied a range of machine learning models, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN)
To improve performance and interpretability, we also applied:

- **Feature selection** via `SelectKBest` and `PCA`
- **Oversampling** using `SMOTE`
- **Hyperparameter tuning** via GridSearchCV

## 🎯 Evaluation Metrics

Due to the severe class imbalance, **accuracy alone is misleading** (even though some models reached up to **96% accuracy**). Therefore, we focus on:

- **Precision** for the bankrupt class (minimizing false positives)
- **Recall** for the bankrupt class (minimizing false negatives)
- **F1-score** to balance both
- ROC-AUC and Precision-Recall curves for further insights

## 📈 Key Findings

The top-performing models in predicting bankrupt companies were:

**SVM**: F1-score ≈ **41%**
**XGBoost**: F1-score ≈ **40%**
**Random Forest**: F1-score ≈ **40%**

These results were obtained after applying **SelectKBest** for feature selection, **SMOTE** for handling class imbalance, and **hyperparameter tuning** to optimize model performance.

## 📉 ROC Curve Comparison

The ROC curves below show each model's ability to distinguish between bankrupt and non-bankrupt companies. Most models performed well, with AUC scores over 0.9.

![ROC Curves](images/ROC Curves.jpeg)

---

## 📊 Model Performance Summary

While overall accuracy was high, it does not reflect the model’s ability to detect bankruptcies. The F1-score for class 1 (bankrupt) shows more realistic performance.

![F1 Score and Accuracy](images/F1 accuracy.jpeg)

## 📁 File Structure
Bankruptcy-Prediction/
│
├── Bankruptcy-Prediction.ipynb # Main Colab notebook
├── README.md # This file
├── requirements.txt #  Python packages used
└── /data #  Folder for sample data

## 📚 References

- [Taiwanese Bankruptcy Dataset](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data) – UCI Machine Learning Repository

## 🚀 Future Work

- Validate on international datasets
- Experiment with deep learning techniques
- Integrate alternative data sources
---

## 🔗 How to Run

1. Open the notebook on [Google Colab](https://colab.research.google.com/)
2. Make sure required libraries are installed (see `requirements.txt`)
3. Run all cells and explore the visualizations and model results

