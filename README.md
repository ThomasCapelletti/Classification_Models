# Loan Status Classification

## Overview
This project performs a comparative analysis of four classification models to predict loan acceptance:
- **Logistic Regression**
- **Linear Discriminant Analysis (LDA)**
- **Quadratic Discriminant Analysis (QDA)**
- **Naïve Bayes (NB)**

The objective is to accurately classify applicants as **approved or rejected** based on specific characteristics. This can help banks improve loan approval efficiency, reduce bad loan risks, and offer customized loan options. The study also explores relationships between independent variables and the dependent variable to identify the best-performing statistical model.

## Performance Evaluation
The models are assessed using:
- **Confusion Matrices**
- **ROC Curves**
- **AUC (Area Under the Curve)**

## Balancing Techniques Used
To handle class imbalance, the following techniques were applied:
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **Undersampling**
- **Oversampling**

## Repository Contents
### CSV Files:
- `loan_data_new`: Original dataset from [Kaggle](https://www.kaggle.com/datasets/willianoliveiragibin/federal-trade-commission-ftc)
- `loan_data_formatted`: Formatted version for better readability (used in **Project_pt.1**)
- `loan_data_formatted2`: Cleaned dataset with selected variables for analysis (used in **Project_pt.2** and **Project_pt.2_B**)
- `loan_data_formatted3`: Dataset containing models for evaluation (**Project_pt.3**)
- `loan_data_formatted3S`: Models generated with **SMOTE** (used in **Project_pt.3_B**)
- `loan_data_formatted3U`: Models generated with **Undersampling** (used in **Project_pt.3_B**)
- `loan_data_formatted3O`: Models generated with **Oversampling** (used in **Project_pt.3_B**)

### R Files:
- `Project_pt.1`: Loads the dataset, performs cleaning, exploratory analysis, and defines variables/dummies.
- `Project_pt.2`: Analyzes correlations and indices for binary variables, selecting relevant features.
- `Project_pt.2_B`: Same as `Project_pt.2`, but applies balancing techniques (**SMOTE, Undersampling, Oversampling**).
- `Project_pt.3`: Compares models from `Project_pt.2`, evaluates Confusion Matrices, and generates ROC Curves.
- `Project_pt.3_B`: Same as `Project_pt.3`, but adapted to models with balancing techniques from `Project_pt.2_B`.

## Results and Comparison
The Excel file in the repository contains:
- Detailed analysis of the **confusion matrix** results for various models.
- A comparison of **84 different models** (4 regression models × 21 variable combinations, considering original and balanced datasets).
- Separate sheets for models:
  - **Standard Models**
  - **Models with SMOTE**
  - **Models with Undersampling**
  - **Models with Oversampling**
- A **custom legend** explaining how the best model was selected based on confusion matrix performance and AUC.

