# Regression with Random Forest and XGBoost (Tidymodels)

## Overview

This project demonstrates an end-to-end regression workflow in R using **tidymodels**, applied to the classic **Pima Indians Diabetes** dataset. While the dataset is commonly used for classification, it is reframed here as a regression problem by modeling **plasma glucose concentration** as a continuous outcome.

The goal is to showcase:

- Proper preprocessing and missing-data handling
- Reproducible train/test splitting
- Cross-validated hyperparameter tuning
- Model comparison using appropriate regression metrics
- Model interpretability via variable importance

This repository serves as a **portfolio-quality example** for data science and applied machine learning roles.

---

## Models Used

- **Random Forest Regression** (ranger engine)  
- **XGBoost Regression** (xgboost engine)  

Both models are implemented using the **tidymodels** framework.

---

## Evaluation Metrics

Regression metrics used in this project:

- **RMSE** – Root Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **R²** – Coefficient of Determination  

Metrics are estimated via **v-fold cross-validation** on the training set and evaluated on a held-out test set.

---

## Project Structure

├── regression_tidymodels.Rmd   # Main analysis (EDA, modeling, tuning, evaluation)
├── README.md                    # Project overview and documentation

---

## Key Methodological Choices

- Invalid zero values in physiological variables are converted to **NA**  
- Missing predictors are handled using **KNN imputation**  
- All numeric predictors are **normalized**  
- Hyperparameters are **tuned via cross-validation**  
- Final performance is assessed using a **test split** to avoid leakage  

---

## Results Summary

- Tree-based models capture nonlinear relationships in glucose levels effectively  
- Random Forest provides strong performance with good interpretability  
- XGBoost offers additional flexibility at the cost of increased tuning complexity  

---

## Reproducibility

All results are **fully reproducible**.  
Random seeds are fixed, and the analysis can be rerun by **knitting the R Markdown file**.

---

## Technologies

- R  
- tidymodels  
- ranger  
- xgboost  
- ggplot2  
- vip  

---

## Author

**Emma Green**

---

## Notes

This project is designed to be easily extensible:

- Add linear or regularized baselines (LM, Lasso, Ridge)  
- Compare models in a unified leaderboard  
- Extend to classification or probabilistic modeling

---
