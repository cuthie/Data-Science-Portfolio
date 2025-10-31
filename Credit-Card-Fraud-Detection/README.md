# Credit Card Fraud Detection — Kaggle Competition

This repository contains a solution for the Kaggle Credit Card Fraud Detection competition￼, which focuses on building a classification model to detect fraudulent credit card transactions.

Overview

The dataset is highly imbalanced, with fraudulent transactions representing a very small fraction of the total. The project applies data resampling and machine learning techniques to improve fraud detection performance.

Methods
	1.	SMOTE (Synthetic Minority Over-sampling Technique)
Used to address severe class imbalance by generating synthetic samples of the minority (fraudulent) class.
	2.	Distributed Random Forest (DRF)
The predictive model is trained using the Distributed Random Forest algorithm from the h2o.ai library, which efficiently handles large-scale datasets.

Future Improvements
	•	Feature Engineering:
Explore the most informative features and retrain the model for improved performance.
	•	Alternative Algorithms:
Experiment with other machine learning models such as XGBoost, Gradient Boosting Machines (GBM), or Neural Networks to potentially achieve higher accuracy and robustness.
