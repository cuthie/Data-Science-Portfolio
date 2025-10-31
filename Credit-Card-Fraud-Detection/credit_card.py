import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.model_selection import train_test_split

# Load dataset
credit = pd.read_csv("creditcard.csv")

# Check imbalance
print("Original class distribution:")
print(credit['Class'].value_counts(normalize=True))

# Train/test split
split_ratio = 0.8
train_data, test_data = train_test_split(
    credit,
    train_size=split_ratio,
    random_state=1234,
    shuffle=True
)

# Separate features and target
X = credit.drop(columns=['Class'])
y = credit['Class']

# SMOTE with ratio similar to R's perc.over=600, perc.under=125 (minority/majority = 0.8)
sm = SMOTE(sampling_strategy=0.8, random_state=1234)
X_res, y_res = sm.fit_resample(X, y)

smote_data = pd.DataFrame(X_res, columns=X.columns)
smote_data['Class'] = y_res

print("Resampled class distribution (SMOTE):")
print(smote_data['Class'].value_counts(normalize=True))

# Initialize h2o
h2o.init(ip="localhost", port=54321, nthreads=-1, max_mem_size="8G")

# Convert to h2o frames
train_h2o = h2o.H2OFrame(train_data)
test_h2o = h2o.H2OFrame(test_data)
smote_h2o = h2o.H2OFrame(smote_data)

# Convert target to categorical
train_h2o['Class'] = train_h2o['Class'].asfactor()
test_h2o['Class'] = test_h2o['Class'].asfactor()
smote_h2o['Class'] = smote_h2o['Class'].asfactor()

# Model fitting (Random Forest)
model_rf = H2ORandomForestEstimator(
    nfolds=10,
    ntrees=1000,
    seed=1234
)
model_rf.train(y="Class", training_frame=smote_h2o)

# Model testing
perf_rf = model_rf.model_performance(test_data=test_h2o)

print(perf_rf)