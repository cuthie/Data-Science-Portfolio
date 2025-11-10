# Store Sales Forecasting (Classical Approach)

This repository demonstrates **time series forecasting** for retail store sales using the **Store Sales Dataset** (available on [Kaggle]( https://www.kaggle.com/competitions/store-sales-time-series-forecasting)). The focus of this notebook is on **classical statistical forecasting** using **ARIMA**.

---

## Overview

The notebook guides you through:
1. **Data loading and preprocessing** – selecting a specific store and product family.
2. **Exploratory data analysis (EDA)** – visualizing trends and seasonality.
3. **Stationarity check and differencing** – preparing the series for ARIMA.
4. **ACF and PACF plots** – identifying potential ARIMA parameters.
5. **ARIMA model fitting** – training the model on historical data.
6. **Forecasting future sales** – generating predictions for a test set.
7. **Evaluation** – calculating MAE and RMSE metrics.
8. **Saving results** – exporting the forecast to CSV.
