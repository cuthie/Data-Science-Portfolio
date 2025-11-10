# Time Series Forecasting: Retail Store Sales (Classical Approach)

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# --- 2. Load Dataset ---
# You can download the dataset from: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
# After downloading, place the file 'train.csv' in your working directory.

df = pd.read_csv('train.csv', parse_dates=['date'])
df = df.sort_values(['store_nbr', 'family', 'date'])
print("Data shape:", df.shape)
df.head()

# --- 3. Select a Single Store and Product Family for Simplicity ---
store_id = 1
product_family = 'GROCERY I'

data = df[(df['store_nbr'] == store_id) & (df['family'] == product_family)][['date', 'sales']]
data = data.set_index('date').asfreq('D')

# Fill missing dates if any
data['sales'] = data['sales'].fillna(method='ffill')

# --- 4. Visualize Time Series ---
plt.figure(figsize=(12,5))
plt.plot(data.index, data['sales'], label='Daily Sales')
plt.title(f'Store {store_id} - {product_family} Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# --- 5. Check Stationarity ---
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("Series is stationary.")
    else:
        print("Series is non-stationary.")

adf_test(data['sales'])

# --- 6. Differencing if Needed ---
data_diff = data['sales'].diff().dropna()
adf_test(data_diff)

plt.figure(figsize=(10,4))
plt.plot(data_diff)
plt.title('Differenced Series')
plt.show()

# --- 7. ACF and PACF Plots ---
fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(data_diff, ax=axes[0])
plot_pacf(data_diff, ax=axes[1])
plt.show()

# --- 8. Train-Test Split ---
split_date = '2017-07-01'
train = data.loc[data.index < split_date]
test = data.loc[data.index >= split_date]

print(f"Train samples: {len(train)} | Test samples: {len(test)}")

# --- 9. Fit ARIMA Model ---
model = ARIMA(train['sales'], order=(1,1,1))  # You can adjust p,d,q based on ACF/PACF
model_fit = model.fit()
print(model_fit.summary())

# --- 10. Forecast ---
forecast = model_fit.forecast(steps=len(test))

# --- 11. Plot Forecast vs Actual ---
plt.figure(figsize=(12,5))
plt.plot(train.index, train['sales'], label='Train')
plt.plot(test.index, test['sales'], label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title(f'ARIMA Forecast for Store {store_id} - {product_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# --- 12. Evaluate Model ---
mae = mean_absolute_error(test['sales'], forecast)
rmse = np.sqrt(mean_squared_error(test['sales'], forecast))
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# --- 13. Save Forecast Results ---
forecast_df = pd.DataFrame({'date': test.index, 'actual': test['sales'], 'forecast': forecast})
forecast_df.to_csv('store_sales_forecast_results.csv', index=False)
print('Forecast results saved to store_sales_forecast_results.csv')