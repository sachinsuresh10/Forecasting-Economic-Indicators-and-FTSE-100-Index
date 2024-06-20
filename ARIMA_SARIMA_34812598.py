#!/usr/bin/env python
# coding: utf-8

# # Preliminary Analysis

# In[67]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

# Load the dataset
file_path = 'K54Ddata_34812598.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Plot ACF and PACF plots to identify model parameters
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['K54D'], ax=ax[0], lags=40)
plot_pacf(data['K54D'], ax=ax[1], lags=40)
plt.show()


# # SARIMA Modeling

# In[28]:


!pip install pmdarima



# In[68]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Load the dataset
file_path = 'K54Ddata_34812598.xlsx'
data = pd.read_excel(file_path, index_col='Date', parse_dates=True)


# Find the best parameters using auto_arima
auto_model = auto_arima(data['K54D'], seasonal=True, m=12, trace=True)

# Get the best model parameters
order = auto_model.order
seasonal_order = auto_model.seasonal_order

print("Best model parameters (p, d, q):", order)
print("Best seasonal parameters (P, D, Q, S):", seasonal_order)


# # SARIMA Forecasting Results

# In[69]:


# Fit SARIMAX model with the best parameters
model = SARIMAX(data['K54D'], order=order, seasonal_order=seasonal_order)
fit_model = model.fit()

# Generate forecasts until December 2024 (12 months)
forecast_index = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]  # Forecast index for the next 12 months
forecast = fit_model.forecast(steps=len(forecast_index))  # Forecast for the next 12 months

# Plot original data and forecasts
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['K54D'], label='Actual', color='red')
plt.plot(forecast_index, forecast, label='Forecast', linestyle='--',color='green')
plt.title('SARIMA Forecasting until December 2024')
plt.xlabel('Year')
plt.ylabel('Earning')
plt.legend()
plt.grid(True)
plt.show()


# # SARIMA Forecasting Values

# In[70]:


# Extract forecasted values from December 2023 to December 2024
forecast_dec_2023_to_dec_2024 = forecast[(forecast_index >= '2023-12-01') & (forecast_index <= '2025-01-01')]

# Display the forecasted values
print(forecast_dec_2023_to_dec_2024)


# # Accuracy(RMSE)

# In[71]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Define the actual values for the forecast period
actual_values = data['K54D'][-12:]  # Assuming the last 12 months are the test period

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values, forecast))

# Print RMSE
print("Root Mean Squared Error (RMSE):", rmse)


# In[ ]:





# In[ ]:




