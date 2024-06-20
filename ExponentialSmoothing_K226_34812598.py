#!/usr/bin/env python
# coding: utf-8

#  # Preliminary Analysis

# In[58]:


import pandas as pd
import matplotlib.pyplot as plt

# Define file path
file_path = "K226data_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot weekly averages
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['K226'], marker='o', color='green', linestyle='-', linewidth=2, markersize=3)
plt.title('Extraction of Crude Petroleum and Natural Gas Over Time')
plt.xlabel('Year')
plt.ylabel('Extraction')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")

# Define file path
file = "K226data_34812598.xlsx"
series = pd.read_excel(file, header=0, index_col=0, parse_dates=True).squeeze()

# Create empty Series for MA7
MA7 = pd.Series(index=series.index)

# Fill series with MA7
for i in np.arange(3, len(series) - 3):
    MA7[i] = np.mean(series[(i-3):(i+4)])

# Create empty Series for MA2x12
MA2x12 = pd.Series(index=series.index)

# Fill series with MA2x12
for i in np.arange(6, len(series) - 6):
    MA2x12[i] = np.sum(series[(i-6):(i+7)] * np.concatenate([[1/24], np.repeat(1/12, 11), [1/24]]))

# Set plot size
plt.figure(figsize=(10, 6))

# Plot original time series
original_plot = series.plot(color='red', label='Original')
MA7_plot = MA7.plot(color='green', label='Seasonal')
MA2x12_plot = MA2x12.plot(color='orange', label='Trend')

# Change font
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

# Change legends
plt.legend()

# Set end date for the plot
plt.xlim(series.index[0], pd.Timestamp('2024-01-01'))

# Show plot
plt.show()


# # Decomposition

# In[60]:


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Define file path
file_path = "K226data_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(data['K226'], model='additive')

# Plot the decomposed components with different styles and colors
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(data.index, decomposition.trend, label='Trend', linestyle='-', color='orange')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(data.index, decomposition.seasonal, label='Seasonal', linestyle='--', color='orange')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(data.index, decomposition.resid, label='Residual', linestyle='-.', color='orange')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(data.index, data['K226'], label='Original', linestyle=':', color='orange')
plt.legend()

plt.tight_layout()
plt.show()

# Generate forecasts using the trend and seasonal components
forecast_model = ExponentialSmoothing(decomposition.trend, seasonal='additive', seasonal_periods=12)
forecast = forecast_model.fit().forecast(len(data))


# # Correlation Matrix

# In[61]:


import seaborn as sns

# Define file path
file_path = "K226data_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Calculate correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix as a heatmap with yellow and green color map
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='YlGn', annot=True, fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Correlation matrix for all numeric variables
CorrelationMatrix = data.corr()
print(CorrelationMatrix)


# # Scatter Plot

# In[62]:


# Define file path
file_path = "K226data_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Extract year from 'Date' column
data['Year'] = data['Date'].dt.year

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Year'], data['K226'], color='blue', alpha=0.5)
plt.title('Scatter Plot of EXtraction Over Time')
plt.xlabel('Year')
plt.ylabel('Extraction')
plt.grid(True)
plt.tight_layout()
plt.show()


# # Auto-Correlation

# In[63]:


from statsmodels.graphics.tsaplots import plot_acf


# Define file path
file_path = "K226data_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(data['K226'], lags=50, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()


# # Holt Winters Model

# In[64]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Define file path
file_path = "K226data_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Fit Holt's linear exponential smoothing model
fit_linear = ExponentialSmoothing(data['K226'], trend='add', seasonal_periods=12).fit()

# Fit Holt-Winters model with additive seasonality
fit_additive = ExponentialSmoothing(data['K226'], seasonal_periods=12, seasonal='add').fit()

# Fit Holt-Winters model with multiplicative seasonality
fit_multiplicative = ExponentialSmoothing(data['K226'], seasonal_periods=12, seasonal='mul').fit()

forecast_linear = fit_linear.forecast(12)
forecast_additive = fit_additive.forecast(12)
forecast_multiplicative = fit_multiplicative.forecast(12)

# Plot forecasting
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['K226'], label='Actual', color='blue')
plt.plot(fit_linear.fittedvalues.index, fit_linear.fittedvalues, label="Holt's Linear Exponential Smoothing Fitted Values", linestyle='-.', color='cyan')
plt.plot(fit_additive.fittedvalues.index, fit_additive.fittedvalues, label='HW Additive Fitted Values', linestyle='-.', color='red')
plt.plot(fit_multiplicative.fittedvalues.index, fit_multiplicative.fittedvalues, label='HW Multiplicative Fitted Values', linestyle='-.', color='purple')
plt.plot(forecast_linear.index, forecast_linear, label="Holt's Linear Exponential Smoothing Forecast", linestyle='--', color='magenta')
plt.plot(forecast_additive.index, forecast_additive, label='Additive Forecast', linestyle='--', color='green')
plt.plot(forecast_multiplicative.index, forecast_multiplicative, label='Multiplicative Forecast', linestyle='--', color='orange')
plt.xlabel('Year')
plt.ylabel('Extraction')
plt.title('Extraction of Crude Petroleum and Natural Gas')
plt.legend()
plt.show()


# # Accuracy(RMSE)

# In[65]:


from statsmodels.tools.eval_measures import rmse

# Calculate RMSE for the Holt's Linear Exponential Smoothing (LES) model
rmse_linear = rmse(data['K226'], fit_linear.fittedvalues)

# Calculate RMSE for the additive model
rmse_additive = rmse(data['K226'], fit_additive.fittedvalues)

# Calculate RMSE for the multiplicative model
rmse_multiplicative = rmse(data['K226'], fit_multiplicative.fittedvalues)

print("RMSE for Holt's Linear Exponential Smoothing Model:", rmse_linear)
print("RMSE for Additive Model:", rmse_additive)
print("RMSE for Multiplicative Model:", rmse_multiplicative)


# #  Linear Exponential Smoothing Forecasted Values

# In[66]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Define file path
file_path = "K226data_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Train with Holt’s linear exponential smoothing
model = ExponentialSmoothing(data['K226'], trend='add', seasonal_periods=12)
fit_model = model.fit()

# Generate forecasts until December 2024
forecast_index = pd.date_range(start="2024-01-01", end="2024-12-31", freq='M')
forecast = fit_model.forecast(len(forecast_index))

# Print the forecasted values
print("Forecasted Extraction from January 2024 to December 2024:")
for i in range(len(forecast_index)):
    print(forecast_index[i].strftime('%Y-%m'), forecast[i])


# In[ ]:




