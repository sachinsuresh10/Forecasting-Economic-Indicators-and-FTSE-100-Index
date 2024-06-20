#!/usr/bin/env python
# coding: utf-8

# In[131]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tools.eval_measures import rmse

# Set seaborn style
sns.set_style("whitegrid")


# # Preliminary Analysis

# In[132]:


# Define file path
file_path = "K54Ddata_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot weekly averages
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['K54D'], marker='o', color='green', linestyle='-', linewidth=2, markersize=3)
plt.title('Average Weekly Earnings Over Time')
plt.xlabel('Year')
plt.ylabel('Earning')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



# In[133]:


# Define file path
file = "K54Ddata_34812598.xlsx"
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
plt.title('Average Weekly Earnings Over Time')
plt.xlabel('Year')
plt.ylabel('Earning')

# Change font
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

# Change legends
plt.legend()

# Set end date for the plot
plt.xlim(series.index[0], pd.Timestamp('2024-01-01'))

# Show plot
plt.show()


# # Decomposition

# In[134]:


# Define file path
file_path = "K54Ddata_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(data['K54D'], model='additive')

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
plt.plot(data.index, data['K54D'], label='Original', linestyle=':', color='orange')
plt.legend()

plt.tight_layout()
plt.show()

# Generate forecasts using the trend and seasonal components
forecast_model = ExponentialSmoothing(decomposition.trend, seasonal='additive', seasonal_periods=12)
forecast = forecast_model.fit().forecast(len(data))


# # Corelation Matrix

# In[135]:


# Define file path
file_path = "K54Ddata_34812598.xlsx"
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

# In[136]:


# Define file path
file_path = "K54Ddata_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Extract year from 'Date' column
data['Year'] = data['Date'].dt.year

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Year'], data['K54D'], color='blue', alpha=0.5)
plt.title('Scatter Plot of Earning Over Time')
plt.xlabel('Year')
plt.ylabel('Earning')
plt.grid(True)
plt.tight_layout()
plt.show()


# # AutoCorrelation

# In[137]:


# Define file path
file_path = "K54Ddata_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(data['K54D'], lags=50, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()


# # Holt-Winters model

# In[138]:


# Define file path
file_path = "K54Ddata_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Fit Holt-Winters model with additive seasonality
fit1 = ExponentialSmoothing(data['K54D'], seasonal_periods=12,seasonal='add').fit()

# Fit Holt-Winters model with multiplicative seasonality
fit2 = ExponentialSmoothing(data['K54D'], seasonal_periods=12,seasonal='mul').fit()

forecast_additive = fit1.forecast(12)
forecast_multiplicative = fit2.forecast(12)

# Plot forecasting
plt.figure(figsize=(10, 6))
plt.plot(fit1.fittedvalues.index, fit1.fittedvalues, label='HW Additive Forecast', linestyle='-.', color='red')
plt.plot(fit2.fittedvalues.index, fit2.fittedvalues, label='HW Multiplicative Forecast', linestyle='-.', color='purple')
plt.plot(forecast_additive.index, forecast_additive, label='Additive Forecast', linestyle='--', color='green')
plt.plot(forecast_multiplicative.index, forecast_multiplicative, label='Multiplicative Forecast', linestyle='--', color='orange')
plt.xlabel('Year')
plt.ylabel('Earning')
plt.title('Average Weekly Earnings')
plt.legend()
plt.show()



# # Accuracy(RMSE)

# In[139]:


from statsmodels.tools.eval_measures import rmse

# Calculate RMSE for the additive model
rmse_additive = rmse(data['K54D'], fit1.fittedvalues)

# Calculate RMSE for the multiplicative model
rmse_multiplicative = rmse(data['K54D'], fit2.fittedvalues)

print("RMSE for Additive Model:", rmse_additive)
print("RMSE for Multiplicative Model:", rmse_multiplicative)


# # Multiplicative Forecasted Values

# In[140]:


# Define file path
file_path = "K54Ddata_34812598.xlsx"
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Train the Holt-Winters model
model = ExponentialSmoothing(data['K54D'], seasonal='mul', seasonal_periods=12)
fit_model = model.fit()

# Generate forecasts until December 2024
forecast_index = pd.date_range(start="2024-01-01", end="2024-12-31", freq='M')
forecast = fit_model.forecast(len(forecast_index))

# Print the forecasted values
print("Forecasted Earnings from January 2024 to December 2024:")
for i in range(len(forecast_index)):
    print(forecast_index[i].strftime('%Y-%m'), forecast[i])



# In[ ]:




