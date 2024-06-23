# Forecasting-Economic-Indicators-and-FTSE-100-Index

This repository contains a comprehensive technical report and associated code for forecasting various economic indicators and the FTSE 100 Index using different time series forecasting methods. The primary methodologies employed are Exponential Smoothing, SARIMA, and Regression Analysis. The analysis is based on datasets from the Office for National Statistics (ONS) and the FTSE 100 share index.

## Repository Structure

- `data/`: Contains the datasets used for the analysis.
- `scripts/`: Contains the Python scripts for different forecasting methods.
- `results/`: Contains the generated plots and forecast results.
- `report/`: Contains the technical report.

## Datasets

The datasets used for the analysis include:
- Average weekly earnings (K54D)
- Retail sales index (EAFV)
- Extraction of crude petroleum and natural gas (K226)
- Manufacturing and business sector turnover and orders (JQ2J)
- FTSE 100 share index (FTSE)

## Methods

### 1. Exponential Smoothing
- **Data Preparation**: Time series data was formatted with dates and numerical values.
- **Preliminary Analysis**: Visual and quantitative examination of trends, seasonality, and anomalies.
- **Modeling**: Holt-Winters' and Holt's Linear Exponential Smoothing models were used.
- **Forecasts**: Generated for the next 12 periods.
- **Scripts**: 
    - `ExponentialSmoothing_K54D.py`
    - `ExponentialSmoothing_EAFV.py`
    - `ExponentialSmoothing_K226.py`
    - `ExponentialSmoothing_JQ2J.py`
    
### 2. ARIMA Forecasting
- **Preliminary Analysis**: ACF and PACF analysis to determine correlation structure.
- **SARIMA Model**: Seasonal ARIMA model fitted using optimal parameters.
- **Scripts**: 
    - `ARIMA_SARIMA.py`
    
### 3. Regression Prediction
- **Data Preparation**: Creation of a new dataset "FTSEdata_34812598.xls" with selected variables.
- **Regression Model**: Multivariate regression model using OLS.
- **Scripts**: 
    - `REGRESSION_FTSE.py`
    
## Instructions

### 1. Data Preparation
- Ensure the datasets are placed in the `data/` directory.
- Modify the file paths in the scripts if necessary to match your directory structure.

### 2. Running the Scripts
- Install the required Python libraries:
  ```sh
  pip install pandas numpy matplotlib seaborn statsmodels pmdarima

python scripts/ExponentialSmoothing_K54D.py
python scripts/ExponentialSmoothing_EAFV.py
python scripts/ExponentialSmoothing_K226.py
python scripts/ExponentialSmoothing_JQ2J.py
python scripts/ARIMA_SARIMA.py
python scripts/REGRESSION_FTSE.py


### 3. Viewing Results
The generated plots and forecasts will be saved in the results/ directory.
Detailed analysis and comparisons can be found in the report/ directory.
### Dependencies
Python 3.6+
pandas
numpy
matplotlib
seaborn
statsmodels
pmdarima
