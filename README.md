#Group Names: Sai babu patarlapalli , Pruthvi Raj p

**Dataset Description**
The dataset used in this script is historical stock market data, specifically for the stock ticker "TATAMOTORS.NS". This likely represents Tata Motors Limited traded on the National Stock Exchange of India. The data is fetched using the yfinance library, which accesses Yahoo Finance to download stock price information. The range of data extends from January 1, 1999, to March 15, 2024.

**Fields in the Dataset**
The standard fields in stock market data from Yahoo Finance typically include:

Date: The date for the stock data point.
Open: The opening price of the stock on that day.
High: The highest price of the stock on that day.
Low: The lowest price of the stock on that day.
Close: The closing price of the stock when the market closes.
Adj Close: The adjusted closing price, which accounts for any corporate actions such as dividends, stock splits, etc.
Volume: The number of shares traded during the day.
Utility of the Dataset
This dataset is particularly useful for building predictive models to estimate future stock prices based on historical patterns. Understanding stock price trends helps in making informed investment decisions, aiding traders and financial analysts in assessing potential risks and returns.

**Prediction Objective**
The script aims to predict the future prices of Tata Motors stock, specifically the Open, Close, High, and Low prices for the subsequent day. These predictions could be beneficial in several ways:

Investment Strategy: Helps in developing trading strategies, whether for short-term gains based on daily trading or for long-term investment decisions.
Risk Management: Provides insights into potential price movements, aiding in risk assessment and mitigation strategies.
Portfolio Management: Enables portfolio managers to make more informed decisions regarding the composition and rebalancing of their investment portfolios.
Process Overview
Data Fetching and Cleaning: Using yfinance, the script downloads the stock data for the specified period. It then cleans the data by filling missing values to ensure continuity.
Feature Engineering: It calculates various technical indicators (VWAP, EMA, RSI, MACD) and additional features like lags and daily price changes. These are used to capture trends and momentum in the stock price movements.
**X Variables (Features)**
Technical Indicators:
VWAP (Volume Weighted Average Price): Reflects the average price weighted by volume.
EMA (Exponential Moving Average): Indicates trends over different periods.
RSI (Relative Strength Index): Measures the speed and change of price movements.
MACD (Moving Average Convergence Divergence): Shows the relationship between two moving averages of the stock's price.
Lagged Features:
Close_lag_n: Previous n days' closing prices.
Adj_Close_lag_n: Previous n days' adjusted closing prices.
Volume_lag_n: Previous n days' trading volumes.
Daily Price Changes:
Daily_Change: Percentage change in closing price from the previous day.
Adj_Daily_Change: Adjusted closing price percentage change from the previous day.
**Y Variables (Targets)**
Open: Next day's opening price.
Close: Next day's closing price.
High: Next day's highest price.
Low: Next day's lowest price.
**##Model Fitting**
TimeSeriesSplit: For prevenations of data leakge

Model Fitting: The model is fitted using the RandomForestRegressor, trained on shifted train data to predict next day stock prices.
Train/Test Splitting: Utilizes TimeSeriesSplit for train/test splitting, maintaining the temporal sequence of the data and providing multiple train-test splits for validation.
Deciding Train/Test Sizes: The last split from TimeSeriesSplit determines the train/test datasets; size decisions are influenced by the need to have sufficient data for training while leaving enough recent data for testing.

Model Selection: RandomForestRegressor 
Model Type Exploration: MultiRegressor RandomForest 
Hyperparameter Selection Process: GridSearchCV is used to systematically explore combinations of parameters (e.g., n_estimators, max_depth) across a pre-defined grid, using cross-validation to evaluate model performance and mitigate overfitting.

**Validation / metrics
Accuracy, r^2:
R2 Score for Open: 0.8635
R2 Score for Close: 0.8453
R2 Score for High: 0.8540
R2 Score for Low: 0.8566







