# Stock Price Prediction Project

# Project Overview
This project aims to predict the stock price of the next day using historical data of Tata Motors (TATAMOTORS.NS) obtained from Yahoo Finance via the yfinance API. 
The focus is on utilizing technical indicators and machine learning models to forecast future stock prices. 
Predictions from this project can be utilized in algorithmic trading, investment planning, and risk management.

# Dataset Description
This dataset consists of historical stock data for Tata Motors (TATAMOTORS.NS) sourced from Yahoo Finance.
It covers a period from January 1, 1999, to April 22, 2024, encompassing approximately 6300 trading days. 
This extensive range allows for a deep analysis of stock trends and patterns over different market phases.

# Data Features
Date: The trading date.
Open: The opening price of the stock for the trading day.
High: The highest price of the stock during the trading day.
Low: The lowest price of the stock during the trading day.
Close: The closing price of the stock at the end of the trading day.
Adjusted Close: The closing price after adjustments for all applicable splits and dividend distributions.
Volume: The number of shares traded during the trading day.

# Technical Indicators
In addition to the basic stock price data, several technical indicators have been calculated and added to enhance the predictive capabilities:

Exponential Moving Average (EMA): A type of moving average that places a greater weight and significance on the most recent data points.
Ichimoku Cloud: A collection of technical indicators that show support and resistance levels, as well as momentum and trend direction.
Keltner Channels: A volatility indicator that shows a central moving average line plus channel lines at a distance above and below.
Pivot Points: A technical analysis indicator used to determine the overall trend of the market over different time frames.

Limitations
It is important to note that this dataset does not account for all market influences that might impact stock prices. Macro-economic indicators, global financial events, political stability, and other external factors are not represented in this data,
which could affect comprehensive market analysis.

# Purpose of Selecting this Dataset
Strategic Relevance: 

The choice of Tata Motors stock data from Yahoo Finance is grounded in the objective to thoroughly analyze stock price movements and identify prevalent market trends.
This data provides a detailed historical record, making it an excellent basis for building predictive models aimed at forecasting stock prices. Such models are valuable 
tools for investors, aiding in informed decision-making regarding the buying and selling of stocks.

Predictive Modeling Goals

The primary goal of this project is to develop a robust predictive model that can forecast future stock prices based on historical data. 
This involves leveraging technical indicators, such as Exponential Moving Average (EMA), Ichimoku Cloud, Keltner Channels, and Pivot Points, which are known to encapsulate underlying market sentiments and dynamics effectively. 
These indicators help in interpreting complex market data and trends, making them indispensable for predictive analytics in finance.

# Features

Numerical Features :

The dataset comprises several numerical features crucial for the analysis and prediction of stock prices:

Price-Related Features:

Open, High, Low, Close: These represent the daily trading range of the stock. The Open and Close reflect the starting and final prices for the trading day, while the High and Low demonstrate the maximum and minimum prices during the day.
Adjusted Close: This adjusts the closing price for the stock, taking into account any corporate actions such as dividends, splits, or rights offerings, hence providing a more accurate reflection of the stock's value.
Volume: Represents the total number of shares traded during the day. High volume can indicate strong interest in the stock on a particular day and can be a sign of market sentiment towards a stock’s movement.

Technical Indicators:

EMA (Exponential Moving Average): Used to smooth out price data over a specified period, helping to identify longer-term trends by mitigating the effects of short-term fluctuations.

Ichimoku Cloud Components:

Tenkan-sen and Kijun-sen: Short-term and medium-term momentum indicators, respectively.
Senkou Span A & B: Represent the edges of the "cloud," projecting potential support and resistance levels into the future.
Chikou Span: Plots the closing price 26 periods back and can indicate potential areas of support or resistance from past prices.
Keltner Channels: Consist of three lines that form a channel around the price, which help in identifying trend direction, breakouts, and volatility.

Categorical Features :

While the dataset is primarily numerical, it includes key categorical features that provide a temporal dimension to the analysis:

Date: Captures the day of trading, which is essential for tracking price movements over time and analyzing trends or cycles in the stock market.
Weekday: Extracted from the Date, this feature allows for the analysis of weekly patterns in stock price movements, such as variations in volatility or trading volume that might occur on specific days of the week.

# Process Overview
The development of this project was methodically structured into distinct steps to ensure effective analysis and precise predictions of Tata Motors' stock prices for the next day. Here's a detailed step-by-step overview of the process:

Step 1: Data Collection
Data Sourcing: Historical stock data for Tata Motors from January 1, 1999, to April 22, 2024, was downloaded using the yfinance library.
Initial Data Review: A preliminary check was performed to ensure data completeness and consistency across features such as Open, High, Low, Close, Volume, and Adjusted Close.

Step 2: Data Preprocessing
Cleaning: The dataset was cleaned to remove any anomalies or missing values, ensuring a reliable dataset for analysis.
Feature Engineering: Computed additional features such as EMA, Ichimoku Cloud components, Keltner Channels, and Pivot Points to enrich the dataset.

Step 3: Exploratory Data Analysis (EDA)
Statistical Analysis: Applied descriptive statistics to explore the distribution and variance of the data.
Visualization: Generated various plots to inspect trends, detect seasonal effects, and analyze correlations between different variables.

Step 4: Model Development
Model Selection: Evaluated several machine learning models, including Gradient Boosting Regressors and Random Forest Regressors, to determine their suitability.
Training: The models were trained using a Time Series Split to maintain the chronological order of data.
Hyperparameter Tuning: Utilized GridSearchCV to fine-tune the model parameters, aiming to maximize predictive accuracy.

Step 5: Model Evaluation and Selection
Testing: Tested the models on unseen data to evaluate their performance, using metrics such as Mean Absolute Error (MAE) and R-squared.
Model Selection: Selected the model that provided the best balance between accuracy and computational efficiency for deployment.

Step 6: Deployment and Monitoring
Integration: Integrated the final model into a simulated trading environment to test its real-time capabilities.
Monitoring: Established continuous monitoring to track the model’s performance over time, making necessary adjustments based on feedback and market changes.

# Outputs and Visualizations
This project generates several outputs and visualizations that illustrate the underlying trends and patterns in the stock price data of Tata Motors. Here are the key elements:

Data Tables
Historical Stock Data: The dataset displays various stock metrics including Open, High, Low, Close, Adjusted Close, and Volume for each trading day from January 1, 1999, to April 22, 2024. This comprehensive table is pivotal for the initial data review and subsequent analyses.

Graphs
Stock Price Movement: A line graph that plots the Open, High, Low, and Close prices over time. This visualization helps to quickly grasp the stock price trends and the volatility of the market on a day-to-day basis.
Technical Indicator Analysis:
EMA (Exponential Moving Average): A line graph showing the 20-day EMA overlaid on the closing prices, which helps in identifying the general price trend and smoothing out price data to better capture the underlying trends.
Keltner Channels and Pivot Points: These are plotted to visualize the volatility and potential support/resistance levels, aiding in the technical analysis for predictive modeling.

Statistical Outputs
Missing Data Check: The output includes a summary of missing values across all columns, ensuring that the data integrity is maintained and that the dataset is ready for robust analysis.
Enhanced Features
Technical Indicators: Additional calculated fields like EMA, Pivot Points, and Keltner Channels are appended to the dataset. These indicators are crucial for developing sophisticated trading strategies and enhancing the predictive models.

File Outputs
CSV Files:
Tatastockdfset.csv: Contains the raw downloaded data.
enhanced_features.csv: Includes the processed data with additional technical indicators, ready for further analysis or model training.

# Setup and Model Accuary 
1. The code starts by downloading stock price data for Tata Motors from Yahoo Finance, covering dates from January 1, 1999, to April 24, 2024.
The fetched data is saved into a CSV file named Tatastockdfset.csv.
![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/166566162/abe6cb36-c2eb-4c8b-9211-86c10f7cc2ce)

We got 6329 rows and 7 columns.

2. Technical Indicators such as EMA, Pivot Points, and Keltner Channels are calculated from the price data.
These indicators are commonly used in trading to analyze market trends and volatility.
The dataframe is then saved with these new features to enhanced_features.csv.

![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/166566162/d1eccc7f-2b65-4ccf-acc9-26cdbc04c74f)

3. EDA

![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/166566162/b233ca3f-0cfc-470c-af71-739d5c27b031)

4. We are plotting the Open, High, Low and close prices

![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/166566162/6a985879-85c5-4363-a87c-2f832cb77e00)

5. Computing the correlation matrix for the entire DataFrame and plotted the heatmap

![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/166566162/2c0712f4-7534-42de-b373-812700a994de)

6. Choosing feature set X and target set Y

Dropping specified columns to create the feature set X
X = df.drop(["Open", "Close", "High", "Low"], axis=1)

Creating the target set y with specified columns
y = df[["Open", "Close", "High", "Low"]]

7. The data is split into training and test sets using train_test_split with 20% of the data reserved for testing.
   This function is used with shuffle=False to preserve the time series nature of the data, ensuring that future data isn't inadvertently used in the training process.

8. TimeSeriesSplit is initialized with 5 splits, tailored for time series data. This ensures that each test set in the cross-validation is strictly ahead in time compared to the training set.

9. Models: Two types of regressors, GradientBoostingRegressor and RandomForestRegressor, are wrapped in MultiOutputRegressor to handle multi-output regression tasks, allowing them to predict multiple dependent variables (like open, close, high, and low prices) simultaneously.

10. Parameter Grids: Specific parameters are set for each model to be tuned using GridSearchCV.

    For the Gradient Boosting Regressor, the number of trees (n_estimators), the pace of learning (learning_rate), and the depth of trees (max_depth) are varied.

    For the RandomForest Regressor, the number of trees and the number of features to consider when looking for the best split are set to vary.

11. We fitted models using GridSearchCV and store best models.  (GridSearchCV is a systematic way to search through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. to prevent overfittiing and bias.
    GridSearchCV is used with a TimeSeriesSplit to fine-tune the models' hyperparameters and validate their performance using time-based cross-validation. )

12. Last comes the model evaulation.
    
    GradientBoostingRegressor:
    Mean Absolute Error: 30.027629130229613
    R-squared: 0.8477301075779412

    The GradientBoostingRegressor did perform well with a lower MAE and a higher R-squared value. An MAE of 30.03 indicates on average the predictions of the model are about 30.03 units away from the actual stock prices.
    The R-squared value of 0.848, which is quite close to 1, suggests that approximately 84.8% of the variance in the stock prices is predictable from the features.

    RandomForestRegressor:
    Mean Absolute Error: 38.48915695987419
    R-squared: 0.8032208244628982

    The RandomForestRegressor shows a higher MAE and a lower R-squared compared to the GradientBoostingRegressor.
    The MAE of 38.49 means the average prediction error is somewhat higher, and with an R-squared of 0.803, about 80.3% of the variance in the stock prices is being explained. This suggests that while the model is still performing well,
    it isn’t capturing as much of the variability as the Gradient Boosting model.

# Conclusion:

The GradientBoostingRegressor outperforms the RandomForestRegressor in this scenario, with more accurate predictions and a better ability to explain the variation in stock prices.
The results suggest that boosting techniques, which focus on reducing bias and variance in sequential corrections of predictions, may be more suited to this particular problem of stock price prediction compared to the bagging approach used in RandomForest.
Considering the above results, if computational resources and model training time allow, you might prefer using the Gradient Boosting approach for your stock price prediction tasks, especially when accuracy is a priority.
