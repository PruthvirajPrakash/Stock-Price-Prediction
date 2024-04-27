#Group Names: Sai babu patarlapalli , Pruthvi Raj p

**Dataset Description**
The dataset used in this script is historical stock market data, specifically for the stock ticker "TATAMOTORS.NS". This likely represents Tata Motors Limited traded on the National Stock Exchange of India. The data is fetched using the yfinance library, which accesses Yahoo Finance to download stock price information. .

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
The script aims to predict the future prices of Tata Motors stock, specifically the Open, Close, High, and Low prices for the next day. 

These predictions could be beneficial in several ways:
Investment Strategy
Portfolio Management

**EDA**
TATA Motors has demonstrated strong performance in the automotive sector due to several factors. Firstly, its diverse product portfolio spanning passenger vehicles, commercial vehicles, and electric vehicles caters to a wide range of consumer needs. Secondly, strategic partnerships and collaborations, such as those with Jaguar Land Rover, have enabled access to advanced technology and global markets.
![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/0951259c-21dd-4aad-8893-4e09509d8b20)

Comparing with Technical Indicator

EMA, in particular, helps traders and analysts identify the trend direction and can sometimes act as a support or resistance level. The chart covers a period from around the year 2000 to 2024

![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/4c37f12d-bd74-4937-9454-caad09c20d37)



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
Open: The opening price of the stock on that day.
High: The highest price of the stock on that day.
Low: The lowest price of the stock on that day.
Close: The closing price of the stock when the market closes
**Correlation**
Lag features are strongly corelated with high low and open and close
![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/473a69cb-bc48-4f7a-b636-13b63da41016)
**Feature importance**
All features for X are used excpet Date
Case:
Comprehensive Information: By utilizing all available features, we ensure that our model considers a wide range of information related to the stock's performance.
Potential Interactions: Allowing the model to access a variety of features enables it to capture potential interactions and nonlinear relationships among variables, which could contribute to better predictive performance.

**##Model Fitting**
**Train / test splitting**
the train and test indices for the last fold (i.e., the fifth split) using list(tscv.split(X_scaled))[-1]. This ensures that the test set contains the most recent data, which is crucial for evaluating the model's predictive performance.

![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/6745512d-eb5b-4658-b680-ecb4083a090c)
 
 **Risk of data leakage**
Target Leakage Risk: 
Lag features and daily price changes are calculated with future information, potentially causing the model to learn from data not available at prediction time.
Train-Test Contamination Risk:
Train-test contamination may occur due to the use of StandardScaler before splitting the data into train and test sets.
**Prevent Data Leakage Happening**
To prevent data leakage:
Feature engineering and preprocessing were performed before train/test splitting.
TimeSeriesSplit was utilized for sequential train/test splitting, preserving temporal order.
This ensures model evaluation on unseen future data, minimizing the risk of leakage.
**Model Selection**
Random Forest Regressor was selected as the model of choice due to its ensemble nature, which handles non-linearity and high dimensionality well. 

The Gradient Boosting Regressor, particularly suitable for time series data, was employed for its sequential learning nature, effectively capturing temporal dependencies. Through GridSearchCV.
**Hyper parameter selection**
The hyperparameter selection process involved utilizing GridSearchCV, iterating over a predefined parameter grid, including `n_estimators`, `max_depth`, and `min_samples_split`, employing cross-validation (cv=3) to optimize performance. This allowed for robust model tuning while preventing overfitting and ensuring generalizability across multiple target variables in a time series context.

![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/839caf70-4a16-438b-8637-6b7e366aac3d)

**Validation / metrics**
For Random Forest Model next day predection R-square

R2 Score for Open: 0.8634 
R2 Score for Close: 0.8453
R2 Score for High: 0.8540
R2 Score for Low: 0.8566 
![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/65d6d4b2-ea15-45ac-906c-13877e7498a7)

For GradientBoostingRegressor next day prediction R-square
R2 Score for Open: 0.8621 
R2 Score for Close: 0.8323 
R2 Score for High: 0.8449 
R2 Score for Low: 0.8430
![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/9a294634-57be-42a8-976a-565fccf437b7)
**2 prediction examples which are for next 2 days Predicition**
New values Example 1
Results:

![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/90a291d0-ca77-44fe-a233-03791808cbec)

New value Example 2
Results:


![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/8abda6c1-d365-40ca-8f33-a3b7ba11de16)


**Production**
Deployment of use of this model
Creating a API,it might usefull for Investemnet starategy
Scalability: Design the deployment architecture to handle scalability requirements, especially if there's a need to process large volumes of data or serve a high number of concurrent users.
Monitoring and Logging: Implement robust monitoring and logging mechanisms to track model performance, input data quality, and any potential errors or anomalies during inference.

Outline precautions about its use here
Regular Model Updates: Continuously update the model with fresh data to ensure its predictions remain accurate and relevant over time.
Monitoring and Evaluation: Implement a monitoring system to detect any performance degradation or drift in model predictions. Regularly evaluate model 

 **To enhance the future  model performance**
 Get more data in future and also the model can be intergarted well with sentiment analysis using NLP techniques.
 













