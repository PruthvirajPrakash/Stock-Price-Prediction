# %%
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# Getting df From Yahoo using Ticker

# %%
#Downloading df
ticker = 'TATAMOTORS.NS'  # Example ticker
start_date = '1999-01-01'
end_date = '2024-04-24'
df = yf.download(ticker, start=start_date, end=end_date)
df.to_csv("Tatastockdfset.csv")

# %%
df = pd.read_csv("Tatastockdfset.csv")
df

# %% [markdown]
# Data cleaning

# %%
missing_count = df.isna().sum()
print(missing_count)

# %%
import pandas as pd
import ta
# Adding Exponential Moving Average (EMA) for a 20-day period
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# Adding Pivot Points
P = (df['High'] + df['Low'] + df['Close']) / 3
df['Pivot_Point'] = P

# Adding Keltner Channels
keltner_channel = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10)
df['Keltner_Channel_hband'] = keltner_channel.keltner_channel_hband()
df['Keltner_Channel_lband'] = keltner_channel.keltner_channel_lband()
df['Keltner_Channel_mband'] = keltner_channel.keltner_channel_mband()


# Save to CSV
df.to_csv("enhanced_features.csv")

# Handle NaN values that may have been created by the indicators
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# %%
df

# %% [markdown]
# EDA Analysis

# %%
#EDA

# %%
import pandas as pd
import matplotlib.pyplot as plt


# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df['Open'], label='Open')
plt.plot(df['High'], label='High')
plt.plot(df['Low'], label='Low')
plt.plot(df['Close'], label='Close')
plt.plot(df['EMA_20'], label='EMA_20', linestyle='--')

plt.title('Financial Indicators Over Time')
plt.xlabel('Date (Index)')
plt.ylabel('Indicator Value')
plt.legend()
plt.grid(True)
plt.show()


# %%
import matplotlib.pyplot as plt

# Plotting the 'Close' column
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Close Price Over Time')
plt.legend()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt



# Ensure the Date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the date as the index of the dataframe
df.set_index('Date', inplace=True)

# Plot each variable
plt.figure(figsize=(14, 10))

# Subplot for 'Open'
plt.subplot(3, 2, 1)
df['Open'].plot(title='Open Price', color='blue')
plt.ylabel('Price')

# Subplot for 'High'
plt.subplot(3, 2, 2)
df['High'].plot(title='High Price', color='green')
plt.ylabel('Price')

# Subplot for 'Low'
plt.subplot(3, 2, 3)
df['Low'].plot(title='Low Price', color='red')
plt.ylabel('Price')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


# %%
df.drop("Date", axis=1, inplace=True)
df

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Compute the correlation matrix for the entire DataFrame
correlation_matrix = df.corr()

# Plotting the heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix with Coolwarm Theme')
plt.show()


# %%
# Drop specified columns to create the feature set X
X = df.drop(["Open", "Close", "High", "Low"], axis=1)

# Create the target set y with specified columns
y = df[["Open", "Close", "High", "Low"]]



# %%
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
# Split the data for non-time series models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#cross-validation
# Define TimeSeriesSplit for models using GridSearchCV
tscv = TimeSeriesSplit(n_splits=5)

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor


# Initialize and setup models
models = {
    'GradientBoostingRegressor': MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
    'RandomForestRegressor': MultiOutputRegressor(RandomForestRegressor(random_state=0))
}

# Parameter grids for each model
param_grids = {
    'GradientBoostingRegressor': {'estimator__n_estimators': [100, 200], 'estimator__learning_rate': [0.05, 0.1], 'estimator__max_depth': [3, 5]},
    'RandomForestRegressor': {'estimator__n_estimators': [10, 50], 'estimator__max_features': ['auto', 'sqrt']}
}






# %% [markdown]
# GridSearchCV is a systematic way to search through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. to prevent overfittiing and bias
# 
# GridSearchCV is used with a TimeSeriesSplit to fine-tune the models' hyperparameters and validate their performance using time-based cross-validation.

# %%
# Fit models using GridSearchCV and store best models
best_models = {}
results = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    results[name] = grid_search.best_estimator_.predict(X_test)

# %% [markdown]
# Model evaluation

# %%
# Evaluate all models
for name, predictions in results.items():
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{name}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}\n")


