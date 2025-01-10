import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv("daily_ohlc_volume.csv", index_col=0, parse_dates=True, delimiter=',')

# Ensure numeric columns
columns_to_convert = ["Close", "High", "Low", "Open", "Volume"]
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN values after conversion
data.dropna(inplace=True)

# Add SMA and EMA
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

# Add Bull/Bear Indicator
data['BullBear'] = np.where(data['Close'] > data['SMA_200'], 1, -1)

# Add other indicators
data['High-Low'] = data['High'] - data['Low']
data['High-Close'] = np.abs(data['High'] - data['Close'].shift())
data['Low-Close'] = np.abs(data['Low'] - data['Close'].shift())
data['True_Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
data['ATR_14'] = data['True_Range'].rolling(window=14).mean()

# Add daily return and direction
data['Daily_Return'] = data['Close'].pct_change()
data['Direction'] = np.where(data['Daily_Return'] > 0, 1, 0)

# Add Money Flow Volume (OBV and CMF)
data['Money_Flow_Volume'] = (data['Close'] - data['Low'] - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
data['CMF'] = data['Money_Flow_Volume'].rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()

# Add Momentum
data['Momentum_5'] = data['Close'] - data['Close'].shift(5)

# Add Volatility
data['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std()

# RSI Calculation
def calculate_rsi(series, window=14):
    delta = series.diff()  # Change in price
    gain = np.where(delta > 0, delta, 0)  # Positive gains
    loss = np.where(delta < 0, -delta, 0)  # Negative losses

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    # Handle division by zero
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)

# Apply RSI to the "Close" column
data["RSI"] = calculate_rsi(data["Close"], window=14)



# Save the dataset with all indicators
data.to_csv("SPY_data_with_all_indicators.csv")

print("Dataset with all indicators has been saved as 'SPY_data_with_all_indicators.csv'.")
