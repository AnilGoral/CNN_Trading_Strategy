import yfinance as yf
import pandas as pd

# Define the ticker symbol and date range
ticker_symbol = "SPY"  # Example: SPY for S&P 500 ETF
start_date = "2018-01-01"
end_date = "2024-12-31"

# Fetch historical data
data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')

# Select necessary columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Handle missing data
data.dropna(inplace=True)

# Display the cleaned data
print(data.head())

# Save to CSV
data.to_csv("daily_ohlc_volume.csv", index=True)