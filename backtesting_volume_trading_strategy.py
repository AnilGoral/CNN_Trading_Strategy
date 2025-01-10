import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data with signals
data = pd.read_csv("SPY_with_signals_and_confidence_threshold.csv", index_col=0, parse_dates=True)

# Parameters for backtesting
initial_balance = 100000  # Starting capital
position_size = 1 # Fraction of capital to allocate per trade
commission = 0.001  # Commission per trade (0.1%)
slippage = 0.001  # Slippage per trade (0.1%)

# Initialize portfolio variables
balance = initial_balance
positions = 0  # Number of shares held
portfolio_value = balance  # Value of portfolio (cash + positions)
portfolio_history = []  # To track portfolio value over time

# Track buy and sell points for visualization
buy_signals = []
sell_signals = []

# Backtesting loop
for i in range(1, len(data)):
    current_price = data.iloc[i]["Close"]
    signal = data.iloc[i]["Signal"]

    # Record buy and sell signals for visualization
    if signal == 1:  # Buy signal
        if balance > 0:  # Only buy if there's cash
            shares_to_buy = (balance * position_size) / current_price
            positions += shares_to_buy
            balance -= shares_to_buy * current_price * (1 + commission + slippage)
            buy_signals.append((data.index[i], current_price))
        else:
            buy_signals.append((data.index[i], np.nan))  # No cash to buy
    elif signal == -1:  # Sell signal
        if positions > 0:  # Only sell if holding positions
            balance += positions * current_price * (1 - commission - slippage)
            positions = 0
            sell_signals.append((data.index[i], current_price))
        else:
            sell_signals.append((data.index[i], np.nan))  # No positions to sell

    # Update portfolio value
    portfolio_value = balance + positions * current_price
    portfolio_history.append(portfolio_value)

# Calculate Buy-and-Hold portfolio
buy_and_hold_positions = initial_balance / data["Close"].iloc[0]  # Buy all at the start
buy_and_hold_value = buy_and_hold_positions * data["Close"]

# Extract buy and sell signals for plotting
buy_dates, buy_prices = zip(*[(date, price) for date, price in buy_signals if not np.isnan(price)])
sell_dates, sell_prices = zip(*[(date, price) for date, price in sell_signals if not np.isnan(price)])

# Calculate performance metrics
returns = np.diff(portfolio_history) / portfolio_history[:-1]
cumulative_return = (portfolio_value / initial_balance) - 1
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assuming daily data
max_drawdown = np.min(portfolio_history / np.maximum.accumulate(portfolio_history) - 1)

# Buy-and-Hold metrics
buy_and_hold_cumulative_return = (buy_and_hold_value.iloc[-1] / initial_balance) - 1
buy_and_hold_max_drawdown = np.min(buy_and_hold_value / buy_and_hold_value.cummax() - 1)

# Print results
print(f"Final Model Portfolio Value: ${portfolio_value:.2f}")
print(f"Model Cumulative Return: {cumulative_return:.2%}")
print(f"Model Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Model Max Drawdown: {max_drawdown:.2%}\n")

print(f"Final Buy-and-Hold Portfolio Value: ${buy_and_hold_value.iloc[-1]:.2f}")
print(f"Buy-and-Hold Cumulative Return: {buy_and_hold_cumulative_return:.2%}")
print(f"Buy-and-Hold Max Drawdown: {buy_and_hold_max_drawdown:.2%}")

# Plot 1: Compare Model Portfolio Value and Buy-and-Hold Value
plt.figure(figsize=(14, 7))
plt.plot(data.index[1:], portfolio_history, label="Model Portfolio Value", linestyle="-", color="orange", alpha=0.8)
plt.plot(data.index, buy_and_hold_value, label="Buy-and-Hold Portfolio Value", linestyle="--", color="blue", alpha=0.8)
plt.title("Portfolio Value Comparison: Model vs. Buy-and-Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid()
plt.show()

# Plot 2: Stock Price with Buy and Sell Signals
plt.figure(figsize=(14, 7))
plt.plot(data.index, data["Close"], label="Stock Price", color="blue", alpha=0.6)
plt.scatter(buy_dates, buy_prices, color="green", label="Buy Signal", marker="^", alpha=0.8)
plt.scatter(sell_dates, sell_prices, color="red", label="Sell Signal", marker="v", alpha=0.8)
plt.title("Stock Price with Buy and Sell Signals")
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.grid()
plt.show()
