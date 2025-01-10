# README

## Project Title: **Volume-Oriented Financial Trading Strategy using CNN and Signal Thresholds**

### Overview
This project aims to develop and backtest a volume-based financial trading strategy using a Convolutional Neural Network (CNN). The model analyzes financial time series data with features derived from price and volume indicators to generate buy, sell, and hold signals. Additionally, the project incorporates thresholds to control the confidence of signals for better risk management.

### Features
- **Data Preprocessing**: Prepares raw financial data, computes technical indicators, and integrates them into the dataset.
- **CNN Model**: Trains a convolutional neural network to predict buy, sell, and hold signals based on the processed indicators.
- **Signal Thresholds**: Introduces thresholds to categorize predictions into strong buy/sell and hold for enhanced control over trades.
- **Backtesting**: Evaluates the strategy’s performance against a buy-and-hold benchmark.
- **Visualization**: Generates comprehensive visualizations of portfolio performance and trading signals.

### Key Indicators Used
1. **Volume-Based Indicators**:
   - Volume
   - Money Flow Volume (MFV)
2. **Price-Based Indicators**:
   - Average True Range (ATR)
   - Momentum (5-day)
   - True Range
3. **Custom Signal Thresholds**:
   - Buy: (0, 1), closer to 1 indicates strong buy.
   - Sell: (-1, 0), closer to -1 indicates strong sell.
   - Hold: (-0.25, 0.25), indicates neutral signals.

The model has more indicators on hand but according to the correlation matrix (correlation_matrix_analysis.py), only these indicators were used.
  

### File Structure
- **feature_extraction.py**: Script for cleaning and preprocessing financial data and calculating indicators.
- **data_preprocessing.py**: Contains the CNN architecture and training logic. Implements thresholds for buy, sell, and hold signals.
- **backtesting_volume_trading_strategy.py**: Backtests the generated signals and compares them against a buy-and-hold strategy. Visualize the results and compare them to Buy and Hold Strategy.
- **cnn_model_visualization.py**: Creates visualizations for CNN model.
- **datasets/**: Contains the input financial data files.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/CNN_Trading_Strategy.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CNN_Trading_Strategy
   ```


### Usage
1. **Prepare Data**:
   - Place your raw financial data (CSV format) in the `datasets/` folder.
   - Run the preprocessing script:
     ```bash
     python data_preprocessing.py
     ```
2. **Train the CNN**:
   - Run the training script:
     ```bash
     python cnn_model.py
     ```

4. **Backtest the Strategy**:
   - Evaluate the generated signals:
     ```bash
     python backtesting.py
     ```


### Threshold Implementation
The model categorizes predictions into **strong buy**, **strong sell**, and **hold** based on confidence thresholds derived from the softmax outputs of the CNN:
- Signals close to 1 or -1 are classified as strong buy/sell.
- Signals near zero indicate low confidence and are interpreted as hold.

### Performance Metrics
The project evaluates the model’s performance using:
- **Cumulative Return**: Measures total portfolio growth.
- **Sharpe Ratio**: Assesses risk-adjusted returns.
- **Max Drawdown**: Quantifies the largest portfolio loss from a peak.

### Results
#### Example Metrics
- Final Portfolio Value: $X,XXX
- Cumulative Return: XX.XX%
- Sharpe Ratio: X.XX
- Max Drawdown: -XX.XX%

### Future Improvements
- **Incorporate Additional Indicators**: Add features like RSI, MACD, or Bollinger Bands.
- **Optimize Thresholds**: Dynamically adjust thresholds based on market conditions.
- **Integrate Risk Management**: Factor in stop-loss and take-profit mechanisms.
- **Parameter Fine Tuning**: Work around with the learning rate, input size, and other CNN parameters.


### Contributing
Feel free to fork this repository and submit pull requests. Contributions to enhance the model, indicators, or performance metrics are welcome!

### License
This project is licensed under the MIT License.

### Contact
For any questions or suggestions, reach out to goralanil@gmail.com .

