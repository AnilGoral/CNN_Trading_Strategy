import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("SPY_data_with_all_indicators.csv", index_col=0, parse_dates=True, delimiter=';')

# Compute the correlation matrix
correlation_matrix = data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Indicators and Volume")
plt.show()

# Extract correlation of indicators with Volume
volume_correlation = correlation_matrix["Volume"].sort_values(ascending=False)
print("\nCorrelation of Indicators with Volume:")
print(volume_correlation)
