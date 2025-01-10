import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Parameters
TRAIN_WINDOW = 200
STEP_SIZE = 80
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 50
LOOKBACK_WINDOW = 20
DESIRED_THRESHOLD = 0.7  # Confidence threshold

# Load the data
data = pd.read_csv("SPY_data_with_all_indicators.csv", index_col=0, parse_dates=True, delimiter=';')

# Define features and target. These features are extracted from the correlation matrix analysis
features = [
    "Volume", 
    "Money_Flow_Volume", 
    "ATR_14", 
    "Momentum_5", 
    "True_Range",
]
target = "Target"
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)

scaler = StandardScaler()

# CNN Model Definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1_input_size = None
        self.fc1 = None  # Placeholder
        self.fc2 = nn.Linear(128, 2)  # Binary classification (Up/Down)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        if self.fc1_input_size is None:
            self.fc1_input_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(self.fc1_input_size, 128)
            self.fc1.to(x.device)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to create 2D samples
def create_2d_samples(X, y, lookback):
    samples, targets = [], []
    for i in range(lookback, len(X)):
        samples.append(X[i - lookback:i])
        targets.append(y[i])
    return np.array(samples), np.array(targets)

# Prepare features and target
X = data[features].values
y = data[target].values
X = scaler.fit_transform(X)

# Split data into train and test
train_set = data.iloc[:TRAIN_WINDOW]
test_set = data.iloc[TRAIN_WINDOW + STEP_SIZE:]

X_train, y_train = train_set[features].values, train_set[target].values
X_test, y_test = test_set[features].values, test_set[target].values

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create 2D samples
X_train_2d, y_train_2d = create_2d_samples(X_train, y_train, LOOKBACK_WINDOW)
X_test_2d, y_test_2d = create_2d_samples(X_test, y_test, LOOKBACK_WINDOW)

X_train_2d = torch.tensor(X_train_2d, dtype=torch.float32).unsqueeze(1)
y_train_2d = torch.tensor(y_train_2d, dtype=torch.long)
X_test_2d = torch.tensor(X_test_2d, dtype=torch.float32).unsqueeze(1)
y_test_2d = torch.tensor(y_test_2d, dtype=torch.long)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and Validation
best_f1 = 0
early_stopping_counter = 0
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_2d.to(device))
    loss = criterion(outputs, y_train_2d.to(device))
    loss.backward()
    optimizer.step()

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_2d.to(device))
        test_probs = F.softmax(test_outputs, dim=1).cpu().numpy()
        test_preds = np.argmax(test_probs, axis=1)
        test_f1 = f1_score(y_test_2d.cpu().numpy(), test_preds)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}, Test F1 Score: {test_f1:.4f}")

    if test_f1 > best_f1:
        best_f1 = test_f1
        early_stopping_counter = 0
        torch.save(model.state_dict(), "best_cnn_model.pth")
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

print(f"Best Test F1 Score: {best_f1:.4f}")

# Generate Signals with Confidence Levels
model.eval()
with torch.no_grad():
    all_samples = []
    for i in range(LOOKBACK_WINDOW, len(X)):
        all_samples.append(X[i - LOOKBACK_WINDOW:i])
    all_samples_tensor = torch.tensor(np.array(all_samples), dtype=torch.float32).unsqueeze(1).to(device)
    all_outputs = model(all_samples_tensor)  # Raw logits
    all_probs = F.softmax(all_outputs, dim=1).cpu().numpy()  # Probabilities
    all_preds = np.argmax(all_probs, axis=1)  # Predicted class

# Add signals and confidence levels to the dataset
data["Signal"] = 0
data["Buy_Confidence"] = 0.0
data["Sell_Confidence"] = 0.0

# Assign confidence levels
data.iloc[LOOKBACK_WINDOW:, data.columns.get_loc("Buy_Confidence")] = all_probs[:, 1]  # Probability of 'Buy' (class 1)
data.iloc[LOOKBACK_WINDOW:, data.columns.get_loc("Sell_Confidence")] = all_probs[:, 0]  # Probability of 'Sell' (class 0)

# Apply threshold logic
data.loc[
    (data["Buy_Confidence"] >= DESIRED_THRESHOLD) & (data["Sell_Confidence"] < DESIRED_THRESHOLD), "Signal"
] = 1  # Buy signal
data.loc[
    (data["Sell_Confidence"] >= DESIRED_THRESHOLD) & (data["Buy_Confidence"] < DESIRED_THRESHOLD), "Signal"
] = -1  # Sell signal

# Save to CSV
data.to_csv("SPY_with_signals_and_confidence_threshold.csv")
print(f"Signals with threshold {DESIRED_THRESHOLD} applied and saved to 'SPY_with_signals_and_confidence_threshold.csv'")
