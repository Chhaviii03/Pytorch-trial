# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load ECG200 dataset
def load_ucr(file_path):
    data = np.loadtxt(file_path)  # Load tab-separated data
    labels = data[:, 0]  # First column is label
    signals = data[:, 1:]  # Remaining columns are ECG signals
    return signals, labels

# File paths (update path if needed)
train_file = "UCR_Dataset/ECG200_TRAIN.txt"  # Replace with .ts if required
test_file = "UCR_Dataset/ECG200_TEST.txt"

# Load train and test data
X_train, y_train = load_ucr(train_file)
X_test, y_test = load_ucr(test_file)

# Convert labels from {-1, 1} to {0, 1}
y_train = (y_train == 1).astype(int)
y_test = (y_test == 1).astype(int)

# Normalize ECG signals
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create PyTorch Dataset
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader
train_dataset = ECGDataset(X_train_tensor, y_train_tensor)
test_dataset = ECGDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the Neural Network Model
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        self.fc1 = nn.Linear(96, 64)  # ECG200 has 96 features per sample
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # Binary classification (0 or 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
model = ECGClassifier()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Model Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot some ECG signals
plt.figure(figsize=(10, 5))
for i in range(5):  # Plot first 5 ECG signals
    plt.plot(X_train[i], label=f"Label: {y_train[i]}")
plt.legend()
plt.title("Sample ECG Signals from ECG200")
plt.show()






