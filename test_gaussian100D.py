import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from optimizer import *
from sklearn.decomposition import PCA


# Generate 2D Gaussian dataset
def generate_data(n_samples=200, n_features=100, noise_rate=0):
    mean_0, cov_0 = np.zeros(n_features), np.eye(n_features)
    mean_1, cov_1 = np.ones(n_features), np.eye(n_features)
    
    X0 = np.random.multivariate_normal(mean_0, cov_0, n_samples // 2)
    X1 = np.random.multivariate_normal(mean_1, cov_1, n_samples // 2)
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    
    if noise_rate != 0:
        noise_indices = np.random.choice(n_samples, int(noise_rate * n_samples), replace=False)
        y[noise_indices] = 1 - y[noise_indices]
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Create dataset without noise
n_features = 3
X, y = generate_data(n_features=n_features)
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Create dataset with label noise
X_noise, y_noise = generate_data(n_features=n_features, noise_rate=0.25)
dataset_noise = TensorDataset(X_noise, y_noise)
train_dataset_noise, _ = torch.utils.data.random_split(dataset_noise, [train_size, test_size])
train_loader_noise = DataLoader(train_dataset_noise, batch_size=100, shuffle=True)

# Define Linear Regression model
class LinearModel(nn.Module):
    def __init__(self, n_features=100):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
        # Manually initialize weights
        # self.linear.weight.data = torch.tensor([[-2, -2.0]], dtype=torch.float32)
        # self.linear.bias.data = torch.tensor([0.0], dtype=torch.float32)
    
    def forward(self, x):
        return self.linear(x).squeeze(1)

# Training the model
def train_model_sgd(train_loader):
    EPOCHS = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel(n_features=n_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    trajectory = []
    losses = []
    
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            trajectory.append(model.linear.weight.data.clone().cpu().numpy().flatten())
            losses.append(loss.item())
    
    return model, np.array(trajectory), np.array(losses)

# Training the model
def train_model_sam(train_loader):
    EPOCHS = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = SAM(model.parameters(), rho=0.05, lr=0.2, weight_decay=0, momentum=0)
    trajectory = []
    losses = []
    
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(X_batch), y_batch).backward()
            optimizer.second_step(zero_grad=True)
            trajectory.append(model.linear.weight.data.clone().cpu().numpy().flatten())
            losses.append(loss.item())
    
    return model, np.array(trajectory), np.array(losses)

def train_model_saner(train_loader):
    EPOCHS = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = SANER(model.parameters(), rho=0.05, lr=0.2, weight_decay=0, momentum=0, group='B', condition=0)
    trajectory = []
    losses = []
    
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(X_batch), y_batch).backward()
            optimizer.second_step(zero_grad=True)
            trajectory.append(model.linear.weight.data.clone().cpu().numpy().flatten())
            losses.append(loss.item())
    return model, np.array(trajectory), np.array(losses)

def train_model(train_loader, optimizer='sgd'):
    if optimizer == 'sgd':
        return train_model_sgd(train_loader)
    elif optimizer == 'sam':
        return train_model_sam(train_loader)
    elif optimizer == 'saner':
        return train_model_saner(train_loader)

# Train models and capture trajectories
opt_name = 'sgd'
model_clean, trajectory_clean, losses_clean = train_model(train_loader, opt_name)
model_noisy, trajectory_noisy, losses_noisy = train_model(train_loader_noise, opt_name)

# PCA for visualization
def visualize_loss_landscape(trajectory, losses, title, filename):
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(trajectory)
    
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], c=losses, cmap='viridis', marker='o', s=20)
    plt.colorbar(sc, label='Loss')
    plt.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], marker='s', color='blue', label='Init', s=100)
    plt.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], marker='*', color='red', label='Final', s=150)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend()
    plt.savefig(filename, format='pdf')
    # plt.show()
    
# Plot and save loss landscapes
visualize_loss_landscape(trajectory_clean, losses_clean, "Loss Landscape (Clean Data)", f"{n_features}D_loss_landscape_clean_{opt_name}.pdf")
visualize_loss_landscape(trajectory_noisy, losses_clean, "Loss Landscape (Noisy Data)", f"{n_features}D_loss_landscape_noisy_{opt_name}.pdf")
