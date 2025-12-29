import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1. Гиперпараметры
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 20
PATIENCE = 3
MODEL_PATH = "model.pt"


# 2. Датасет
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# 3. Архитектура модели
class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.

    This model consists of:
    - Two convolutional blocks with ReLU, BatchNorm, MaxPool.
    - Fully-connected classifier with dropout regularization.
    - Kaiming initialization to optimize convergence with ReLU.

    Forward Pass:
        Input  -> Conv/ReLU/BN/Pool -> Conv/ReLU/BN/Pool -> FC -> Output logits (10 classes)

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Computes forward pass.
    _init_weights(layer):
        Initializes layers using Kaiming He initialization for better gradient flow.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        """
        Applies custom weight initialization to Conv2D and Linear layers.

        Uses Kaiming Normal initialization which is optimal for networks with ReLU activation.
        Sets bias to zero to stabilize early training.
        """
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


model = CNN().to(device)

# 4. Потери, оптимизатор, шедулер
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


# 5. Early Stopping
best_loss = float('inf')
patient = 0

train_losses = []
val_losses = []
val_accs = []


# 6. Функции обучения и валидации
def train_one_epoch(epoch: int):
    """
    Executes a single training epoch.

    Parameters
    ----------
    epoch : int
        Current epoch number (for log formatting and reporting).

    Process Description
    -------------------
    - Sets model to training mode.
    - Iterates over mini-batches of training data.
    - For each batch:
        * Loads data to device (CPU/GPU).
        * Resets gradients.
        * Computes model output and loss.
        * Backpropagates gradients.
        * Updates weights using optimizer.
    - Tracks and returns average training loss.

    Returns
    -------
    float
        Average loss value across all batches for the epoch.
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}", unit="batch")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"batch_loss": loss.item()})

    return running_loss / len(train_loader)


def validate_one_epoch(epoch: int):
    """
    Performs validation for one epoch without gradient updates.

    Parameters
    ----------
    epoch : int
        Current epoch number (for log formatting).

    Process Description
    -------------------
    - Sets model to evaluation mode.
    - Disables gradient computation (torch.no_grad()).
    - Iterates over validation dataset.
    - Calculates:
        * Cumulative validation loss
        * Classification accuracy
    - Does NOT update weights.

    Returns
    -------
    tuple
        (avg_val_loss: float, accuracy: float)
    """
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            val_loss += criterion(preds, y).item()
            correct += (preds.argmax(dim=1) == y).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / len(test_data)

    return avg_val_loss, accuracy


# 7. Основной цикл обучения
"""
Main Training Loop
==================
This loop orchestrates the entire training lifecycle:

For each epoch:
    1) Train on all training batches.
    2) Validate on test set.
    3) Log and store metrics.
    4) Apply early stopping:
        - If validation loss improves: save checkpoint to disk.
        - If stagnates for N epochs (PATIENCE): interrupt training.
    5) Update learning rate scheduler.
"""
for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch(epoch)
    val_loss, accuracy = validate_one_epoch(epoch)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(accuracy)

    print(f"\nEpoch {epoch}: "
          f"Train Loss={train_loss:.4f}, "
          f"Val Loss={val_loss:.4f}, "
          f"Val Acc={accuracy*100:.2f}%")

    if val_loss < best_loss:
        best_loss = val_loss
        patient = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model improved — checkpoint saved.")
    else:
        patient += 1
        print(f"No improvement, patience = {patient}/{PATIENCE}")
        if patient >= PATIENCE:
            print("Early stopping activated.")
            break

    scheduler.step()


# 8. Загрузка лучшей модели
"""
Model Reload
============
Loads the best-performing model from disk (lowest validation loss)
for inference and production usage.
"""
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("\nBest model loaded.")


# 9. Инференс
x, y = test_data[0]
with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device)).argmax(dim=1).item()
print(f"Inference Example => Prediction: {pred}, True Label: {y}")
