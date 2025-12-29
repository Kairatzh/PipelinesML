"""
Train CNN on MNIST dataset for use with OpenCV inference pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 64
EPOCHS = 5
LR = 0.001
MODEL_PATH = "model/mnist_cnn.pt"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    losses = 0

    for img, label in train_loader:
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        preds = model(img)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        losses += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {losses/len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("Model saved.")
