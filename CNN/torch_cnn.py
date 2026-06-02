import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

batch_size = 100
num_batches = 600
reg_coef = 0.075
beta = 0.9
gamma = 0.99
alpha = 0.001
N = 60000

train_data = datasets.MNIST(root='./CNN', train=True, download=True, transform=transforms.ToTensor())
test_data  = datasets.MNIST(root='./CNN', train=False, download=True, transform=transforms.ToTensor())

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 18, 9, padding = 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 9, 5, padding = 2)
        self.fc1 = nn.Linear(441, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 441)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=100, shuffle=False)

for epoch in range(5):
    # validation
    cnn.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad(): # no gradient tracking needed
        for x_batch, y_batch in test_loader:
            logits = cnn(x_batch)
            val_loss += loss_fn(logits, y_batch).item()
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
    
    print(f"epoch {epoch} | val loss {val_loss/len(test_loader):.4f} | acc {correct/len(test_loader):.4f}")    

    # training
    cnn.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = cnn(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()