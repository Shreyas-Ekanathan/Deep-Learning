import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#same datagen as before
def generate_data(N):
    D = 3
    K = 10
    n = int(N / K)
    X = np.zeros((n*K, D))
    y = np.zeros(n*K, dtype='uint8')
    for j in range(K):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0.0, 1, n)
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2 # azimuthal angle
        phi = np.linspace(j*np.pi/K, (j+1)*np.pi/K, n) + np.random.randn(n)*0.1 # polar angle
        X[ix] = np.c_[
            r*np.sin(phi)*np.cos(t) + np.random.uniform(-0.1, 0.1, size=n),
            r*np.sin(phi)*np.sin(t) + np.random.uniform(-0.1, 0.1, size=n),
            r*np.cos(phi) + np.random.uniform(-0.1, 0.1, size=n)
        ]
        y[ix] = j
    y_onehot = np.zeros((n*K, K))
    y_onehot[np.arange(n*K), y] = 1
    return X, y

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

model = MLP(3, 128, 10)
N = 25000
train_set_input, train_set_output = generate_data(N)
test_set_input, test_set_output = generate_data(1000)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

#data handling
X = torch.tensor(train_set_input, dtype=torch.float32)
Y = torch.tensor(train_set_output, dtype=torch.long)
dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size=250, shuffle=True)

test_x = torch.tensor(test_set_input, dtype=torch.float32)
test_y = torch.tensor(test_set_output, dtype=torch.long)

val_dataset = TensorDataset(test_x, test_y)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)

num_epochs = 150
for epoch in range(num_epochs):
    # training
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
    
    # validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad(): # no gradient tracking needed
        for x_batch, y_batch in val_loader:
            logits = model(x_batch)
            val_loss += loss_fn(logits, y_batch).item()
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
    
    print(f"epoch {epoch} | val loss {val_loss/len(val_loader):.4f} | acc {correct/len(val_dataset):.4f}")    
