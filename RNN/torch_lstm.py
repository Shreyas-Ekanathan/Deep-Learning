import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu
print(device)

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input, state=None):
        out, state = self.lstm(input, state)
        logits = self.fc(out)
        return logits, state

def sample(model, char_to_idx, idx_to_char, seed_char, length, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # start from a seed character
        idx = char_to_idx[seed_char]
        x = F.one_hot(torch.tensor([[idx]]), vocab_size).float().to(device)
        state = None
        result = [seed_char]
        for i in range(length):
            logits, state = model(x, state)
            probs = F.softmax(logits[0, 0] / temperature, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            result.append(idx_to_char[idx])
            # feed predicted character back in
            x = F.one_hot(torch.tensor([[idx]]), vocab_size).float().to(device)
        
    model.train()
    return ''.join(result)


with open('RNN/textbook.txt', 'r') as f:
    text = f.read()

import string
text = ''.join(c for c in text if c in string.printable)

# vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# encode entire text as integers
data = np.array([char_to_idx[c] for c in text])


batch_size = 64
chunk_size = 200
trim = len(data) - (len(data) % batch_size)
data = data[:trim]
streams = data.reshape(batch_size, -1)
num_epochs = 400
state = None

model = LSTM(256, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    model.train()
    for i in range(0, streams.shape[1] - chunk_size, chunk_size):
        inputs  = torch.tensor(streams[:, i:i+chunk_size]).to(device)
        targets = torch.tensor(streams[:, i+1:i+chunk_size+1]).to(device)
        x = torch.nn.functional.one_hot(inputs, vocab_size).float().to(device)
        optimizer.zero_grad()
        logits, state = model(x, state)
        state = tuple(s.detach() for s in state)
        loss = loss_fn(logits.permute(0, 2, 1), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if (i % 5000 == 0):
            print("Loss = ", loss.item())
    print("Epoch ", epoch)
    print(sample(model, char_to_idx, idx_to_char, seed_char='T', length=300))
    scheduler.step()
    

print("Final Sample!")
print(sample(model, char_to_idx, idx_to_char, seed_char='T', length=1000))