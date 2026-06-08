import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import random

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu

class translator(nn.Module):
    def __init__(self, H, E, eng_vocab_size, sp_vocab_size):
        super().__init__()
        self.eng_embedding = nn.Embedding(eng_vocab_size, E)
        self.sp_embedding = nn.Embedding(sp_vocab_size, E)
        self.encoder = nn.GRU(E, H, 1, bidirectional = True, batch_first = True)
        self.attention = nn.Linear(3 * H, 1)
        self.decoder = nn.GRU(E + 2 * H, H, 2, batch_first = True) 
        self.fc_out = nn.Linear(H, sp_vocab_size)
        
    def forward(self, input, target, target_len, teacher_ratio, state=None):
        #input is B x T if we have T tokens per input
        batch_size = input.shape[0]
        embedding = self.eng_embedding(input) #B x T x E
        states, hidden = self.encoder(embedding, state)
        #states is like B x T x 2H, hidden is like 1 x B x H
        #unsqueeze adds the given dimension, squeeze drops it
        outputs = []
        for t in range(target_len):
            if (t == 0):
                #start with sos token
                prev_output = torch.full((batch_size,), 1, dtype=torch.long, device=input.device)
            else:
                #at some point this will need to address the teacher stuff
                prev_output =  target[:, t - 1] if random.random() < teacher_ratio else last_logits.argmax(dim = -1)
                
            sp_embedding = self.sp_embedding(prev_output) 
            
            s = hidden[-1]                                         
            s_expanded = s.unsqueeze(1).expand(-1, states.shape[1], -1)    
            attention_input = torch.cat([s_expanded, states], dim=-1)        
            attn = F.softmax(self.attention(attention_input), dim = 1) #B x T x 1, one val per hidden state
            #take weighted average now to build c
            c = (attn * states).sum(dim = 1)
            
            decoder_input = torch.cat([sp_embedding, c], dim = 1).unsqueeze(1)
            output, hidden = self.decoder(decoder_input, hidden)
            logits = self.fc_out(output.squeeze(1))
            last_logits = logits
            outputs.append(logits)
        return torch.stack(outputs, dim=1)
    
    def sample(self, input, state=None):
        with torch.no_grad():
            input = input.to(device)
            batch_size = input.shape[0]
            embedding = self.eng_embedding(input) #B x T x E
            states, hidden = self.encoder(embedding, state)
            #states is like B x T x 2H, hidden is like 1 x B x H
            #unsqueeze adds the given dimension, squeeze drops it
            prev_output = torch.full((batch_size,), 1, dtype=torch.long, device=input.device)
            outputs = [prev_output]
            while prev_output.item() != 2 and len(outputs) < 100:
                sp_embedding = self.sp_embedding(prev_output) 
                s = hidden[-1]                                         
                s_expanded = s.unsqueeze(1).expand(-1, states.shape[1], -1)    
                attention_input = torch.cat([s_expanded, states], dim=-1)        
                attn = F.softmax(self.attention(attention_input), dim = 1) #B x T x 1, one val per hidden state
                #take weighted average now to build c
                c = (attn * states).sum(dim = 1)
                
                decoder_input = torch.cat([sp_embedding, c], dim = 1).unsqueeze(1)
                output, hidden = self.decoder(decoder_input, hidden)
                logits = self.fc_out(output.squeeze(1))
                prev_output = logits.argmax(dim=-1)
                outputs.append(prev_output)
            return torch.stack(outputs, dim=1)

class TranslationPairs(Dataset):
    def __init__(self, pairs, eng_word2idx, spa_word2idx):
        self.pairs = [(encode(e, eng_word2idx), encode(s, spa_word2idx)) 
                      for e, s in pairs]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

        
pairs = []
with open("translation/spa.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            eng, spa = parts[0], parts[1]
            pairs.append((eng, spa))

def build_vocab(sentences, max_vocab=15000):
    counter = Counter()
    for sentence in sentences:
        for word in sentence.lower().split():
            counter[word] += 1
    
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for word, count in counter.most_common(max_vocab - 4):
        vocab[word] = len(vocab)
    
    return vocab

eng_sentences = [pair[0] for pair in pairs]
spa_sentences = [pair[1] for pair in pairs]

eng_word2idx = build_vocab(eng_sentences)
spa_word2idx = build_vocab(spa_sentences)  
eng_idx2word = {v: k for k, v in eng_word2idx.items()}
spa_idx2word = {v: k for k, v in spa_word2idx.items()}

def encode(sentence, vocab, max_len=40):
    tokens = sentence.lower().split()[:max_len]
    indices = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
    indices = [vocab["<SOS>"]] + indices + [vocab["<EOS>"]]
    return indices

def pad(sequences, pad_idx=0):
    max_len = max(len(s) for s in sequences)
    return [s + [pad_idx] * (max_len - len(s)) for s in sequences]
          
eng_vocab_size = len(eng_word2idx) #number of words in english dictionary
sp_vocab_size = len(spa_word2idx) #number of words in spanish dictionary

def collate_fn(batch):
    eng_batch, spa_batch = zip(*batch)
    eng_padded = pad(eng_batch)
    spa_padded = pad(spa_batch)
    return torch.tensor(eng_padded), torch.tensor(spa_padded)

dataset = TranslationPairs(pairs, eng_word2idx, spa_word2idx)
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True, num_workers=4, persistent_workers=True)
model = translator(512, 256, eng_vocab_size, sp_vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
num_epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        print("Epoch:", epoch)
        counter = 0
        for english, spanish in loader:
            english, spanish = english.to(device), spanish.to(device)
            optimizer.zero_grad()
            teacher_forcing_ratio = max(0.2, 1.0 - epoch * 0.025)  
            logits = model(english, spanish, spanish.shape[1], teacher_forcing_ratio).permute(0, 2, 1)
            loss = loss_fn(logits, spanish)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if counter == 100:
                print("Loss: ", loss.item())
                counter = 0
            counter += 1
        model.eval()
        sample_pairs = pairs[::len(pairs)//10][:10]
        for eng, _ in sample_pairs:
            encoded = torch.tensor([encode(eng, eng_word2idx)], dtype=torch.long)
            output = model.sample(encoded)[0].tolist()
            words = [spa_idx2word.get(i, "<UNK>") for i in output if i not in (0, 1, 2)]
            print(f"{eng} -> {' '.join(words)}")

    with open("translation/results.txt", "w") as f:
        sample_pairs = pairs[::len(pairs)//250][:250]
        model.eval()
        for eng, _ in sample_pairs:
            encoded = torch.tensor([encode(eng, eng_word2idx)], dtype=torch.long)
            output = model.sample(encoded)[0].tolist()
            words = [spa_idx2word.get(i, "<UNK>") for i in output if i not in (0, 1, 2)]
            f.write(f"{eng} -> {' '.join(words)}\n")

        sample_pairs2 = pairs[-100:]
        for eng, _ in sample_pairs2:
            encoded = torch.tensor([encode(eng, eng_word2idx)], dtype=torch.long)
            output = model.sample(encoded)[0].tolist()
            words = [spa_idx2word.get(i, "<UNK>") for i in output if i not in (0, 1, 2)]
            f.write(f"{eng} -> {' '.join(words)}\n")
