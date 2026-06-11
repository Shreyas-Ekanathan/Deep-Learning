import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import random
import re
import matplotlib.pyplot as plt

#implement the transformer from attention is all you need

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        #h heads, that's why we have h * d_k or h * d_v
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h
        self.W_Q = nn.Linear(d_model, d_model, bias=False) #project to query dimension h times, stacked together
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False) #project back to model hidden dim

    def forward(self, Q, K, V, mask = None):
        B, T_Q, _ = Q.shape
        T_K = K.shape[1]
        d_k = self.d_k
        h = self.h
        d_v = self.d_v
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        Q = Q.view(B, T_Q, h, d_k).transpose(1, 2) #B x h x T x d_k
        K = K.view(B, T_K, h, d_k).transpose(1, 2)
        V = V.view(B, T_K, h, d_v).transpose(1, 2)
        
        vals = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        if mask is not None:
            vals = vals.masked_fill(mask == 0, -1e9) #become 0 after softmax
            
        attn_weights = F.softmax(vals, dim=-1) # B x h x T_Q x T_K
        self.attn_weights = attn_weights
        attn = attn_weights @ V # B x h x T_Q x d_v
        attn = attn.transpose(1, 2) #B x T x h x d_v
        attn = attn.contiguous().view(B, T_Q, h * d_v)
        
        return self.W_O(attn)
        
        
class EncoderUnit(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)   
                 
    def forward(self, input, src_mask):
        context = self.attn(input, input, input, src_mask) #self attention
        normed_context = self.norm1(context) + input
        ff_out = self.ff(normed_context)
        return self.norm2(ff_out) + normed_context #more traditional to do pre-LN
    
class Encoder(nn.Module):
    def __init__(self, d_model, h, N):
        super().__init__()
        self.layers = nn.ModuleList([EncoderUnit(d_model, h) for i in range(N)])
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
    
class DecoderUnit(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)   
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, input, encoder_output, target_mask, src_mask):
        context = self.self_attn(input, input, input, target_mask)
        normed_context = self.norm1(context) + input
        cross_attn_context = self.cross_attn(normed_context, encoder_output, encoder_output, src_mask)
        normed_cross_attn = self.norm2(cross_attn_context)+ normed_context
        ff_out = self.ff(normed_cross_attn)
        return self.norm3(ff_out)  + normed_cross_attn #more traditional to do pre-LN
    
class Decoder(nn.Module):
    def __init__(self, d_model, h, N):
        super().__init__()
        self.layers = nn.ModuleList([DecoderUnit(d_model, h) for i in range(N)])
    
    def forward(self, x, encoding, target_mask, src_mask):
        for layer in self.layers:
            x = layer(x, encoding, target_mask, src_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model, h, N, eng_vocab_size, sp_vocab_size):
        super().__init__()
        self.eng_embedding = nn.Embedding(eng_vocab_size, d_model)
        self.sp_embedding = nn.Embedding(sp_vocab_size, d_model)
        self.encoder = Encoder(d_model, h, N)
        self.decoder = Decoder(d_model, h, N)
        self.fc_out = nn.Linear(d_model, sp_vocab_size)
        self.d_model = d_model
        
    def padding_mask(self, seq):
        # seq: B x T, returns B x 1 x 1 x T
        return (seq != 0).unsqueeze(1).unsqueeze(2)

    def causal_mask(self, T, device):
        return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    
    def positional_encoding(self, T, device):
        d_model = self.d_model
        pe = torch.zeros(T, d_model, device=device)
        pos = torch.arange(0, T, device=device).unsqueeze(1) #0, 1, ..., T     
        div = torch.pow(10000, torch.arange(0, d_model, 2, device=device) / d_model) #even indices
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)  
        return pe.unsqueeze(0) 

    def forward(self, input, target):
        target_mask = self.padding_mask(target) & self.causal_mask(target.shape[1], input.device)
        src_mask = self.padding_mask(input)
        eng_embedding = self.eng_embedding(input) + self.positional_encoding(input.shape[1], input.device)
        encoding = self.encoder(eng_embedding, src_mask)
        sp_embedding = self.sp_embedding(target) + self.positional_encoding(target.shape[1], input.device)
        out = self.decoder(sp_embedding, encoding, target_mask, src_mask)
        return self.fc_out(out)    
                       
    def sample(self, input, max_len=200):
        with torch.no_grad():
            src_mask = self.padding_mask(input)
            encoding = self.encoder(self.eng_embedding(input) + self.positional_encoding(input.shape[1], input.device), src_mask)
            generated = torch.full((input.shape[0], 1), 1, dtype=torch.long, device=input.device)

            for i in range(max_len):
                decoder_input = self.sp_embedding(generated) + self.positional_encoding(generated.shape[1], input.device)
                target_mask = self.causal_mask(generated.shape[1], input.device)
                out = self.decoder(decoder_input, encoding, target_mask, src_mask)
                logits = self.fc_out(out)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == 2:
                    break
            
            return generated

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

def tokenize(sentence):
    return re.findall(r"\w+|[^\w\s]", sentence.lower())

def build_vocab(sentences, max_vocab=10000):
    counter = Counter()
    for sentence in sentences:
        for word in tokenize(sentence):
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
    tokens = tokenize(sentence)[:max_len]
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
model = Transformer(256, 8, 2, eng_vocab_size, sp_vocab_size).to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
num_epochs = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

if __name__ == "__main__":
    best_loss = float('inf')
    print(device)
    for epoch in range(num_epochs):
        model.train()
        print("Epoch:", epoch)
        avg_loss = 0
        for english, spanish in loader:
            english, spanish = english.to(device), spanish.to(device)
            optimizer.zero_grad()
            logits = model(english, spanish[:, :-1]).permute(0, 2, 1)
            loss = loss_fn(logits, spanish[:, 1:])
            loss.backward()
            avg_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        avg_loss /= len(loader)
        print(f"Avg loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'translation/transformer_results/best_model.pt')
            print("Checkpoint saved")
            
        scheduler.step()
        model.eval()
        sample_pairs = pairs[::len(pairs)//10][:10]
        for eng, _ in sample_pairs:
            encoded = torch.tensor([encode(eng, eng_word2idx)], dtype=torch.long).to(device)
            output_tensor = model.sample(encoded)
            output = output_tensor[0].tolist()
            words = [spa_idx2word.get(i, "<UNK>") for i in output if i not in (0, 1, 2)]
            print(f"{eng} -> {' '.join(words)}")
    
    with open("translation/transformer_results/results.txt", "w") as f:
        sample_pairs = pairs[::len(pairs)//250][:250]
        model.eval()
        for eng, _ in sample_pairs:
            encoded = torch.tensor([encode(eng, eng_word2idx)], dtype=torch.long).to(device)
            output_tensor = model.sample(encoded)
            output = output_tensor[0].tolist()
            words = [spa_idx2word.get(i, "<UNK>") for i in output if i not in (0, 1, 2)]
            f.write(f"{eng} -> {' '.join(words)}\n")

        sample_pairs2 = pairs[-100:]
        for eng, _ in sample_pairs2:
            encoded = torch.tensor([encode(eng, eng_word2idx)], dtype=torch.long).to(device)
            output_tensor = model.sample(encoded)
            output = output_tensor[0].tolist()
            words = [spa_idx2word.get(i, "<UNK>") for i in output if i not in (0, 1, 2)]
            f.write(f"{eng} -> {' '.join(words)}\n")

    #for plotting
    def plot_attention(input_sentence, output_sentence, attn_matrix, filename):
        attn_matrix = attn_matrix[:len(output_sentence), :len(input_sentence)]

        fig, ax = plt.subplots(figsize=(len(input_sentence), len(output_sentence) * 0.6))
        im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(input_sentence)))
        ax.set_yticks(range(len(output_sentence)))
        ax.set_xticklabels(input_sentence, rotation=45, ha='left', fontsize=10)
        ax.set_yticklabels(output_sentence, fontsize=10)
        ax.xaxis.set_label_position('top')
        
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

    sample_pairs3 = pairs[::len(pairs)//10][:10]
    for (idx, (eng, _)) in enumerate(sample_pairs3):
        encoded = torch.tensor([encode(eng, eng_word2idx)], dtype=torch.long).to(device)
        output_tensor = model.sample(encoded)
        output = output_tensor[0].tolist()
        words = [spa_idx2word.get(i, "<UNK>") for i in output if i not in (0, 1, 2)]
        cross_attn = model.decoder.layers[0].cross_attn.attn_weights
        cross_attn = cross_attn[0].mean(dim=0).cpu().numpy()
        plot_attention(tokenize(eng), words, cross_attn, f"translation/transformer_results/attn_{idx}.png")
