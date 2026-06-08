import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class translator:
    def __init__(self, H, E):
        super().__init__()
        self.eng_embedding = nn.Embedding(eng_vocab_size, E)
        self.sp_embedding = nn.Embedding(sp_vocab_size, E)
        self.encoder = nn.GRU(E, H, 1, bidirectional = True, batch_first = True)
        self.attention = nn.Linear(3 * H, 1)
        self.decoder = nn.GRU(E + 2 * H, H, 2, batch_first = True) 
        self.fc_out = nn.Linear(H, sp_vocab_size)
        
    def forward(self, input, target, target_len, state=None):
        #input is B x T if we have T tokens per input
        batch_size = input.shape[0]
        embedding = self.eng_embedding(input) #B x T x E
        states, hidden = self.encoder(embedding, state)
        hidden = hidden[0:1] + hidden[1:2]
        #states is like B x T x 2H, hidden is like 1 x B x H
        #unsqueeze adds the given dimension, squeeze drops it
        outputs = []
        for t in range(target_len):
            if (t == 0):
                #start with sos token
                prev_output = torch.full((batch_size,), 1, dtype=torch.long)  
            else:
                #at some point this will need to address the teacher stuff
                prev_output = target[:, t - 1]
                
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
            outputs.append(logits)
        return torch.stack(outputs, dim=1)
        
        
pairs = []
with open("spa.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            eng, spa = parts[0], parts[1]
            pairs.append((eng, spa))

eng_vocab_size = 0 #number of words in english dictionary
sp_vocab_size = 0 #number of words in spanish dictionary
chunk_size = 0 #number of tokens per input
batch_size = 0 