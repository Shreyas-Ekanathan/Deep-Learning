import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from kornia.losses import ssim_loss
import wandb
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

#so we are implementing a GCN
#messages are passed via aggregation just by summing surrounding hidden states
#and then apply a linear transformation to the message w the current hidden state to get the next hidden state

class GCN(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for i in range(4)
        ])
        
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_graph, adj_matrix):
        edges = self.edge_embedder(adj_matrix.unsqueeze(-1)).squeeze(-1)
        edges = edges.masked_fill(torch.eye(edges.shape[0], dtype=torch.bool, device=edges.device), 0.0) #0 out the diagonal
        
        input_graph = self.node_embedder(input_graph)
        for layer in self.layers[:-1]:
            transfer_layer, state_layer = layer[0], layer[1]
            message = edges @ transfer_layer(input_graph) 
            state_update = state_layer(input_graph)
            input_graph = F.relu(state_update + message)
            
        layer = self.layers[-1]
        transfer_layer, state_layer = layer[0], layer[1]
        message = edges @ transfer_layer(input_graph) 
        state_update = state_layer(input_graph)
        input_graph = state_update + message
        input_graph = self.out(input_graph)
        total_energy = input_graph.sum(dim=0)
        return total_energy
    
molecules = None

num_epochs = 10
model = GCN(0) #temporary! need to do data work next
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.MSELoss()
accum_steps = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    for i, mol in enumerate(molecules):
        energy_pred = model(mol.input_graph, mol.adj_matrix, mol.degree_matrix)
        loss = loss_fn(energy_pred, mol.target_energy) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    scheduler.step()
