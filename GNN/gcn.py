import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from kornia.losses import ssim_loss
import wandb
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import GNN.datagen as datagen

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
        edges = edges.masked_fill(torch.eye(edges.shape[-1], dtype=torch.bool, device=edges.device), 0.0) #0 out the diagonal
        
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
        total_energy = input_graph.sum(dim=1)
        return total_energy.squeeze(-1)
    

num_epochs = 50
model = GCN(5, 1, 64) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

train_dataset = datagen.DATASET()
test_dataset = datagen.DATASET(1000, energy_mean=train_dataset.energy_mean, energy_std=train_dataset.energy_std)
train_loader = DataLoader(train_dataset, batch_size = 100, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 100, shuffle = False)

for epoch in range(num_epochs):
    for node_features, adj_list, true_energy in train_loader:
        energy_pred = model(node_features, adj_list)
        loss = loss_fn(energy_pred, true_energy)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    
    with torch.no_grad():
        total_loss = 0
        for node_features, adj_list, true_energy in test_loader:
            energy_pred = model(node_features, adj_list)
            loss = loss_fn(energy_pred, true_energy)
            total_loss += loss
            
        print(f"Epoch {epoch}, loss = {total_loss}")

#lets try out a full trajectory now, and see how we deviate

def simulate_trajectory(species, positions, init_velocities):
    node_features = torch.nn.functional.one_hot(torch.tensor(species), num_classes=5).float().unsqueeze(0)
    positions_t = torch.tensor(positions, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    velocities = torch.tensor(init_velocities, dtype=torch.float32).unsqueeze(0)
    masses = torch.tensor([datagen.sample_molecules[species[i]].mass for i in range(9)], dtype=torch.float32)[None, :, None]
    dt = 0.001
    sol = []

    def energy_and_forces(pos):
        diffs = pos.unsqueeze(2) - pos.unsqueeze(1)
        adj_list = torch.norm(diffs, dim=-1)  # differentiable
        energy_pred = model(node_features, adj_list)
        forces_pred = -torch.autograd.grad(energy_pred.sum(), pos)[0]
        return forces_pred

    forces_pred = energy_and_forces(positions_t)
    for t in range(1000):
        #step
        positions_t = (positions_t + velocities * dt + 0.5 * forces_pred / masses * (dt ** 2)).detach().requires_grad_(True)
        new_forces = energy_and_forces(positions_t)
        velocities = velocities + 0.5 * (forces_pred + new_forces) / masses * dt
        forces_pred = new_forces

        if (t % 10 == 0):
            sol.append(positions_t.detach().squeeze(0).numpy().copy())

    return sol


true_traj1 = train_dataset.holdout_trajectory
true_traj2 = test_dataset.holdout_trajectory

species1, pos_vector1, init_velocities1 = true_traj1
sol1 = simulate_trajectory(species1, pos_vector1[0], init_velocities1)

species2, pos_vector2, init_velocities2 = true_traj2
sol2 = simulate_trajectory(species2, pos_vector2[0], init_velocities2)

from GNN.plot import plot_trajectory_paths, plot_deviation

plot_trajectory_paths(pos_vector1, sol1, save_path="GNN/LJ_potentials/gcn_results/traj1_paths.png")
plot_deviation(pos_vector1, sol1, dt_per_snapshot=0.01, save_path="GNN/LJ_potentials/gcn_results/traj1_deviation.png")

plot_trajectory_paths(pos_vector2, sol2, save_path="GNN/LJ_potentials/gcn_results/traj2_paths.png")
plot_deviation(pos_vector2, sol2, dt_per_snapshot=0.01, save_path="GNN/LJ_potentials/gcn_results/traj2_deviation.png")

