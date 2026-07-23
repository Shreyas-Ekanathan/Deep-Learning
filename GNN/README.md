# GNN interatomic potentials (from scratch)

Learn the energy of a 2D Lennard-Jones system with a GNN, get forces by differentiating the energy, then run the learned potential forward as a molecular dynamics sim. Everything hand-rolled: message passing, attention, the data generator, the Verlet integrator.

Setup: 9 particles, 5 species, on a plane. Fully-connected graph, where nodes are one-hot species and edges are pairwise distances. Ground truth is the real LJ potential (Lorentz-Berthelot mixing) with analytic forces; trajectories come from velocity-Verlet. `datagen.py` builds the dataset (random configs + MD snapshots). Models predict total energy, and forces are `-∇E` via autograd, so the force field stays conservative for free. The learned forces then go back into Verlet to simulate a full trajectory against the true one.

## Models

- `gcn.py`: GCN. Edge embedder maps each distance to a scalar weight, aggregate neighbor states weighted by that plus a self term, 4 layers, sum per-node energies.
- `gat.py`: GAT. Attention over `[h_i, h_j, edge]` instead of fixed edge weights. The detail that matters: the message value carries the edge too (`transfer([h_j, edge])`), not just the attention weight. Without it the model can't route distance magnitude, since softmax normalizes it away, and it barely learns.
- `gatv2.py`: same GAT, trained on energy + force (`MSE(E) + 4·MSE(F)`), forces from autograd with `create_graph=True`. Energy-only training gives good energy but mediocre forces, and forces are what the sim runs on.

## Results

`compare_models.py` trains all three on one shared dataset and reports RMSE in physical units.

| model | energy RMSE | force RMSE |
|---|---|---|
| GCN (energy) | 1.75 | 14.1 |
| GAT (energy) | 0.31 | 3.9 |
| GAT (energy + force) | 0.44 | 2.7 |

Two clean wins: GAT beats GCN on forces by ~3.6x, and adding the force loss cuts force error another ~33% for a small energy cost. That energy-for-force trade is the standard result for ML potentials, and forces are what you want for dynamics.

## On the trajectory plots

Rolling the learned potential out and comparing paths (`*_results/`) is a bad way to rank models. The dynamics are chaotic, so any small force error blows up exponentially and every model's trajectory diverges at about the same rate regardless of quality. Force RMSE at fixed configs is the honest metric; the trajectory plots are mostly for show.
