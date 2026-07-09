import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import datagen2 as datagen


#compare all 3 models
#partially claude written
def load_class(path, name):
    src = open(path).read()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    ns = {'torch': torch, 'nn': nn, 'F': F, 'hidden_dim': 64}
    exec(ast.get_source_segment(src, node), ns)
    return ns[name]


GCN = load_class('GNN/gcn.py', 'GCN')
GAT = load_class('GNN/gatv2.py', 'GAT')  # same architecture for both GATs

torch.manual_seed(0)
np.random.seed(0)
train = datagen.DATASET(n=10000)
test = datagen.DATASET(n=1000, energy_mean=train.energy_mean, energy_std=train.energy_std, force_std=train.force_std)
ESTD, FSTD, EMEAN = train.energy_std, train.force_std, train.energy_mean
print(f"energy_std={ESTD:.3f}, force_std={FSTD:.3f}")

def loader(ds, shuffle):
    g = torch.Generator(); g.manual_seed(0)  
    return DataLoader(ds, batch_size=100, shuffle=shuffle, generator=g)

def build_adj(pos):
    return torch.norm(pos.unsqueeze(2) - pos.unsqueeze(1), dim=-1)

def train_model(model, use_force, epochs=40, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    trl = loader(train, shuffle=True)
    for ep in range(epochs):
        for nf, pos, ne, nfrc in trl:
            if use_force: #only gat v2
                pos = pos.requires_grad_(True)
                e_pred = model(nf, build_adj(pos))
                f_pred = -torch.autograd.grad((e_pred * ESTD).sum(), pos, create_graph=True)[0] / FSTD
                loss = F.mse_loss(e_pred, ne) + 4 * F.mse_loss(f_pred, nfrc)
            else:
                e_pred = model(nf, build_adj(pos))
                loss = F.mse_loss(e_pred, ne)
            loss.backward(); opt.step(); opt.zero_grad()
        sched.step()
    return model

def evaluate(model):
    tel = loader(test, shuffle=False)
    se_e = n_e = se_f = n_f = 0
    for nf, pos, ne, nfrc in tel:
        pos = pos.requires_grad_(True)
        e_pred = model(nf, build_adj(pos))
        f_pred_phys = -torch.autograd.grad((e_pred * ESTD).sum(), pos)[0]  # physical units
        true_f_phys = nfrc * FSTD
        se_f += ((f_pred_phys - true_f_phys) ** 2).sum().item(); n_f += true_f_phys.numel()
        e_pred_phys = (e_pred * ESTD + EMEAN).detach()
        true_e_phys = ne * ESTD + EMEAN
        se_e += ((e_pred_phys - true_e_phys) ** 2).sum().item(); n_e += true_e_phys.numel()
    return (se_e / n_e) ** 0.5, (se_f / n_f) ** 0.5


configs = [
    ("GCN  (energy only)",  GCN(5, 1, 64), False),
    ("GAT  (energy only)",  GAT(5, 1, 64), False),
    ("GAT  (energy+force)", GAT(5, 1, 64), True),
]

results = []
for name, model, use_force in configs:
    print(f"training {name} ...", flush=True)
    train_model(model, use_force)
    e_rmse, f_rmse = evaluate(model)
    results.append((name, e_rmse, f_rmse))

print()
for name, e_rmse, f_rmse in results:
    print(f"{name}: energy RMSE {e_rmse:.3f}, force RMSE {f_rmse:.3f}")


# energy_std=7.559, force_std=32.922
# training GCN  (energy only) ...
# training GAT  (energy only) ...
# training GAT  (energy+force) ...

# GCN  (energy only): energy RMSE 1.753, force RMSE 14.147
# GAT  (energy only): energy RMSE 0.309, force RMSE 3.930
# GAT  (energy+force): energy RMSE 0.440, force RMSE 2.653