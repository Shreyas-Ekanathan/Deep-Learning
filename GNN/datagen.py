import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob

class molecule():
    def __init__(self, sigma, epsilon, mass):
        self.sigma = sigma #radius
        self.epsilon = epsilon #polarizability
        self.mass = mass
        
sample_molecules = [molecule(0.8, 0.7, 0.2), molecule(1.0, 1.0, 0.3), molecule(1.2, 1.4, 0.6), molecule(1.4, 1.9, 0.4), molecule(1.6, 2.5, 0.9)]

def compute_params(i, j):
    sigma = (sample_molecules[i].sigma + sample_molecules[j].sigma) / 2
    epsilon = np.sqrt(sample_molecules[i].epsilon * sample_molecules[j].epsilon)
    return sigma, epsilon
    
def compute_energy(positions, species):
    V = 0
    for i in range(9):
        for j in range(i + 1, 9):
            coord1 = positions[i]
            coord2 = positions[j]
            r = np.linalg.norm(np.array(coord1) - np.array(coord2))
            sigma, epsilon = compute_params(species[i], species[j])
            V_local = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
            V += V_local
            
    return V
    
def compute_forces(positions, species): 
    forces = np.zeros((9, 2))
    for i in range(9):
        for j in range(i + 1, 9):
            #force on molecule i from molecule j
            # negative of force on mol j by mol i by newtons 3rd law
            coord1 = positions[i]
            coord2 = positions[j]
            r = np.linalg.norm(np.array(coord1) - np.array(coord2))
            r_vec = np.array(coord1) - np.array(coord2)
            sigma, epsilon = compute_params(species[i], species[j])
            magnitude = 24 * epsilon / r * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) #derivative of energy
            forces[i] += magnitude * r_vec / r
            forces[j] -= magnitude * r_vec / r
    return forces

def generate_random_configs():
    #put 9 random molecules at random coordinates
    #return species identities and positions
    valid = False
    while (not valid): #loop until valid config found
        species = np.random.randint(0, 5, size=9)
        positions = np.random.uniform(-4, 4, size=(9, 2))
        valid = True
        for i in range(9):
            for j in range(i + 1, 9):
                sigma, epsilon = compute_params(species[i], species[j])
                coord1 = positions[i]
                coord2 = positions[j]
                r = np.linalg.norm(np.array(coord1) - np.array(coord2))
                if (r < 0.85 * sigma):
                    valid = False
                    break
            if (not valid):
                break
    
    return species, positions
    
def gen_trajectory():
    n = 1000 #1000 steps of trajectory
    dt = 0.001 #timestepping size
    species, positions = generate_random_configs()
    velocities = np.random.normal(0, 1, size=(len(species), 2))
    velocities -= velocities.mean(axis=0)  # remove net drift
    init_velocities = velocities
    forces = compute_forces(positions, species)
    
    masses = np.array([sample_molecules[species[i]].mass for i in range(9)])[:, None]  # shape (9, 1)
    pos_vector = []
    pos_vector.append(positions.copy())
    for t in range(1, n):
        positions += velocities * dt + 0.5 * forces / masses * (dt ** 2)
        new_forces = compute_forces(positions, species)
        velocities += 0.5 * (forces + new_forces)/ masses * dt
        forces = new_forces
            
        if (t % 10 == 0):
            pos_vector.append(positions.copy())
            
    return species, pos_vector, init_velocities


class DATASET(Dataset):
    def __init__(self, n=25000, energy_mean=None, energy_std=None):
        self.examples = []
        for i in range(n):
            species, positions = generate_random_configs()
            target_energy = compute_energy(positions, species)
            self.examples.append((species, positions, target_energy))

        for i in range(n // 100): #100 * 100 = 10000 more samples
            species, pos_vector, _ = gen_trajectory()
            for j in range(len(pos_vector)):
                target_energy = compute_energy(pos_vector[j], species)
                self.examples.append((species, pos_vector[j], target_energy))

        self.holdout_trajectory = gen_trajectory()

        energies = np.array([e for _, _, e in self.examples])
        # reuse the training set's stats when given, so train/test are normalized consistently
        self.energy_mean = energy_mean if energy_mean is not None else energies.mean()
        self.energy_std = energy_std if energy_std is not None else energies.std()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        species, positions, target_energy = self.examples[idx]
        node_features = torch.nn.functional.one_hot(torch.tensor(species), num_classes=5).float()  # one hot
        adj_list = np.zeros((9, 9), dtype=np.float32)
        for i in range(9):
            for j in range(i + 1, 9):
                #distance between pos i and pos j
                coord1 = positions[i]
                coord2 = positions[j]
                r = np.linalg.norm(np.array(coord1) - np.array(coord2))
                adj_list[i][j] = r
                adj_list[j][i] = r
        normalized_energy = (target_energy - self.energy_mean) / self.energy_std
        return node_features, adj_list, np.float32(normalized_energy)
    
