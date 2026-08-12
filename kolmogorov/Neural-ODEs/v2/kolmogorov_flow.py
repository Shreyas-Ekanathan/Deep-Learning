#what is the problem we are working on?
# 2d navier stokes with a sinusoidal body force injected into the system (Kolmogorov Flow)
# fairly chaotic system

from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchdiffeq import odeint
import torch.fft as fourier
import torch
import os

HERE = os.path.dirname(os.path.abspath(__file__)) #resolve paths against this file, not the cwd

#so, what are we changing in v2?
#lets make our model more general - we will pass in viscosity as a parameter to the model, and
#the model should learn how to integrate viscosity into its math, and it'll embed the viscosity into something
#that it can use to learn

#fluid is incompressible if density is constant throughout motion (divergence(u) = 0), stream function hence valid

#formulation:

# omega (vorticity) is the state variable that evolves over time
# FFT into exponential domain, and then solve the PDE (poisson equation) for the stream function
# from the stream function, derive horizontal nad vertical velocity (u and v respectively)
# IFFT to physical space
# inject forcing, FFT to solve the PDE navier stokes in the Fourier space, take a step

#we use the fourier space because all of the operations are cheap (derivative, poisson equation, etc)
# psuedospectral solver

def RHS(t, omega, nu):
    #the RHS of the PDE
    
    omega_hat = fourier.fft2(omega) #in fourier space now
    #Laplacian(stream function) = -omega hat, simple laplace equation wiht closed form solution in this space
    psi_hat = omega_hat / K2 
    
    u_hat = 1j * KY * psi_hat
    v_hat = -1j * KX * psi_hat
    
    domega_dx = 1j * KX * omega_hat #in fourier space
    domega_dy = 1j * KY * omega_hat
    
    u = fourier.ifft2(u_hat).real #back to real space for state vars
    v = fourier.ifft2(v_hat).real
    
    domega_dx = fourier.ifft2(domega_dx).real #same as above
    domega_dy = fourier.ifft2(domega_dy).real
    
    advection = u * domega_dx + v * domega_dy 
    advection_hat = fourier.fft2(advection) * mask
    
    domega_dt = -advection_hat - nu * K2 * omega_hat + f_hat #fourier space
    domega_dt = fourier.ifft2(domega_dt).real #real space
    return domega_dt

N = 64 #64x64 grid
L = 2 * np.pi
x_1d = torch.linspace(0, L, N+1)[:-1]      
y_1d = torch.linspace(0, L, N+1)[:-1]    
  
kx_1d = fourier.fftfreq(N, d=L/N) * 2 * np.pi 
ky_1d = fourier.fftfreq(N, d=L/N) * 2 * np.pi #for unit length, not rad

KX, KY = torch.meshgrid(kx_1d, ky_1d, indexing='ij') #corresponding wave numbers for each place on the grid
X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij') #physical space
K2 = KX ** 2 + KY ** 2
K2[0, 0] = 1.0
K_mag = torch.sqrt(K2) #wavenumber at each point
E = K_mag**4 * torch.exp(-2 * (K_mag / 8) **2) #energy distribution

kx_max = KX.abs().max()
ky_max = KY.abs().max()
mask = (KX.abs() < (2/3) * kx_max) & (KY.abs() < (2/3) * ky_max)

def f(x, y):
    return 100 * torch.cos(4 * y) #forcing

f_vals = f(X, Y)     
f_hat = fourier.fft2(f_vals)

#datagen time
#lets get 5 nu values, 75 initial conditions, 150 samples per for trianing
samples = []
nus = []
for nu in [0.03, 0.043, 0.06, 0.088, 0.12]:
    print(f"Running with nu = {nu}")
    for i in range(75):
        print(f"Running init condition {i + 1}")
        real_part = torch.randn(N, N)
        imag_part = torch.randn(N, N)
        random_hat = real_part + 1j * imag_part
        random_hat = random_hat * torch.sqrt(E)
        omega0 = fourier.ifft2(random_hat).real
        omega0 = omega0 / omega0.std() * 5 #normalized
        #this is our init condition, now we integrate to hit steady state
        
        t = torch.linspace(0, 50, 2) #tuned end time via enstrophy stabilization test
        omega0 = odeint(lambda s, w: RHS(s, w, nu), omega0, t, rtol = 1e-3, atol = 1e-6)[-1] #integrate until stabilized, and then proceed
        
        #now get samples
        T = 15
        num_snapshots = 150
        save_pts = torch.linspace(0, T, num_snapshots)
        omega = odeint(lambda s, w: RHS(s, w, nu), omega0, save_pts, rtol = 1e-4, atol = 1e-7)
        
        samples.append(omega)
        nus.append(nu)
        
all_runs = torch.stack(samples)
nus = torch.tensor(nus, dtype=torch.float32)
torch.save({"nu": nus, "omega": all_runs}, os.path.join(HERE, "kolmogorov_train_dataset.pt"))


test_samples = []
nus = []
for nu in [0.025, 0.0375, 0.06, 0.095, 0.15]:
    print(f"Running with nu = {nu}")
    for i in range(10):
        print(f"Running init condition {i + 1}")
        real_part = torch.randn(N, N)
        imag_part = torch.randn(N, N)
        random_hat = real_part + 1j * imag_part
        random_hat = random_hat * torch.sqrt(E)
        omega0 = fourier.ifft2(random_hat).real
        omega0 = omega0 / omega0.std() * 5 #normalized
        #this is our init condition, now we integrate to hit steady state
        
        t = torch.linspace(0, 50, 2) #tuned end time via enstrophy stabilization test
        omega0 = odeint(lambda s, w: RHS(s, w, nu), omega0, t, rtol = 1e-3, atol = 1e-6)[-1] #integrate until stabilized, and then proceed
        
        #now get samples
        T = 15
        num_snapshots = 150
        save_pts = torch.linspace(0, T, num_snapshots)
        omega = odeint(lambda s, w: RHS(s, w, nu), omega0, save_pts, rtol = 1e-4, atol = 1e-7)
        
        test_samples.append(omega)
        nus.append(nu)
        
all_runs = torch.stack(test_samples)
nus = torch.tensor(nus, dtype=torch.float32)
torch.save({"nu": nus, "omega": all_runs}, os.path.join(HERE, "kolmogorov_test_dataset.pt"))
