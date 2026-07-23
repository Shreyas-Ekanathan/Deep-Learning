# Deep Learning (from scratch)

A portfolio built from the ground up. The foundational architectures are each implemented twice: once from scratch in numpy with hand-derived backprop, then again in PyTorch. The larger projects build on that toward applied and research-adjacent work. (Numerical-methods and SciML work lives in separate repos; this is the ML side.)

## Main projects

- **[MRI to CT translation](MRI2CT/)**: six image-translation architectures (U-Net, TransUNet, pix2pix, CycleGAN, conditional diffusion, latent-space alignment) benchmarked on paired medical slices. TransUNet wins (val L1 ~0.024). Includes three diffusion samplers (DDPM, DDIM, DPM-Solver-2) and a documented negative result. See the readme.
- **[GNN interatomic potentials](GNN/)**: learn a 2D Lennard-Jones energy with a GNN, get forces by autograd, then run the learned potential as a molecular dynamics sim. GCN vs GAT vs force-trained GAT, compared on force RMSE. See the readme.
- **[Acrobot control](RL/acrobot-control/)**: from-scratch DQN and PPO swing-up (both solved), then continuous-action PPO for swing-up and balance, finished with an LQR handoff that actually holds the top. See the readme. (`RL/` also has counterfactual regret minimization for poker: tabular CFR and Deep CFR.)

## Other projects

- **[Translation](translation/)**: English to Spanish seq2seq. GRU with Bahdanau attention, and a Transformer built from scratch (custom multi-head attention, not nn.Transformer). Attention heatmaps included.
- **[VAE](VAE/)**: convolutional VAE and class-conditional VAE on CIFAR-10. Reparameterization, KL annealing, latent interpolation, t-SNE. See the readme.

## Foundations (scratch, then PyTorch)

Each implemented from scratch in numpy first, then rebuilt in PyTorch.

- **[linear regression](linear_regression/)**: normal equations and gradient descent.
- **[MLP](MLP/)**: hand-derived backprop. The second version adds Adam, batchnorm, dropout, and L2 by hand on a spiral classifier.
- **[CNN](CNN/)**: convolution via im2col with manual conv and maxpool backprop, on MNIST.
- **[RNN](RNN/)**: vanilla RNN with backprop-through-time for character-level text, then an LSTM version using PyTorch.
