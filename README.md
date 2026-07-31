# Deep Learning (from scratch)

The goal of this work was to explore the trajectory of deep learning in depth (get it?). To that end, I focused on understanding the underlying mechanisms that drive deep learning, instead of just learning how to make function calls. That desire is reflected in my foundational projects, where I implemented an MLP, CNN, and RNN from scratch, including rolling out backprop. From there, I started branching into modern deep learning, exploring the state of the art architectures and trying out some projects of my own interest. The result is this portfolio. 

This is a portfolio built from the ground up, as I started learning ML in the summer of 2026. This repo was started in May 2026, and has been growing ever since.

The foundational architectures are each implemented twice: once from scratch in numpy with hand-derived backprop, then again in PyTorch. 

The larger projects build on that toward applied and research-adjacent work. I plan on continuing to develop and enhance the projects here. 

**On AI:** All of the models and training were done by me. The only places that AI helped in this work were 1) in generating evaluation scripts, since modern coding agents are good at this and it has somewhat less learning value for me, and 2) in debugging, both with respect to bugs and some model failures. When this was done, I made sure that I understood all of the work, and heavily AI-written bodies of code are cited as such. It is ironic that a repo trying to learn how frontier AI works makes sparing use of it, but as Feynman said, "What I cannot create, I do not understand," hence the existence of this whole repo. 

## Main projects

Each of the below projects has its own readme that details more of the project design. 

- **[Kolmogorov flow with neural ODEs](Neural-ODEs/kolmogorov/)**: write a pseudospectral Navier-Stokes solver for a chaotic 2D turbulent flow, then learn its dynamics as a convolutional latent neural ODE. Benchmarked per lead time against persistence and climatology the way a weather model is: MSE 0.20 against 0.93 for the mean field, anomaly correlation still 0.76 at the end of the window. Version 1 treated viscosity as a constant, while a follow up treated it as a variable for the model to learn from. This project sits where my numerics background meets scientific machine learning, the learned-dynamics problem underlying weather and climate emulation.
- **[GNN interatomic potentials](GNN/)**: learn a 2D Lennard-Jones energy with a GNN, get forces by autograd, then run the learned potential as a molecular dynamics sim. GCN vs GAT vs force-trained GAT, compared on force RMSE.
- **[MRI to CT translation](MRI2CT/)**: six image-translation architectures (U-Net, TransUNet, pix2pix, CycleGAN, conditional diffusion, latent-space alignment) benchmarked on paired medical slices. TransUNet wins (val L1 ~0.024). Includes three diffusion samplers (DDPM, DDIM, DPM-Solver-2). 
- **[Acrobot control](RL/acrobot-control/)**: from-scratch DQN and PPO swing-up (both solved), then continuous-action PPO for swing-up and balance, finished with an LQR handoff that actually holds the top. (Note that `RL/` also has counterfactual regret minimization for poker: tabular CFR and Deep CFR.)

## Other projects

These are some more traditional projects that people exploring deep learning follow, and I did the same.

- **[Translation](translation/)**: English to Spanish seq2seq. GRU with Bahdanau attention, and a Transformer built from scratch (custom multi-head attention, not nn.Transformer). Attention heatmaps included.
- **[VAE](VAE/)**: convolutional VAE and class-conditional VAE on CIFAR-10. Reparameterization, KL annealing, latent interpolation, t-SNE. See the readme.

## Foundations

Each project is implemented from scratch in numpy first, then rebuilt in PyTorch. The goal of these projects was to understand the internals of these algorithms fully.

- **[linear regression](linear_regression/)**: normal equations and gradient descent.
- **[MLP](MLP/)**: hand-derived backprop. The second version adds Adam, batchnorm, dropout, and L2 by hand on a spiral classifier.
- **[CNN](CNN/)**: convolution via im2col with manual conv and maxpool backprop, on MNIST.
- **[RNN](RNN/)**: vanilla RNN with backprop-through-time for character-level text, then an LSTM version using PyTorch.
