# MRI to CT Translation (from scratch)

We explore six architectures for translating MRI scans to CT scans, going from basic supervised to unsupervised to generative. Dataset is paired MRI/CT slices, 128x128 grayscale, normalized to [-1, 1].

## Models

- `U-NET/u-net.py`: baseline. Standard U-Net with 4 encoder/decoder blocks and skip connections. Supervised L1 + SSIM loss. Starting point for everything else.
- `pix2pix/pix2pix.py`: U-Net generator with a patch discriminator. Paired adversarial training (the discriminator sees (MRI, CT) pairs and scores realism, the generator has to fool it alongside minimizing L1). Based on the original pix2pix paper with some modernizing changes (instance norm, WGAN-GP).
- `cycleGAN/cycleGAN.py`: CycleGAN with WGAN-GP discriminators. Unpaired training (two U-Nets for MRI->CT and CT->MRI, two discriminators, cycle consistency loss). WGAN-GP keeps training stable; cycle loss stops the generators from ignoring the input.
- `Trans-U-NET/trans-unet.py`: TransUNet. CNN encoder (4 blocks, down to 8x8x256) feeds a 4-layer transformer bottleneck, then a CNN decoder with skip connections. The transformer attends over 64 spatial patches at the bottleneck. Supervised L1 + SSIM.
- `latent-spaces/disjoint-latent-spaces.py`: Two VAEs (one per modality) trained independently, then a small MLP maps MRI latent to CT latent, and the CT VAE decodes. Negative result, documented below.
- `diffusion/diffusion.py`: Conditional DDPM. U-Net denoiser takes noisy CT concatenated with MRI as a 2-channel input. Cosine noise schedule, sinusoidal time embeddings projected into each encoder/decoder block. Three samplers: DDPM (1000 steps), DDIM (50 steps, deterministic), DPM-Solver-2 (50 steps, second-order Heun's method).

## Results

| model | val/l1 |
|---|---|
| TransUNet | 0.024 |
| U-Net | ~0.03 |
| pix2pix | ~0.05 |
| CycleGAN | 0.08 |
| Diffusion (DDIM, 140 epochs) | 0.397 |
| Latent spaces | N/A |

TransUNet wins. Being directly supervised on paired data with an explicit L1 target is a large advantage on a small dataset. The transformer bottleneck helps with global structure that pure CNNs miss.

## On the latent space approach

The idea was to train disjoint VAEs and learn a mapping between latent spaces. The VAEs could not reconstruct the images well to begin with (the dataset is too sparse for the prior to learn). If the encoder does not capture the manifold, the MLP mapping does not matter. Gray blobs. Documented as a negative result.

## On diffusion

Training loss converges cleanly (0.042 to 0.014 over 140 epochs). DDIM samples produce recognizable anatomy, correct shapes and positions, just softer than the supervised models. DDPM accumulates too much stochastic error with an imperfect noise predictor. DPM-Solver-2 underperforms DDIM here because the second-order correction amplifies errors when the model is not fully converged, and the dataset is too small to train it to that point.

The broader takeaway: supervised methods work well on small medical imaging datasets. Generative approaches need more data to compete, not a failure of the architectures, just the wrong data regime for them.
