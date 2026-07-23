# VAE on CIFAR-10 (from scratch)

Convolutional VAE on CIFAR-10, then a class-conditional version. The point was to build the generative machinery and then poke at the latent space: reconstructions, interpolations, sampling, and t-SNE.

Setup: 3-conv encoder down to a 256-dim latent, 3 transposed-conv decoder. ELBO is BCE reconstruction plus KL, with beta annealed up to 0.3 over the first half of training, so it learns to reconstruct before the KL pressure kicks in. Adam, cosine LR, 75 epochs.

## Results

Reconstructions (`data_analysis/reconstructions2.png`) capture the coarse shape, color, and layout of each image but come out blurry. That is expected for a VAE with a pixel reconstruction loss on natural images: the Gaussian latent plus an averaging loss wash out high-frequency detail. You can tell a horse from a dog from a truck, just softly.

Interpolations between two encoded images (`data_analysis/interpolation*.png`) are smooth with no jumps, which is the sign the latent space is continuous rather than memorizing points.

The t-SNE of the latent means (`data_analysis/tsne.png`) shows only weak class structure. Vehicles and animals pull apart a little, but the classes mostly overlap. An unsupervised VAE organizes its latent by low-level appearance more than by semantic class, so this is about what you would expect without label supervision.

## Class-conditional version

`class_cond_vae.py` feeds a learned class embedding into both the encoder and decoder. That lets you pick the class at generation time and sample within it instead of getting a random draw. Same analysis set in `class_data_analysis/`.
