import os
import math
import importlib.util
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, ROOT)
from data.dataset import MRI_CT_DATASET

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Diffusion samples iteratively, so scoring the full 6778-slice test set is slow.
NUM_EVAL_BATCHES = 25
BATCH_SIZE = 8
DDIM_STEPS = 300 # diffusion sampler steps 
DIFFUSION_T = 1000
NUM_VIS = 6


def load_module(rel_path, mod_name):
    """Import a model file as a standalone module (runs only top-level defs)."""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ckpt(model, rel_path):
    """Load a state_dict, stripping the torch.compile `_orig_mod.` prefix."""
    sd = torch.load(os.path.join(ROOT, rel_path), map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


# ---- metrics (inputs are images mapped to [0, 1]) ----------------------------
def psnr_metric(pred, target):
    mse = F.mse_loss(pred, target).item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


try:
    from kornia.metrics import ssim as _kornia_ssim

    def ssim_metric(pred, target):
        return _kornia_ssim(pred, target, window_size=11).mean().item()
except Exception:  # older kornia: derive from the loss used during training
    from kornia.losses import ssim_loss

    def ssim_metric(pred, target):
        return 1.0 - ssim_loss(pred, target, window_size=11).item()


# ---- build models + their MRI->CT predict fns --------------------------------
unet_mod = load_module("U-NET/u-net.py", "unet_mod")
pix2pix_mod = load_module("pix2pix/pix2pix.py", "pix2pix_mod")
cyclegan_mod = load_module("cycleGAN/cycleGAN.py", "cyclegan_mod")
transunet_mod = load_module("Trans-U-NET/trans-unet.py", "transunet_mod")
diffusion_mod = load_module("diffusion/diffusion.py", "diffusion_mod")

unet = load_ckpt(unet_mod.u_net(), "U-NET/unet_epoch_19.pth")
pix2pix = load_ckpt(pix2pix_mod.u_net(), "pix2pix/unet_epoch_11.pth")
cyclegan = load_ckpt(cyclegan_mod.u_net(), "cycleGAN/unet_epoch_19.pth")
transunet = load_ckpt(transunet_mod.trans_UNET(), "Trans-U-NET/trans_unet_epoch_18.pth")
diffusion = load_ckpt(diffusion_mod.u_net(), "diffusion/diffusion_epoch_140.pth")

# precompute the diffusion noise schedule once
alpha_bars = diffusion.cosine_scheduler(torch.arange(DIFFUSION_T).to(device))


def diffusion_predict(mri):
    return diffusion_mod.ddpm_sample(diffusion, mri, alpha_bars)


# latent-spaces: encode MRI -> convert latent -> decode CT. Needs all THREE
# checkpoints, which only exist after re-running disjoint-latent-spaces.py with
# VAE saving. Guarded so the rest of the benchmark still runs if they're absent.
latent_predict = None
try:
    ls_mod = load_module("latent-spaces/disjoint-latent-spaces.py", "ls_mod")
    ls_mri_enc = load_ckpt(ls_mod.VAE(256), "latent-spaces/mri_encoder_epoch_19.pth")
    ls_ct_enc = load_ckpt(ls_mod.VAE(256), "latent-spaces/ct_encoder_epoch_19.pth")
    ls_conv = load_ckpt(ls_mod.MLP(256), "latent-spaces/disjoint_converter_epoch_19.pth")

    def latent_predict(mri):
        z = ls_mri_enc.encode(mri)[0]          # sampled MRI latent (matches training)
        return ls_ct_enc.decode(ls_conv(z))
except FileNotFoundError:
    print("skipping latent-spaces: checkpoints not found "
          "(re-run disjoint-latent-spaces.py to create them)")


configs = [
    ("U-Net (supervised)",       lambda mri: unet(mri)),
    ("pix2pix (WGAN-GP)",        lambda mri: pix2pix(mri)),
    ("cycleGAN (unsupervised)",  lambda mri: cyclegan(mri)),
    ("Trans-U-Net",              lambda mri: transunet(mri)),
    ("diffusion (DDPM)",         diffusion_predict),
]
if latent_predict is not None:
    configs.append(("latent-spaces (VAE+MLP)", latent_predict))

# ---- score every model on the same batches -----------------------------------
test_loader = DataLoader(MRI_CT_DATASET(f"{ROOT}/data/test/mri", f"{ROOT}/data/test/ct"),
                         batch_size=BATCH_SIZE, shuffle=False)

results = []
for name, predict in configs:
    print(f"evaluating {name} ...", flush=True)
    mae = psnr = ssim = 0.0
    n = 0
    with torch.no_grad():
        for i, (mri, ct) in enumerate(test_loader):
            if i >= NUM_EVAL_BATCHES:
                break
            mri, ct = mri.to(device), ct.to(device)
            pred = predict(mri)
            pred01 = (pred.clamp(-1, 1) + 1) / 2      # [-1,1] -> [0,1]
            ct01 = (ct.clamp(-1, 1) + 1) / 2
            mae += F.l1_loss(pred01, ct01).item()
            psnr += psnr_metric(pred01, ct01)
            ssim += ssim_metric(pred01, ct01)
            n += 1
    results.append((name, mae / n, psnr / n, ssim / n))

# ---- report ------------------------------------------------------------------
imgs = NUM_EVAL_BATCHES * BATCH_SIZE
print(f"\nMRI -> CT comparison on {imgs} test slices (metrics on [0,1] images)")
print(f"{'model':<26}{'MAE↓':>10}{'PSNR(dB)↑':>12}{'SSIM↑':>10}")
print("-" * 58)
for name, mae, psnr, ssim in results:
    print(f"{name:<26}{mae:>10.4f}{psnr:>12.2f}{ssim:>10.4f}")


# ---- qualitative image grid: MRI -> real CT -> each model's prediction --------
sample_mri, sample_ct = next(iter(test_loader))          # first (fixed) batch
sample_mri = sample_mri[:NUM_VIS].to(device)
sample_ct = sample_ct[:NUM_VIS].to(device)

grid_rows = [("MRI input", sample_mri), ("Real CT", sample_ct)]
with torch.no_grad():
    for name, predict in configs:
        grid_rows.append((name, predict(sample_mri).clamp(-1, 1)))

fig, axes = plt.subplots(len(grid_rows), NUM_VIS,
                         figsize=(1.8 * NUM_VIS, 1.8 * len(grid_rows)))
for r, (label, imgs) in enumerate(grid_rows):
    for c in range(NUM_VIS):
        ax = axes[r, c]
        ax.imshow(imgs[c, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(label, fontsize=10)
fig.suptitle("MRI → CT  —  qualitative comparison", fontsize=14)
fig.tight_layout()
grid_path = os.path.join(ROOT, "comparison_grid.png")
fig.savefig(grid_path, dpi=150, bbox_inches="tight")
print(f"\nsaved {grid_path}")

# ---- metric bars: one small-multiple per metric (different scales, one axis) --
# Color follows the model, not its rank; the per-metric winner is emphasized.
BASE, BEST = "#4C72B0", "#DD8452"      # CVD-safe blue / orange
names = [r[0] for r in results]
ypos = list(range(len(names)))
panels = [
    ("MAE ↓",     [r[1] for r in results], min, "{:.3f}"),
    ("PSNR dB ↑", [r[2] for r in results], max, "{:.1f}"),
    ("SSIM ↑",    [r[3] for r in results], max, "{:.3f}"),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
for ax, (title, vals, best_fn, fmt) in zip(axes, panels):
    best = vals.index(best_fn(vals))
    colors = [BEST if i == best else BASE for i in range(len(vals))]
    ax.barh(ypos, vals, color=colors, height=0.62)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()                                   # first model on top
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, max(vals) * 1.18)                    # headroom for labels
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, v in zip(ypos, vals):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)
fig.suptitle("MRI → CT  —  metrics (per-metric best highlighted)", fontsize=13)
fig.tight_layout()
metrics_path = os.path.join(ROOT, "comparison_metrics.png")
fig.savefig(metrics_path, dpi=150, bbox_inches="tight")
print(f"saved {metrics_path}")

# MRI -> CT comparison on 200 test slices (metrics on [0,1] images)
# model                           MAE↓   PSNR(dB)↑     SSIM↑
# ----------------------------------------------------------
# U-Net (supervised)            0.0127       27.83    0.9286
# pix2pix (WGAN-GP)             0.0138       27.01    0.9188
# cycleGAN (unsupervised)       0.0335       21.46    0.7861
# Trans-U-Net                   0.0126       27.74    0.9292
# diffusion (DDIM)              0.1589       15.33    0.5586
# diffusion (DDPM)              0.2319       11.31    0.7453