import os
import sys
import glob
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model import NODE, KolmogorovDataset, HERE

#idea was mine, evaluation code writing itself was done by Claude

#how far ahead has the model actually learned to predict?
#the window averaged MSE cannot distinguish a model that holds out to t=1.45 from one
#that is sharp until t=0.5 and then blurs to the mean, so score every lag separately

WINDOW = 30
STRIDE = 10
DT = 0.05

#lags to show fields and spectra for: one inside the skilful range, one near the
#ACC=0.6 crossover, one out where we expect the prediction to have gone smooth
SHOW_LAGS = [0, 5, 10, 20, 29]
SPECTRUM_LAGS = [5, 14, 29]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#categorical slots 1-3; these three validate all-pairs in both modes
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
#signed vorticity is a polarity encoding: two opposite hues, neutral gray midpoint
DIVERGING = LinearSegmentedColormap.from_list("vorticity", ["#2a78d6", "#f0efec", "#e34948"])
#error magnitude is a sequential encoding: one hue, light -> dark
SEQUENTIAL = LinearSegmentedColormap.from_list("error", ["#cde2fb", "#3987e5", "#0d366b"])

def style_axes(ax):
    #hairline solid grid and axes, one shade off the surface, so the marks carry the chart
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)

def radial_spectrum(field):
    #shell averaged enstrophy spectrum; the honest test for blurring, since a
    #prediction that has gone smooth loses power at high k while keeping low k intact
    n = field.shape[-1]
    power = (torch.fft.fft2(field).abs() ** 2) / n ** 4
    freq = torch.fft.fftfreq(n, d=1.0 / n)
    kx, ky = torch.meshgrid(freq, freq, indexing="ij")
    shell = torch.sqrt(kx ** 2 + ky ** 2).round().long().clamp(max=n // 2)
    out = torch.zeros(*field.shape[:-2], n // 2 + 1)
    out.index_add_(-1, shell.flatten(), power.flatten(start_dim=-2))
    return out

def latest_checkpoint():
    paths = glob.glob(os.path.join(HERE, "model_epoch_*.pth"))
    if not paths:
        raise FileNotFoundError(f"no model_epoch_*.pth found in {HERE}")
    return max(paths, key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))

def evaluate(ckpt_path):
    #sigma and the climatology both have to come from the train set, same as in training
    train = torch.load(os.path.join(HERE, "kolmogorov_train_dataset.pt")).unsqueeze(2)
    sigma = train.std()
    climatology = (train / sigma).mean(dim=(0, 1)) #per pixel time mean, shape (1, N, N)
    del train

    test = torch.load(os.path.join(HERE, "kolmogorov_test_dataset.pt")).unsqueeze(2)
    dataset = KolmogorovDataset(test, WINDOW, STRIDE, DT, sigma)
    loader = DataLoader(dataset, batch_size=32)

    model = NODE(1)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    #accumulate sums rather than per batch means so uneven batch sizes stay correctly weighted
    sq_model = torch.zeros(WINDOW)
    sq_pers = torch.zeros(WINDOW)
    sq_clim = torch.zeros(WINDOW)
    #anomaly correlation, accumulated over every window and pixel at each lag
    acc_xy = torch.zeros(WINDOW)
    acc_xx = torch.zeros(WINDOW)
    acc_yy = torch.zeros(WINDOW)
    count = 0
    windows = 0
    spec_true = None
    spec_pred = None
    sample = None #one window kept whole, for the field montage

    with torch.no_grad():
        for x0, traj, _ in loader:
            pred = model(x0, dataset.t)
            pers = traj[:, :1].expand_as(traj) #hold omega_0 for the whole window
            count += traj.shape[0] * traj[0, 0].numel()
            windows += traj.shape[0]

            sq_model += ((pred - traj) ** 2).sum(dim=(0, 2, 3, 4))
            sq_pers += ((pers - traj) ** 2).sum(dim=(0, 2, 3, 4))
            sq_clim += ((climatology - traj) ** 2).sum(dim=(0, 2, 3, 4))

            pred_anom = pred - climatology
            true_anom = traj - climatology
            acc_xy += (pred_anom * true_anom).sum(dim=(0, 2, 3, 4))
            acc_xx += (pred_anom ** 2).sum(dim=(0, 2, 3, 4))
            acc_yy += (true_anom ** 2).sum(dim=(0, 2, 3, 4))

            st = radial_spectrum(traj[:, :, 0]).sum(dim=0)
            sp = radial_spectrum(pred[:, :, 0]).sum(dim=0)
            spec_true = st if spec_true is None else spec_true + st
            spec_pred = sp if spec_pred is None else spec_pred + sp
            if sample is None:
                sample = (traj[0, :, 0].clone(), pred[0, :, 0].clone())

    mse_model = sq_model / count
    mse_pers = sq_pers / count
    mse_clim = sq_clim / count
    acc = acc_xy / torch.sqrt(acc_xx * acc_yy)
    return {
        "t": dataset.t,
        "mse_model": mse_model,
        "mse_pers": mse_pers,
        "mse_clim": mse_clim,
        "acc": acc,
        "spec_true": spec_true / windows,
        "spec_pred": spec_pred / windows,
        "sample": sample,
        "windows": windows,
    }

def plot_curves(res, tag):
    #left: error against both trivial baselines. right: how far ahead the skill survives
    fig, (ax_mse, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    t = res["t"]

    for values, name, color in [(res["mse_model"], "Neural ODE", SERIES[0]),
                                (res["mse_pers"], "Persistence", SERIES[1]),
                                (res["mse_clim"], "Climatology", SERIES[2])]:
        ax_mse.plot(t, values, color=color, linewidth=2, label=name, solid_capstyle="round")
    #label only the series that matters, at its endpoint, and let the legend carry the rest
    ax_mse.annotate(f"{res['mse_model'][-1]:.3f}", (t[-1], res["mse_model"][-1]),
                    textcoords="offset points", xytext=(6, -3), color=INK_2, fontsize=8)
    ax_mse.set_title("Error by lead time", color=INK, fontsize=11, loc="left", pad=10)
    ax_mse.set_xlabel("lead time t", color=INK_2, fontsize=9)
    ax_mse.set_ylabel("MSE (normalised units)", color=INK_2, fontsize=9)
    ax_mse.set_ylim(0, None)
    leg = ax_mse.legend(frameon=False, fontsize=9, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK_2)
    style_axes(ax_mse)

    #single series, so the title names it and no legend box is needed
    ax_acc.plot(t, res["acc"], color=SERIES[0], linewidth=2, solid_capstyle="round")
    ax_acc.axhline(0.6, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax_acc.annotate("ACC = 0.6, the usual usefulness floor", (t[-1], 0.6),
                    textcoords="offset points", xytext=(-4, 6), ha="right",
                    color=MUTED, fontsize=8)
    below = (res["acc"] < 0.6).nonzero()
    if len(below):
        lag = int(below[0])
        ax_acc.plot([lag * DT], [res["acc"][lag]], marker="o", markersize=8,
                    color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=2)
        ax_acc.annotate(f"horizon t = {lag * DT:.2f}", (lag * DT, res["acc"][lag]),
                        textcoords="offset points", xytext=(8, 8), color=INK_2, fontsize=8)
    ax_acc.set_title("Anomaly correlation of the Neural ODE", color=INK, fontsize=11,
                     loc="left", pad=10)
    ax_acc.set_xlabel("lead time t", color=INK_2, fontsize=9)
    ax_acc.set_ylabel("ACC", color=INK_2, fontsize=9)
    ax_acc.set_ylim(0, 1)
    style_axes(ax_acc)

    fig.tight_layout()
    path = os.path.join(HERE, f"eval_curves_{tag}.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path

def plot_fields(res, tag):
    #the qualitative check the scalars cannot give you: does the rollout still look
    #like turbulence at long lead times, or has it gone smooth?
    truth, pred = res["sample"]
    lags = [l for l in SHOW_LAGS if l < truth.shape[0]]
    scale = truth.abs().max().item() #one shared symmetric scale, or blurring hides itself
    err_max = (pred - truth).abs().max().item()

    fig, axes = plt.subplots(3, len(lags), figsize=(2.1 * len(lags) + 1.4, 6.6),
                             facecolor=SURFACE)
    rows = [("Truth", truth, DIVERGING, -scale, scale),
            ("Neural ODE", pred, DIVERGING, -scale, scale),
            ("|error|", (pred - truth).abs(), SEQUENTIAL, 0, err_max)]

    images = []
    for r, (name, data, cmap, vmin, vmax) in enumerate(rows):
        for c, lag in enumerate(lags):
            ax = axes[r, c]
            im = ax.imshow(data[lag], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(AXIS)
                side.set_linewidth(0.8)
            if r == 0:
                ax.set_title(f"t = {lag * DT:.2f}", color=INK, fontsize=10, pad=8)
            if c == 0:
                ax.set_ylabel(name, color=INK, fontsize=10, labelpad=10)
        images.append(im)

    #truth and prediction share one scale, so they share one bar; the error row needs its own
    for image, group in [(images[0], axes[0:2, :]), (images[2], axes[2:3, :])]:
        bar = fig.colorbar(image, ax=group.ravel().tolist(), fraction=0.02, pad=0.015)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=MUTED, labelsize=7, length=2, width=0.8)

    fig.suptitle("Vorticity rollout against ground truth", color=INK, fontsize=12,
                 x=0.02, ha="left")
    path = os.path.join(HERE, f"eval_fields_{tag}.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_spectra(res, tag):
    #small multiples, two series each: if the prediction is blurring, its spectrum
    #falls away from the truth at high k while low k stays put
    lags = [l for l in SPECTRUM_LAGS if l < res["spec_true"].shape[0]]
    fig, axes = plt.subplots(1, len(lags), figsize=(3.7 * len(lags), 3.8),
                             facecolor=SURFACE, sharey=True)
    #the generator dealiases with a 2/3 mask, so the top shells are numerically empty;
    #cut the axis at the last shell that carries real power or the plot is all cliff
    reference = res["spec_true"][lags[0]]
    cutoff = int((reference > reference.max() * 1e-6).nonzero().max())
    k = torch.arange(cutoff + 1)[1:]
    floor = min(res["spec_true"][lags[-1]][1:cutoff + 1].min(),
                res["spec_pred"][lags[-1]][1:cutoff + 1].min())

    for i, lag in enumerate(lags):
        ax = axes[i]
        ax.loglog(k, res["spec_true"][lag][1:cutoff + 1], color=SERIES[0], linewidth=2,
                  label="Truth", solid_capstyle="round")
        ax.loglog(k, res["spec_pred"][lag][1:cutoff + 1], color=SERIES[1], linewidth=2,
                  label="Neural ODE", solid_capstyle="round")
        ax.axvline(4, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_ylim(floor * 0.5, None)
        if i == 0:
            ax.annotate("k = 4, forcing scale", (4, ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(6, -12), color=MUTED, fontsize=8)
            ax.set_ylabel("enstrophy per shell", color=INK_2, fontsize=9)
            leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
            for text in leg.get_texts():
                text.set_color(INK_2)
        ax.set_title(f"t = {lag * DT:.2f}", color=INK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("wavenumber k", color=INK_2, fontsize=9)
        style_axes(ax)

    fig.suptitle("Enstrophy spectrum: is the prediction losing its small scales?",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(HERE, f"eval_spectra_{tag}.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path

def report(ckpt_path):
    res = evaluate(ckpt_path)
    t = res["t"]
    mse_model, mse_pers, mse_clim, acc = (res["mse_model"], res["mse_pers"],
                                          res["mse_clim"], res["acc"])

    print(f"checkpoint: {os.path.basename(ckpt_path)}")
    print(f"eval windows: {res['windows']}, {len(t)} lags, dt = {DT}\n")
    print(" lag     t     model    persist    clim     skill      ACC")
    for i in range(len(t)):
        skill = mse_model[i] / mse_clim[i] #below 1 means better than predicting the mean field
        print(f" {i:3d}  {t[i]:5.2f}   {mse_model[i]:7.4f}  {mse_pers[i]:7.4f}  "
              f"{mse_clim[i]:7.4f}  {skill:7.3f}  {acc[i]:7.3f}")

    print(f"\nwindow mean MSE: model {mse_model.mean():.4f}, "
          f"persistence {mse_pers.mean():.4f}, climatology {mse_clim.mean():.4f}")

    #the two horizons worth quoting: where the model stops beating the mean field, and
    #where the anomaly correlation drops through 0.6 (the usual usefulness threshold)
    lost = (mse_model >= mse_clim).nonzero()
    if len(lost):
        print(f"beats climatology out to lag {int(lost[0]) - 1} (t = {(int(lost[0]) - 1) * DT:.2f})")
    else:
        print(f"beats climatology across the whole window (through t = {t[-1]:.2f})")

    below = (acc < 0.6).nonzero()
    if len(below):
        print(f"ACC drops below 0.6 at lag {int(below[0])} (t = {int(below[0]) * DT:.2f})")
    else:
        print(f"ACC stays above 0.6 across the whole window (through t = {t[-1]:.2f})")

    tag = os.path.basename(ckpt_path).replace("model_", "").replace(".pth", "")
    print()
    for path in (plot_curves(res, tag), plot_fields(res, tag), plot_spectra(res, tag)):
        print(f"wrote {os.path.basename(path)}")

report(latest_checkpoint())


