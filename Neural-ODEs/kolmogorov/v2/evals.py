import os
import sys
import glob
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex

from model import (NODE, KolmogorovDataset, standardise_nu, unstandardise_nu, HERE, device)

#idea was mine, evaluation code writing itself was done by Claude

#how far ahead has the model actually learned to predict?
#the window averaged MSE cannot distinguish a model that holds out to t=1.45 from one
#that is sharp until t=0.5 and then blurs to the mean, so score every lag separately

#v2 adds the question the scalars above cannot answer at all: the model is *conditioned*
#on nu, but does it use it? the first run said no, flatly: 0.0% change at every checkpoint when
#nu was shuffled. amplitude used to scale like 1/nu under a single global sigma, so x0 leaked nu
#and the input was redundant. windows are now normalised individually, which removes the scale
#cue (though not the spectral one, since low nu still means finer structure). so we score every
#nu separately, then feed deliberately wrong nu to see whether anything changes

WINDOW = 30
STRIDE = 10
#must match the dt model.py trains with, not the true 15/149 spacing of the data, because the
#model was fitted against a t_grid built from this number and the rollout has to be scored on
#the same clock it learned on
DT = 0.10

BENCHMARKS = os.path.join(HERE, "benchmarks") #one subfolder per checkpoint

#lags to show fields and spectra for: one inside the skilful range, one near the
#ACC=0.6 crossover, one out where we expect the prediction to have gone smooth
SHOW_LAGS = [0, 5, 10, 20, 29]
SPECTRUM_LAGS = [5, 14, 29]

#nu values the model was trained on; anything else in the test set is unseen, and the
#figures mark the two cases differently so "seen" is never carried by colour alone.
#keep this in step with the train sweep in kolmogorov_flow.py or every group gets mislabelled
TRAIN_NUS = [0.03, 0.043, 0.06, 0.088, 0.12]
#a fixed initial condition rolled out under each of these, to read off the response to nu.
#spans the trained band and just past both ends, so the sweep shows extrapolation too. capped
#at 6 by nu_ramp, which is where a sequential ramp stops separating its steps
NU_SWEEP = [0.025, 0.04, 0.06, 0.09, 0.13]

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

def nu_ramp(n):
    #nu is an ordered variable, not an identity, so series keyed by nu take a sequential
    #ramp rather than categorical hues. the start of the ramp is clipped to 0.32 because
    #the light end of SEQUENTIAL is a fill colour and disappears as a 2px line on SURFACE.
    #both bounds are checked, not eyeballed: over 0.32 -> 1.0 the lightest step holds 2.41
    #WCAG against SURFACE (floor 2.0) and adjacent steps hold >= 0.072 OKLCH dL (floor
    #0.06) out to six steps, which is where the ramp stops separating and you facet instead
    if n > 6:
        raise ValueError(f"{n} steps: a sequential ramp stops separating past 6, facet instead")
    if n == 1:
        return [to_hex(SEQUENTIAL(1.0))]
    return [to_hex(SEQUENTIAL(0.32 + 0.68 * i / (n - 1))) for i in range(n)]

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

def get_checkpoints():
    paths = glob.glob(os.path.join(HERE, "model_epoch_*.pth"))
    if not paths:
        raise FileNotFoundError(f"no model_epoch_*.pth found in {HERE}")
        # return all checkpoints sorted by epoch number (ascending)
    def _epoch_key(p):
        return int(os.path.basename(p).split("_")[-1].split(".")[0])
    return sorted(paths, key=_epoch_key)

_EVAL_DATA = None

def eval_data():
    #the train tensor is ~1 GB and only supplies the climatology, so load it once and reuse it
    #across every checkpoint in the sweep
    global _EVAL_DATA
    if _EVAL_DATA is None:
        #the climatology has to come from the train set and, critically, has to live in the same
        #units the model predicts in. windows are now normalised individually by their own x0
        #std, so a climatology built from raw omega would be off by a factor of ~15 and every
        #skill number derived from it would be meaningless. so build it out of the training
        #dataset itself, which applies exactly the normalisation the model was fitted against
        train = torch.load(os.path.join(HERE, "kolmogorov_train_dataset.pt"))
        train_ds = KolmogorovDataset(train, WINDOW, STRIDE, DT)
        #one climatology pooled over every nu, deliberately. a per nu climatology would be a
        #stronger baseline, but two of the test nu never appear in training so it does not
        #exist for them, and a baseline that changes between groups makes the groups
        #incomparable, which is the whole point of splitting them
        total = torch.zeros(1, 64, 64)
        for i in range(len(train_ds)):
            total += train_ds[i][2].mean(dim=0) #mean over the lags of one normalised window
        climatology = total / len(train_ds) #(1, N, N)
        del train, train_ds

        test = torch.load(os.path.join(HERE, "kolmogorov_test_dataset.pt"))
        _EVAL_DATA = (KolmogorovDataset(test, WINDOW, STRIDE, DT), climatology)
    return _EVAL_DATA

def nu_groups(dataset):
    #window indices bucketed by nu, so each group can be scored through the same code path as
    #the whole set. dataset.windows stores the RAW physical nu (it is __getitem__ that
    #standardises), so these keys stay in physical units and compare directly against TRAIN_NUS
    groups = {}
    for idx, (nu, _, _) in enumerate(dataset.windows):
        groups.setdefault(round(float(nu), 4), []).append(idx)
    return dict(sorted(groups.items()))

def shuffle_within_batch(nu):
    #the ablation: keep x0 and the target, break the pairing to nu. shuffling (rather than
    #substituting a constant) keeps the marginal distribution of nu exactly as trained, so a
    #degradation cannot be blamed on handing the embedding an out of distribution number.
    #operates on already standardised values, so a permutation is all it is
    return nu[torch.randperm(nu.shape[0], device=nu.device)]

def as_model_nu(physical_nu, like):
    #physical nu -> the standardised scalar the network expects, shaped like a batch of them.
    #every place that injects a hand picked nu has to go through this, or the embedding gets
    #handed a raw 0.06 where it was trained to see 0.0
    value = standardise_nu(torch.tensor(float(physical_nu)))
    return torch.full_like(like, value.item())

def load_model(ckpt_path):
    #map_location matters: a checkpoint trained on mps carries mps tensors
    model = NODE(1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def evaluate(model, indices=None, nu_map=None, shuffle=False):
    #indices restricts the scoring to one nu group; nu_map rewrites the nu fed to the model
    #while leaving the target alone, which is how the ablation runs through this same path
    dataset, climatology = eval_data()
    source = dataset if indices is None else Subset(dataset, indices)
    #the shuffle ablation needs mixed nu inside a batch, otherwise permuting is a no op
    loader = DataLoader(source, batch_size=32, shuffle=shuffle)
    t_grid = dataset.t.to(device)

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
        for nu, x0, traj, _, _ in loader:
            nu = nu.to(device) #already standardised log nu, straight from the dataset
            if nu_map is not None:
                nu = nu_map(nu)
            #run the model on the accelerator, then bring the prediction back: every
            #statistic below (and the ffts in particular) is cheap and cpu-safe
            pred = model(x0.to(device), t_grid, nu).cpu()
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

def plot_curves(res, out_dir):
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
    #label the threshold below the line at the left: the curve starts near 1.0 and only
    #descends, so that corner is the one region it can never reach. above the line is
    #exactly where it travels, and to the right is where it ends up once skill is lost
    ax_acc.annotate("ACC = 0.6 usefulness floor", (t[0], 0.6),
                    textcoords="offset points", xytext=(4, -14), ha="left", va="top",
                    color=MUTED, fontsize=8)
    below = (res["acc"] < 0.6).nonzero()
    if len(below):
        lag = int(below[0])
        ax_acc.plot([lag * DT], [res["acc"][lag]], marker="o", markersize=8,
                    color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=2)
    ax_acc.set_title("Anomaly correlation of the Neural ODE", color=INK, fontsize=11,
                     loc="left", pad=10)
    ax_acc.set_xlabel("lead time t", color=INK_2, fontsize=9)
    ax_acc.set_ylabel("ACC", color=INK_2, fontsize=9)
    ax_acc.set_ylim(0, 1)
    style_axes(ax_acc)

    fig.tight_layout()
    path = os.path.join(out_dir, "curves.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path

def plot_fields(res, out_dir):
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

    fig.suptitle("Vorticity Rollout Against Ground Truth", color=INK, fontsize=12)
    path = os.path.join(out_dir, "fields.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_spectra(res, out_dir):
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
            #the spectrum peaks at k=4, so anchor the label to the bottom of the axes
            #instead: blended transform, data coords in x and axes fraction in y
            ax.annotate("k = 4, forcing scale", xy=(4, 0.02),
                        xycoords=ax.get_xaxis_transform(),
                        xytext=(6, 0), textcoords="offset points",
                        ha="left", va="bottom", color=MUTED, fontsize=8)
            ax.set_ylabel("enstrophy per shell", color=INK_2, fontsize=9)
            leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
            for text in leg.get_texts():
                text.set_color(INK_2)
        ax.set_title(f"t = {lag * DT:.2f}", color=INK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("wavenumber k", color=INK_2, fontsize=9)
        style_axes(ax)

    fig.suptitle("Enstrophy Spectrum: Is the Prediction Losing its Small Scales?",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out_dir, "spectra.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path

def plot_nu_skill(res_by_nu, out_dir):
    #one line per nu, so a conditioning failure shows up as the curves collapsing onto each
    #other and an extrapolation failure shows up as the unseen nu peeling away from the rest
    nus = list(res_by_nu)
    colors = nu_ramp(len(nus))
    fig, (ax_mse, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    for nu, color in zip(nus, colors):
        res = res_by_nu[nu]
        seen = any(abs(nu - tn) < 1e-6 for tn in TRAIN_NUS)
        #dash the unseen nu: seen/unseen is the comparison the figure exists to make, so it
        #cannot ride on colour, which is already spent encoding the value of nu itself
        style = "-" if seen else (0, (5, 2))
        #raw MSE is not comparable across nu here: amplitude scales like 1/nu, so the lowest
        #nu is several times the highest and a shared linear axis would flatten every other
        #group against the floor. dividing by that group's own climatology error indexes all
        #four to a common base, and 1.0 lands on a meaning: no better than the mean field
        skill = res["mse_model"] / res["mse_clim"]
        for ax, values in ((ax_mse, skill), (ax_acc, res["acc"])):
            ax.plot(res["t"], values, color=color, linewidth=2, linestyle=style,
                    solid_capstyle="round",
                    label=f"nu = {nu:g}" + ("" if seen else " (unseen)"))

    ax_mse.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax_mse.annotate("no better than climatology", (0, 1.0), textcoords="offset points",
                    xytext=(4, -14), ha="left", va="top", color=MUTED, fontsize=8)
    ax_mse.set_title("Skill against climatology, per viscosity", color=INK, fontsize=11,
                     loc="left", pad=10)
    ax_mse.set_ylabel("MSE / climatology MSE", color=INK_2, fontsize=9)
    ax_mse.set_ylim(0, None)

    ax_acc.axhline(0.6, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax_acc.annotate("ACC = 0.6 usefulness floor", (0, 0.6), textcoords="offset points",
                    xytext=(4, -14), ha="left", va="top", color=MUTED, fontsize=8)
    ax_acc.set_title("Anomaly correlation, per viscosity", color=INK, fontsize=11,
                     loc="left", pad=10)
    ax_acc.set_ylabel("ACC", color=INK_2, fontsize=9)
    ax_acc.set_ylim(0, 1)

    for ax in (ax_mse, ax_acc):
        ax.set_xlabel("lead time t", color=INK_2, fontsize=9)
        style_axes(ax)
    #one legend for both panels, below them: the curves fill the plot area at one end or the
    #other depending on how well the checkpoint does, so any in-axes corner eventually collides
    handles, labels = ax_mse.get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(nus),
                     loc="lower center", bbox_to_anchor=(0.5, 0))
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(out_dir, "nu_skill.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path

def plot_ablation(res_by_nu, res_wrong_by_nu, out_dir):
    #the load bearing figure for the whole v2 claim: how much worse does the rollout get when
    #the model is handed the wrong nu? near zero means the nu input is decorative and the
    #encoder is reading nu off the amplitude of x0 instead
    #
    #plotting the two absolute MSEs side by side is the obvious move and it is the wrong one:
    #amplitude scales like 1/nu, so the lowest nu group is several times the tallest and the
    #within-pair gap, which is the entire point, shrinks to nothing for every other group.
    #the ratio is the quantity of interest, so encode the ratio and let the table carry the
    #absolute numbers
    nus = list(res_by_nu)
    x = torch.arange(len(nus)).float()
    deltas = [(res_wrong_by_nu[nu]["mse_model"].mean() / res_by_nu[nu]["mse_model"].mean()
               - 1.0).item() * 100 for nu in nus]
    fig, ax = plt.subplots(figsize=(1.5 * len(nus) + 3.4, 4.2), facecolor=SURFACE)

    #one series, so the title names it and no legend box is needed. thin bars, generous gaps
    ax.bar(x, deltas, 0.34, color=SERIES[0], zorder=3)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=4)
    for xi, d in zip(x.tolist(), deltas):
        #label above a rising bar and below a falling one, so the text never sits on the mark
        ax.annotate(f"{d:+.1f}%", (xi, d), textcoords="offset points",
                    xytext=(0, 5 if d >= 0 else -13), ha="center", color=INK_2, fontsize=9)

    #the interpretation threshold, drawn once rather than explained in a caption
    ax.axhline(15, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("15%: below here the conditioning is doing little", (len(nus) - 0.5, 15),
                textcoords="offset points", xytext=(-2, 4), ha="right", va="bottom",
                color=MUTED, fontsize=8)

    ax.set_xticks(x.tolist())
    ax.set_xticklabels([f"nu = {nu:g}" + ("" if any(abs(nu - tn) < 1e-6 for tn in TRAIN_NUS)
                                          else "\n(unseen)") for nu in nus])
    ax.set_xlim(-0.6, len(nus) - 0.4)
    ax.set_title("Does the model actually use its nu input?", color=INK, fontsize=11,
                 loc="left", pad=10)
    ax.set_ylabel("MSE increase when fed the wrong nu (%)", color=INK_2, fontsize=9)
    #headroom for the labels, and keep 0 and the 15% line in frame however small the effect is
    top = max(20.0, max(deltas) * 1.25) if deltas else 20.0
    ax.set_ylim(min(0.0, min(deltas) * 1.35) - 2, top)
    style_axes(ax)
    ax.grid(axis="x", visible=False) #bars sit on categories, so a vertical grid says nothing

    fig.tight_layout()
    path = os.path.join(out_dir, "ablation.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path

def plot_nu_response(model, out_dir):
    #the positive version of the ablation: hold x0 fixed, sweep nu, and check the rollouts
    #move the way physics says they should. nu multiplies the dissipation term, so raising it
    #has to pull enstrophy down. this is a claim about direction, which no MSE can make
    dataset, _ = eval_data()
    _, x0, traj, _, _ = dataset[0]
    true_nu = round(float(dataset.windows[0][0]), 4) #windows hold physical nu
    t_grid = dataset.t.to(device)
    colors = nu_ramp(len(NU_SWEEP))

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    x0_batch = x0.unsqueeze(0).to(device)

    #x0 is normalised to unit std, so every swept rollout starts from identical enstrophy by
    #construction. any spread that appears downstream is the nu conditioning and nothing else
    finals = []
    with torch.no_grad():
        for nu, color in zip(NU_SWEEP, colors):
            nu_tensor = as_model_nu(nu, torch.zeros(1, device=device))
            pred = model(x0_batch, t_grid, nu_tensor).cpu()[0]
            enstrophy = 0.5 * (pred ** 2).mean(dim=(1, 2, 3)) #per lag, normalised units
            finals.append(enstrophy.mean().item()) #window mean, for the ordering check below
            ax.plot(dataset.t, enstrophy, color=color, linewidth=2, solid_capstyle="round",
                    label=f"nu = {nu:g}")

    #the truth for this window, as the anchor the swept curves should bracket
    true_enstrophy = 0.5 * (traj ** 2).mean(dim=(1, 2, 3))
    ax.plot(dataset.t, true_enstrophy, color=INK_2, linewidth=1.4, linestyle=(0, (4, 3)),
            label=f"truth (nu = {true_nu:g})")

    ax.set_title("Response to nu from one fixed initial condition", color=INK, fontsize=11,
                 loc="left", pad=10)
    ax.set_xlabel("lead time t", color=INK_2, fontsize=9)
    ax.set_ylabel("enstrophy (normalised units)", color=INK_2, fontsize=9)
    ax.set_ylim(0, None)
    style_axes(ax)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.tight_layout()
    path = os.path.join(out_dir, "nu_response.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    #hand back the ordering check so the summary can state it rather than leaving it to the eye
    monotone = all(finals[i] >= finals[i + 1] for i in range(len(finals) - 1))
    return path, finals, monotone

def report(ckpt_path):
    dataset, _ = eval_data()
    model = load_model(ckpt_path)
    res = evaluate(model)
    t = res["t"]
    mse_model, mse_pers, mse_clim, acc = (res["mse_model"], res["mse_pers"],
                                          res["mse_clim"], res["acc"])

    #benchmarks/epoch_40/, named off the checkpoint so a sweep never overwrites itself
    tag = os.path.basename(ckpt_path).replace("model_", "").replace(".pth", "")
    out_dir = os.path.join(BENCHMARKS, tag)
    os.makedirs(out_dir, exist_ok=True)

    #build the summary once, then both print it and keep it beside the figures
    lines = [f"checkpoint: {os.path.basename(ckpt_path)}",
             f"eval windows: {res['windows']}, {len(t)} lags, dt = {DT}",
             "",
             " lag     t     model    persist    clim     skill      ACC"]
    for i in range(len(t)):
        skill = mse_model[i] / mse_clim[i] #below 1 means better than predicting the mean field
        lines.append(f" {i:3d}  {t[i]:5.2f}   {mse_model[i]:7.4f}  {mse_pers[i]:7.4f}  "
                     f"{mse_clim[i]:7.4f}  {skill:7.3f}  {acc[i]:7.3f}")

    lines += ["", f"window mean MSE: model {mse_model.mean():.4f}, "
                  f"persistence {mse_pers.mean():.4f}, climatology {mse_clim.mean():.4f}"]

    #the two horizons worth quoting: where the model stops beating the mean field, and
    #where the anomaly correlation drops through 0.6 (the usual usefulness threshold)
    lost = (mse_model >= mse_clim).nonzero()
    if len(lost):
        lines.append(f"beats climatology out to lag {int(lost[0]) - 1} (t = {(int(lost[0]) - 1) * DT:.2f})")
    else:
        lines.append(f"beats climatology across the whole window (through t = {t[-1]:.2f})")

    below = (acc < 0.6).nonzero()
    if len(below):
        lines.append(f"ACC drops below 0.6 at lag {int(below[0])} (t = {int(below[0]) * DT:.2f})")
    else:
        lines.append(f"ACC stays above 0.6 across the whole window (through t = {t[-1]:.2f})")

    #--- the v2 questions: score each nu on its own, then break the pairing to nu ---------
    groups = nu_groups(dataset)
    res_by_nu = {nu: evaluate(model, indices=idx) for nu, idx in groups.items()}
    #"mismatched" means the farthest other test nu, which is the strongest perturbation the
    #test set can offer, so a model that does use nu has nowhere to hide
    wrong_nu = {nu: max(groups, key=lambda other: abs(other - nu)) for nu in groups}
    res_wrong_by_nu = {
        nu: evaluate(model, indices=idx,
                     nu_map=lambda n, w=wrong_nu[nu]: as_model_nu(w, n))
        for nu, idx in groups.items()
    }
    #and the single headline number: nu shuffled across the whole test set at once
    torch.manual_seed(0) #the permutation is random, so pin it or the number moves per run
    res_shuffled = evaluate(model, nu_map=shuffle_within_batch, shuffle=True)

    lines += ["", "per viscosity (window mean over all lags)",
              "'fed nu' is the wrong value substituted for the ablation: the farthest other "
              "test nu,", "the strongest perturbation this test set can offer", "",
              "      nu    seen    windows    model     clim    skill      ACC    fed nu   "
              "  MSE     delta"]
    for nu, r in res_by_nu.items():
        seen = "yes" if any(abs(nu - tn) < 1e-6 for tn in TRAIN_NUS) else "no"
        wrong = res_wrong_by_nu[nu]["mse_model"].mean()
        delta = wrong / r["mse_model"].mean() - 1.0
        #4 decimals, not 2: the sweeps carry values like 0.0375 and 0.043, and rounding them
        #to 2 places collides distinct groups onto the same printed label
        lines.append(f" {nu:7.4f}  {seen:>5}  {r['windows']:9d}  {r['mse_model'].mean():7.4f}  "
                     f"{r['mse_clim'].mean():7.4f}  {r['mse_model'].mean() / r['mse_clim'].mean():7.3f}  "
                     f"{r['acc'].mean():7.3f}  {wrong_nu[nu]:8.4f}  {wrong:7.4f}  {delta:+6.1%}")
    #lag 0 is decoder(encoder(x0)) with no integration, so it cannot depend on nu at all;
    #if the per lag columns differ there, something is wrong with how nu is being threaded
    lines.append("(lag 0 is identical by construction: no integration, so nu cannot enter)")

    #the ablation verdict, stated rather than left to the reader to infer from the table
    shuffled_mse = res_shuffled["mse_model"].mean()
    true_mse = mse_model.mean()
    penalty = shuffled_mse / true_mse - 1.0
    lines += ["", f"nu shuffled across the test set: MSE {shuffled_mse:.4f} vs {true_mse:.4f} "
                  f"with true nu ({penalty:+.1%})"]
    if penalty < 0.02:
        lines.append("  -> the nu input is doing essentially nothing; the encoder is reading nu "
                     "off the amplitude of x0. the conditioning claim is NOT supported")
    elif penalty < 0.15:
        lines.append("  -> the model leans on nu only weakly; most of the signal is still coming "
                     "from x0 itself")
    else:
        lines.append("  -> the model genuinely depends on its nu input")

    nu_response_path, finals, monotone = plot_nu_response(model, out_dir)
    lines += ["", "response to nu from one fixed x0 (window mean enstrophy)"]
    for nu, value in zip(NU_SWEEP, finals):
        lines.append(f"  nu = {nu:6.4f}   enstrophy {value:.4f}")
    lines.append("  -> enstrophy falls monotonically with nu, as the physics requires"
                 if monotone else
                 "  -> NOT monotonic in nu: the learned response has the wrong shape, since nu "
                 "multiplies dissipation and can only damp")

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")

    #the same per lag numbers in a form you can plot across checkpoints later
    with open(os.path.join(out_dir, "metrics.csv"), "w") as fh:
        fh.write("lag,t,mse_model,mse_persistence,mse_climatology,skill,acc\n")
        for i in range(len(t)):
            fh.write(f"{i},{t[i]:.4f},{mse_model[i]:.6f},{mse_pers[i]:.6f},"
                     f"{mse_clim[i]:.6f},{mse_model[i] / mse_clim[i]:.6f},{acc[i]:.6f}\n")

    #per nu and per lag, so the conditioning behaviour is plottable across checkpoints too
    with open(os.path.join(out_dir, "metrics_by_nu.csv"), "w") as fh:
        fh.write("nu,seen,lag,t,mse_model,mse_climatology,acc,mse_mismatched_nu\n")
        for nu, r in res_by_nu.items():
            seen = int(any(abs(nu - tn) < 1e-6 for tn in TRAIN_NUS))
            wrong = res_wrong_by_nu[nu]["mse_model"]
            for i in range(len(t)):
                fh.write(f"{nu:.4f},{seen},{i},{t[i]:.4f},{r['mse_model'][i]:.6f},"
                         f"{r['mse_clim'][i]:.6f},{r['acc'][i]:.6f},{wrong[i]:.6f}\n")

    print()
    for path in (plot_curves(res, out_dir), plot_fields(res, out_dir), plot_spectra(res, out_dir),
                 plot_nu_skill(res_by_nu, out_dir),
                 plot_ablation(res_by_nu, res_wrong_by_nu, out_dir),
                 nu_response_path):
        print(f"wrote {os.path.relpath(path, HERE)}")
    print(f"wrote {os.path.relpath(out_dir, HERE)}/summary.txt, metrics.csv "
          f"and metrics_by_nu.csv\n")


for checkpoint in get_checkpoints():
    report(checkpoint)


