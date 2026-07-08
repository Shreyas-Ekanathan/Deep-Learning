import numpy as np
import matplotlib.pyplot as plt

#claude generated plotting script

# categorical identity colors: ground truth vs. predicted (not one per atom —
# with 9 atoms, small multiples carry atom identity instead of color)
COLOR_TRUE = "#2a78d6"   # blue
COLOR_PRED = "#e34948"   # red

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def _style_axes(ax):
    ax.grid(True, color=GRIDLINE, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(AXIS_LINE)


def plot_trajectory_paths(true_traj, pred_traj, save_path="trajectory_paths.png"):
    """3x3 small-multiples grid: one subplot per atom, ground truth vs predicted 2D path."""
    true_traj = np.array(true_traj)   # (T, 9, 2)
    pred_traj = np.array(pred_traj)   # (T, 9, 2)
    n_steps = min(len(true_traj), len(pred_traj))
    n_atoms = true_traj.shape[1]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), facecolor=SURFACE)
    for atom in range(n_atoms):
        ax = axes.flat[atom]
        ax.set_facecolor(SURFACE)
        ax.plot(true_traj[:n_steps, atom, 0], true_traj[:n_steps, atom, 1],
                color=COLOR_TRUE, linewidth=2, label="Ground truth")
        ax.plot(pred_traj[:n_steps, atom, 0], pred_traj[:n_steps, atom, 1],
                color=COLOR_PRED, linewidth=2, linestyle="--", label="Predicted")
        ax.scatter(*true_traj[0, atom], color=COLOR_TRUE, s=24, zorder=3)
        ax.set_title(f"Atom {atom}", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=10, labelcolor=INK_PRIMARY, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Simulated vs. ground-truth atom trajectories", fontsize=13,
                 color=INK_PRIMARY, y=1.06)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_deviation(true_traj, pred_traj, dt_per_snapshot=0.01, save_path="trajectory_deviation.png"):
    """Single line: RMSE across all atoms between predicted and ground-truth positions, over time."""
    true_traj = np.array(true_traj)
    pred_traj = np.array(pred_traj)
    n_steps = min(len(true_traj), len(pred_traj))
    rmse = np.sqrt(((true_traj[:n_steps] - pred_traj[:n_steps]) ** 2).mean(axis=(1, 2)))
    time = np.arange(n_steps) * dt_per_snapshot

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(time, rmse, color=COLOR_TRUE, linewidth=2)
    ax.set_xlabel("simulation time", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("position RMSE (all atoms)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Deviation from ground truth over time", fontsize=12, color=INK_PRIMARY)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
