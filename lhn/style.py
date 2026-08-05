"""Shared matplotlib style for poster-quality figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(
    topo="#1b6ca8",      # topological / non-reciprocal
    triv="#c44536",      # trivial / reciprocal
    classical="#2a9d3f",  # classical Markov
    ps="#7b5aa6",        # postselected
    gray="#5a5a5a",
    accent="#e08a1e",
)


def use_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


def panel_label(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", va="top", ha="left")
