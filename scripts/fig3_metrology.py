"""
FIGURE 3 -- Boundary-rate response and exceptional-point spectral response.

(a) Fisher information for estimating a weak boundary link theta, against system
    size. Biased (w = -1): exponential in N. Reciprocal (w = 0): N^2.
(b) At a genuine Liouvillian exceptional point the eigenvalue splitting responds
    as sqrt(eps) -- a divergent spectral susceptibility.
(c) ...and yet the quantum Fisher information is analytic through the EP, with no
    local maximum. The divergent susceptibility does not produce a corresponding singularity in the Fisher information FOR
    THIS TASK (stationary-state estimation of the drive in this model). That is
    not a no-go theorem for all EP sensors, probes, or protocols.

SCOPE: (a) is a linear-response result (theta -> 0) for one specific boundary
perturbation and an occupation readout. It is not a generic consequence of
point-gap topology, and it saturates once theta * rho^N reaches the bulk rates.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt

from lhn import LHNParams
from lhn.metrology import (lhn_sensor_fisher, ep_splitting, ep_liouvillian,
                           steady_state, d_steady_state,
                           quantum_fisher_information)
from lhn.style import use_style, C, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)   # so a fresh clone works even without figures/
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

# ---- (a) Fisher information vs system size ------------------------------
ax = axes[0]
Ns = np.arange(4, 27, 2)
for GR, GL, lab, col, mk in [(1.0, 0.4, "biased  $w = -1$", C["topo"], "o"),
                             (0.7, 0.7, "reciprocal  $w = 0$", C["triv"], "s")]:
    F = [lhn_sensor_fisher(LHNParams(N=int(N), J=0.5, G_R=GR, G_L=GL, gphi=0.2),
                           theta0=0.0)["F_C"] for N in Ns]
    F = np.array(F)
    ax.semilogy(Ns, F, mk + "-", color=col, ms=5.5, mec="k", mew=0.35, label=lab)
    if GR != GL:
        sl = np.polyfit(Ns[-5:], np.log(F[-5:]), 1)[0]
        ax.semilogy(Ns, np.exp(np.polyval(np.polyfit(Ns[-5:], np.log(F[-5:]), 1), Ns)),
                    ":", color=col, lw=1.3)
        ax.text(0.42, 0.34, f"$\\propto e^{{{sl:.2f}N}}$", transform=ax.transAxes,
                color=col, fontsize=10)
    else:
        pw = np.polyfit(np.log(Ns[-5:]), np.log(F[-5:]), 1)[0]
        ax.text(0.5, 0.10, f"$\\propto N^{{{pw:.2f}}}$", transform=ax.transAxes,
                color=col, fontsize=10)
ax.set_xlabel("system size $N$")
ax.set_ylabel("Fisher information  $F(\\theta \\to 0)$")
ax.set_title("Boundary-rate response in the local regime")
ax.legend(loc="upper left")
panel_label(ax, "a")

# ---- (b) EP square-root response ----------------------------------------
ax = axes[1]
gamma = 1.0
Om_ep = gamma / 4.0
eps = np.logspace(-8, -2, 40)
sp = np.array([ep_splitting(Om_ep + e, gamma, 0.0) for e in eps])
slope = np.polyfit(np.log(eps), np.log(sp), 1)[0]
ax.loglog(eps, sp, "-", lw=2.2, color=C["accent"], label="Liouvillian EP")
ax.loglog(eps, 2 * np.sqrt(gamma * eps / 2), "--", lw=1.4, color="k",
          label="$\\sqrt{\\epsilon}$ reference")
# A detuning perturbation at the EP acts outside the defective block and is linear.
sp_det = np.array([ep_splitting(Om_ep, gamma, e) for e in eps])
ax.loglog(eps, sp_det, "-", lw=1.8, color=C["gray"],
          label=r"detuning at EP  ($\propto \delta$)")
ax.set_xlabel(r"perturbation magnitude  $\epsilon$ or $\delta$")
ax.set_ylabel("eigenvalue splitting")
ax.set_title(f"Spectral response:  exponent $= {slope:.3f}$")
ax.legend(loc="upper left", fontsize=8)
panel_label(ax, "b")

# ---- (c) QFI is smooth through the EP -----------------------------------
ax = axes[2]
Oms = np.linspace(0.02, 0.6, 260)


def qfi_omega(Om, h=1e-6):
    L0 = ep_liouvillian(Om, gamma, 0.0)
    dL = (ep_liouvillian(Om + h, gamma, 0.0)
          - ep_liouvillian(Om - h, gamma, 0.0)) / (2 * h)
    r = steady_state(L0, 2)
    return quantum_fisher_information(r, d_steady_state(L0, dL, r, 2))


def suscept(Om):
    """Magnitude of the analytic derivative of the coalescing-pair splitting.

    On resonance the relevant splitting is
    2*sqrt(abs(Omega**2-(gamma/4)**2)).  Using this expression prevents the
    plotted diagnostic from switching to a different, accidentally closer
    Liouvillian eigenvalue pair far from the exceptional point.
    """
    a = gamma / 4.0
    den = np.sqrt(abs(Om * Om - a * a))
    return np.inf if den == 0 else 2.0 * Om / den


F = np.array([qfi_omega(o) for o in Oms])
S = np.array([suscept(o) for o in Oms])
ax.plot(Oms, S, "-", lw=1.8, color=C["accent"], label="spectral susceptibility")
ax.set_yscale("log")
ax.set_ylabel("$|\\,d(\\rm splitting)/d\\Omega\\,|$", color=C["accent"])
ax.tick_params(axis="y", colors=C["accent"])
ax.axvline(Om_ep, color="k", ls=":", lw=1.2)
ax.text(Om_ep + 0.012, ax.get_ylim()[1] * 0.25, "EP", fontsize=9)
ax2 = ax.twinx()
ax2.plot(Oms, F, "-", lw=2.4, color=C["topo"], label="quantum Fisher information")
ax2.set_ylabel("$F_Q$", color=C["topo"])
ax2.tick_params(axis="y", colors=C["topo"])
ax2.grid(False)
ax.set_xlabel("drive strength  $\\Omega$")
ax.set_title("Spectral susceptibility and stationary-state $F_Q$")
ax.text(0.02, 0.02, "drive estimation,\nstationary state", transform=ax.transAxes,
        fontsize=7.4, color=C["gray"], style="italic")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=8)
panel_label(ax, "c")

fig.suptitle("Boundary-rate Fisher information and exceptional-point spectral response "
             "(model-specific comparison)", fontsize=12.5, y=1.005)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_metrology.png"))
print(f"EP splitting exponent = {slope:.4f}")
print(f"F_Q at EP = {qfi_omega(Om_ep):.5f};  F_Q far from EP (Om=0.05) = {qfi_omega(0.05):.5f}")
print("saved figures/fig3_metrology.png")
