"""Figure 1: model, exact finite-size reduction, and scalar winding."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from lhn import (
    LHNParams, liouvillian_block_q, winding_map, exact_schur_scalar,
    effective_markov_symbol, postselected_hamiltonian,
)
from lhn.style import use_style, C, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

p = LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.35, gphi=8.0, pbc=True)
lam = -0.4
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))

# (a) model schematic
ax = axes[0, 0]
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-1.5, 1.8); ax.axis("off")
xs = np.arange(5)
for x in xs:
    ax.plot(x, 0, "o", ms=15, mfc="white", mec="black", mew=1.2)
    ax.text(x, 0, f"{x+1}", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(x, 1.25), xytext=(x, 0.35),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=C["gray"]))
    ax.text(x+0.06, 1.38, r"$\gamma_\phi$", ha="center", fontsize=8)
for x in range(4):
    ax.add_patch(FancyArrowPatch((x+0.12, 0.22), (x+0.88, 0.22),
                                arrowstyle="-|>", mutation_scale=11,
                                lw=2.0, color=C["topo"]))
    ax.add_patch(FancyArrowPatch((x+0.88, -0.22), (x+0.12, -0.22),
                                arrowstyle="-|>", mutation_scale=11,
                                lw=1.5, color=C["triv"]))
    ax.plot([x+0.15, x+0.85], [0.55, 0.55], color="black", lw=1.6)
    ax.text(x+0.5, 0.72, r"$J$", ha="center", fontsize=8)
ax.text(2, -0.72, r"directional recycling: $\Gamma_R\neq\Gamma_L$", ha="center")
ax.text(2, -1.12, r"reciprocal coherent hopping: $J=J^*$", ha="center")
ax.text(0.0, 1.65, r"$\rho\mapsto\mathcal{L}[\rho]$", fontsize=11, fontweight="bold")
ax.text(4.95, 0.25, r"$\Gamma_R$", color=C["topo"], fontsize=9)
ax.text(4.95, -0.28, r"$\Gamma_L$", color=C["triv"], fontsize=9)
ax.set_title("Liouvillian Hatano--Nelson chain")
panel_label(ax, "a", dx=-0.08, dy=1.02)

# (b) winding map around slow band
ax = axes[0, 1]
qs = np.linspace(0, 2*np.pi, 360, endpoint=False)
band = np.concatenate([np.linalg.eigvals(liouvillian_block_q(q, p)) for q in qs])
slow = band[band.real > -4.6]
RE, IM, W = winding_map(lambda q: liouvillian_block_q(q, p),
                        (-4.6, 0.5), (-1.0, 1.0),
                        n_re=105, n_im=85, n_q=192)
m = ax.pcolormesh(RE, IM, W, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
ax.plot(slow.real, slow.imag, ".", ms=1.0, color="black", alpha=0.48,
        label="population-associated PBC band")
ax.plot([lam], [0], marker="*", ms=11, mfc="white", mec="black", mew=0.9,
        label=fr"$\lambda_0={lam}$")
cb = fig.colorbar(m, ax=ax, fraction=0.047, pad=0.03)
cb.set_label(r"$w(\lambda)$")
ax.set_xlabel(r"Re $\lambda$"); ax.set_ylabel(r"Im $\lambda$")
ax.set_title("Full Liouvillian point gap")
ax.legend(loc="lower left", fontsize=7.5)
panel_label(ax, "b")

# (c) exact Schur loops and Markov approximation
ax = axes[1, 0]
for gph, col in [(4.33, C["accent"]), (8.0, C["topo"])]:
    pg = p.copy(gphi=gph)
    s = np.array([exact_schur_scalar(q, pg, lam) for q in qs])
    w = np.array([effective_markov_symbol(q, pg) - lam for q in qs])
    ax.plot(s.real, s.imag, color=col, lw=2.0,
            label=fr"exact $S_N$, $\gamma_\phi={gph:g}$")
    ax.plot(w.real, w.imag, color=col, lw=1.2, ls="--",
            label=fr"Markov, $\gamma_\phi={gph:g}$")
ax.plot([0], [0], "x", color="black", ms=8, mew=1.3)
ax.axhline(0, color="0.75", lw=0.6); ax.axvline(0, color="0.75", lw=0.6)
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel(r"Re $[S_N(q,\lambda_0)]$")
ax.set_ylabel(r"Im $[S_N(q,\lambda_0)]$")
ax.set_title("Exact scalar reduction and Markov limit")
ax.legend(loc="best", fontsize=7.1, ncol=2)
panel_label(ax, "c")

# (d) determinant phase factorisation
ax = axes[1, 1]
full_phase, schur_phase, coh_phase = [], [], []
for q in qs:
    M = liouvillian_block_q(q, p) - lam*np.eye(p.N)
    D = M[1:, 1:]
    s = exact_schur_scalar(q, p, lam)
    full_phase.append(np.angle(np.linalg.det(M)))
    schur_phase.append(np.angle(s))
    coh_phase.append(np.angle(np.linalg.det(D)))
for arr, lab, col, ls in [
    (full_phase, r"$\det(\mathcal{L}_q-\lambda_0)$", C["topo"], "-"),
    (schur_phase, r"$S_N(q,\lambda_0)$", C["classical"], "--"),
    (coh_phase, r"$\det D_{\lambda_0}(q)$", C["gray"], ":"),
]:
    ph = np.unwrap(np.r_[arr, arr[0]])
    ph = (ph-ph[0])/(2*np.pi)
    ax.plot(np.r_[qs, 2*np.pi]/np.pi, ph, ls=ls, lw=2.0, color=col, label=lab)
ax.set_xlabel(r"$q/\pi$"); ax.set_ylabel(r"accumulated phase$/2\pi$")
ax.set_title("Exact winding reduction at finite $N$")
ax.legend(loc="lower left", fontsize=7.4)
panel_label(ax, "d")

fig.tight_layout()
path = os.path.join(OUT, "fig1_reduction_topology.png")
fig.savefig(path)
print("saved", path)
