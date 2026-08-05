"""
FIGURE 1 -- Point-gap winding and finite-size boundary diagnostics.

(a) Winding-number map with the PBC Liouvillian spectrum overlaid: an open point
    gap with w = -1.
(b) OBC spectrum collapses inside the PBC loop: strong boundary sensitivity.
(c) The BULK no-jump Hamiltonian H_eff is a line: zero point-gap WINDING for
    every reference point off it. Note carefully: a line segment still has point
    gaps -- any lambda off the segment is point-gapped. What vanishes is the
    winding, not the gap. Open boundaries additionally produce non-extensive
    edge shifts, so PBC and OBC spectra are not identical.
(d) OBC stationary population: a strongly nonuniform, approximately exponential profile at finite J. The exact p_j proportional to (G_R/G_L)^j result applies at J=0.
(e) Distribution of eigenoperator centres of mass under OBC. There is a clear
    directional shift relative to the reciprocal case.
(f) FINITE-SIZE DIAGNOSTICS. The fraction of eigenoperators localized near
    the edge decreases over the sizes studied, and the OBC dissipative gap closes
    more slowly than the PBC one. These calculations do not establish an
    extensive Liouvillian skin effect, although they do show boundary-sensitive
    spectra and a strongly nonuniform stationary state.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt

from lhn import LHNParams, liouvillian, postselected_hamiltonian, steady_state
from lhn.models import liouvillian_block_q
from lhn.topology import (winding_map, bloch_postselected, winding_number,
                          mode_centre_of_mass)
from lhn.style import use_style, C, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)   # so a fresh clone works even without figures/

N = 10
p = LHNParams(N=N, J=0.6, G_R=1.0, G_L=0.35, gphi=0.25, pbc=True)
ev_pbc = np.linalg.eigvals(liouvillian(p))
ev_obc = np.linalg.eigvals(liouvillian(p.copy(pbc=False)))

fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))

# ---- (a) winding map + PBC spectrum -------------------------------------
ax = axes[0, 0]
pad = 0.35
RE, IM, W = winding_map(lambda q: liouvillian_block_q(q, p),
                        (ev_pbc.real.min() - pad, ev_pbc.real.max() + pad),
                        (ev_pbc.imag.min() - pad, ev_pbc.imag.max() + pad),
                        n_re=110, n_im=110, n_q=176)
lim = max(1, int(np.abs(W).max()))
m = ax.pcolormesh(RE, IM, W, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
qs = np.linspace(0, 2 * np.pi, 400)
band = np.concatenate([np.linalg.eigvals(liouvillian_block_q(q, p)) for q in qs])
ax.plot(band.real, band.imag, ".", ms=0.7, color="k", alpha=0.35)
ax.plot(ev_pbc.real, ev_pbc.imag, "o", ms=3.2, mfc=C["topo"], mec="k", mew=0.3,
        label="PBC spectrum")
ax.plot([-0.4], [0.0], marker="*", ms=10, mfc="white", mec="k", mew=0.8,
        label=r"$\lambda_0=-0.4$ ($w=-1$)")
cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("winding $w(\\lambda)$")
ax.set_xlabel("Re $\\lambda$"); ax.set_ylabel("Im $\\lambda$")
ax.set_title("Liouvillian point gap")
ax.legend(loc="upper left", fontsize=8)
panel_label(ax, "a")

# ---- (b) PBC vs OBC ------------------------------------------------------
ax = axes[0, 1]
ax.plot(band.real, band.imag, "-", lw=0.6, color=C["gray"], alpha=0.55,
        label="PBC continuum")
ax.plot(ev_obc.real, ev_obc.imag, "s", ms=3.4, mfc=C["triv"], mec="k", mew=0.3,
        label="OBC spectrum")
ax.set_xlabel("Re $\\lambda$"); ax.set_ylabel("Im $\\lambda$")
ax.set_title("Boundary-sensitive spectrum")
ax.legend(loc="upper left", fontsize=8)
panel_label(ax, "b")

# ---- (c) no-jump Hamiltonian --------------------------------------------
ax = axes[0, 2]
ks = np.linspace(0, 2 * np.pi, 400)
psb = np.array([bloch_postselected(p)(k)[0, 0] for k in ks])
ev_ps = np.linalg.eigvals(postselected_hamiltonian(p))
ev_ps_o = np.linalg.eigvals(postselected_hamiltonian(p.copy(pbc=False)))
w_ps = winding_number(bloch_postselected(p), lam=psb.mean() - 0.6j)
ax.plot(psb.real, psb.imag, "-", lw=2.4, color=C["ps"], alpha=0.75,
        label="$H_{\\rm eff}(k)$ bulk: a line")
ax.plot(ev_ps.real, ev_ps.imag, "o", ms=4.0, mfc=C["ps"], mec="k", mew=0.3,
        label="PBC")
ax.plot(ev_ps_o.real, ev_ps_o.imag, "x", ms=5.5, color="k", mew=1.1,
        label="OBC: edge shifts")
ax.set_xlabel("Re $E$"); ax.set_ylabel("Im $E$")
ax.set_ylim(psb.imag.mean() - 1.0, psb.imag.mean() + 1.0)
ax.set_title(f"No-jump $H_{{\\rm eff}}$: zero winding ($w={w_ps}$)")
ax.legend(loc="upper left", fontsize=7.6)
panel_label(ax, "c")

# ---- (d) stationary population ------------------------------------------
ax = axes[1, 0]
for GR, GL, lab, col, mk in [(1.0, 0.35, "non-reciprocal ($w=-1$)", C["topo"], "o"),
                             (0.675, 0.675, "reciprocal ($w=0$)", C["triv"], "s")]:
    pp = p.copy(G_R=GR, G_L=GL, pbc=False, theta=0.0)
    rho = steady_state(liouvillian(pp), pp.N)
    prof = np.array(np.real(np.diag(rho)), dtype=float); prof /= prof.sum()
    ax.semilogy(np.arange(1, pp.N + 1), prof, mk + "-", color=col, ms=5,
                mec="k", mew=0.3, label=lab)
ax.set_xlabel("site $j$"); ax.set_ylabel("stationary population $p_j$")
ax.set_title("Edge-localised stationary state")
ax.legend(loc="upper left", fontsize=8)
panel_label(ax, "d")

# ---- (e) eigenoperator centre-of-mass distribution ----------------------
ax = axes[1, 1]
NC = 12
for GR, GL, lab, col in [(1.0, 0.35, "non-reciprocal", C["topo"]),
                         (0.675, 0.675, "reciprocal", C["triv"])]:
    pp = LHNParams(N=NC, J=0.6, G_R=GR, G_L=GL, gphi=0.25, pbc=False)
    com = mode_centre_of_mass(liouvillian(pp), NC)
    com = com[~np.isnan(com)] / (NC - 1)
    ax.hist(com, bins=22, range=(0, 1), alpha=0.55, color=col, label=lab,
            edgecolor="k", linewidth=0.4)
    ax.axvline(com.mean(), color=col, ls="--", lw=1.8)
ax.set_xlabel("eigenoperator centre of mass  $\\bar{j}/(N-1)$")
ax.set_ylabel("count")
ax.set_title(f"Mode localisation, OBC ($N={NC}$)")
ax.legend(loc="upper left", fontsize=8)
panel_label(ax, "e")

# ---- (f) finite-size diagnostics ----------------------------------------
ax = axes[1, 2]
Ns = np.array([6, 8, 10, 12, 14, 16])
frac, gapO, gapP = [], [], []
for n in Ns:
    pn = LHNParams(N=int(n), J=0.6, G_R=1.0, G_L=0.35, gphi=0.25, pbc=False)
    Lo = liouvillian(pn)
    com = mode_centre_of_mass(Lo, int(n))
    com = com[~np.isnan(com)]
    frac.append(np.mean(com > 0.7 * (n - 1)))
    e = np.linalg.eigvals(Lo); e = e[np.abs(e) > 1e-9 * max(1, np.abs(e).max())]
    gapO.append(-e.real.max())
    e = np.linalg.eigvals(liouvillian(pn.copy(pbc=True)))
    e = e[np.abs(e) > 1e-9 * max(1, np.abs(e).max())]
    gapP.append(-e.real.max())
ax.plot(Ns, frac, "o-", color=C["topo"], ms=5.5, mec="k", mew=0.3,
        label="fraction of modes with $\\bar{j}>0.7(N{-}1)$")
ax.set_xlabel("system size $N$")
ax.set_ylabel("edge-localised mode fraction", color=C["topo"])
ax.tick_params(axis="y", colors=C["topo"])
ax.set_ylim(0, 0.16)
ax2 = ax.twinx()
sO = np.polyfit(np.log(Ns), np.log(gapO), 1)[0]
sP = np.polyfit(np.log(Ns), np.log(gapP), 1)[0]
ax2.loglog(Ns, gapO, "s--", color=C["triv"], ms=4.5, mec="k", mew=0.3,
           label=f"OBC gap $\\propto N^{{{sO:.2f}}}$")
ax2.loglog(Ns, gapP, "^--", color=C["gray"], ms=4.5, mec="k", mew=0.3,
           label=f"PBC gap $\\propto N^{{{sP:.2f}}}$")
ax2.set_ylabel("dissipative gap")
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7.0)
ax.set_title("Finite-size boundary diagnostics")
panel_label(ax, "f")

fig.suptitle("Point-gap winding and boundary sensitivity",
             fontsize=12.5, y=1.005)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(os.path.join(OUT, "fig1_topology.png"))

print(f"w(Liouvillian) max on grid = {int(np.abs(W).max())};  w(H_eff) = {w_ps}")
print(f"edge-mode fraction vs N: {[round(f,3) for f in frac]}  (decreasing over sampled sizes)")
print(f"OBC gap ~ N^{sO:.3f};  PBC gap ~ N^{sP:.3f}  "
      f"(OBC closes more slowly over sampled sizes)")
print("saved figures/fig1_topology.png")
