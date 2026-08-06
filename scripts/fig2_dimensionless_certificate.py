"""Figure 2: sharpened certificate and dimensionless certified regions."""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from lhn import (
    LHNParams, exact_schur_scalar, effective_markov_symbol,
    classical_point_gap_margin, analytical_error_bound,
    pointwise_gap_reserve, winding_certificate,
    interval_validated_pointwise_certificate, dimensionless_coordinates,
    centered_dimensionless_uniform_reserve,
    centered_dimensionless_pointwise_reserve,
)
from lhn.style import use_style, C, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
LAM = -0.4
BASE = dict(N=10, J=1.0, G_R=1.0, G_L=0.35, pbc=True)

# thresholds on the original parameter path
def global_reserve(g, uniform):
    p = LHNParams(gphi=float(g), **BASE)
    return winding_certificate(p, LAM, uniform_in_N=uniform)["reserve"]

def point_reserve_dense(g):
    p = LHNParams(gphi=float(g), **BASE)
    q = np.linspace(0, 2*np.pi, 20001, endpoint=False)
    return float(np.min(pointwise_gap_reserve(q, p, LAM, uniform_in_N=True)))

thr_finite = brentq(lambda x: global_reserve(x, False), 6.0, 9.0)
thr_uniform = brentq(lambda x: global_reserve(x, True), 7.0, 10.0)
thr_point = brentq(point_reserve_dense, 4.2, 4.5)
# interval validation at a rounded value, used in the manuscript
p433 = LHNParams(gphi=4.33, **BASE)
validated = interval_validated_pointwise_certificate(
    p433, LAM, uniform_in_N=True, initial_intervals=256
)
assert validated["certified"]

fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))

gs = np.linspace(4.05, 30.0, 150)
actual, finite, uniform, margins, pres = [], [], [], [], []
q_actual = np.linspace(0, 2*np.pi, 4096, endpoint=False)
q_point = np.linspace(0, 2*np.pi, 8192, endpoint=False)
for g in gs:
    p = LHNParams(gphi=float(g), **BASE)
    actual.append(max(abs(exact_schur_scalar(q, p, LAM) -
                          (effective_markov_symbol(q, p)-LAM)) for q in q_actual))
    finite.append(analytical_error_bound(p, LAM, uniform_in_N=False))
    uniform.append(analytical_error_bound(p, LAM, uniform_in_N=True))
    margins.append(classical_point_gap_margin(p, LAM))
    pres.append(np.min(pointwise_gap_reserve(q_point, p, LAM, uniform_in_N=True)))

# (a) global bounds
ax = axes[0, 0]
ax.semilogy(gs, actual, color=C["topo"], lw=2.0, label="actual scalar error")
ax.semilogy(gs, finite, color=C["accent"], lw=1.6, ls="--", label="finite-$N$ global bound")
ax.semilogy(gs, uniform, color=C["triv"], lw=1.6, ls=":", label="all-$N$ global bound")
ax.semilogy(gs, margins, color=C["classical"], lw=2.0, label="classical point-gap margin")
ax.axvline(thr_finite, color=C["accent"], lw=1.0, ls="--")
ax.axvline(thr_uniform, color=C["triv"], lw=1.0, ls=":")
ax.set_xlabel(r"$\gamma_\phi$"); ax.set_ylabel("rate")
ax.set_title("Global gap-versus-error certificate")
ax.legend(fontsize=7.2, loc="upper right")
panel_label(ax, "a")

# (b) pointwise reserve
ax = axes[0, 1]
ax.plot(gs, pres, color=C["topo"], lw=2.2,
        label=r"$\min_q\{|W_{\rm eff}-\lambda_0|-\epsilon(q)\}$")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(thr_point, color=C["accent"], lw=1.8, ls="--",
           label=fr"threshold $\gamma_\phi\simeq{thr_point:.3f}$")
ax.axvline(thr_uniform, color=C["triv"], lw=1.2, ls=":",
           label=fr"global all-$N$: {thr_uniform:.2f}")
ax.scatter([4.33], [point_reserve_dense(4.33)], s=38, color="black", zorder=4,
           label="interval-validated at 4.33")
ax.set_xlabel(r"$\gamma_\phi$"); ax.set_ylabel("pointwise reserve")
ax.set_title("Momentum-resolved all-$N$ certificate")
ax.legend(fontsize=7.1, loc="lower right")
panel_label(ax, "b")

# dimensionless maps at centered reference and fixed delta
coords = dimensionless_coordinates(LHNParams(gphi=8.0, **BASE), -(
    BASE["G_R"] + BASE["G_L"] + 4*BASE["J"]**2/(8.0+BASE["G_R"]+BASE["G_L"])))
delta0 = coords["delta"]
js = np.linspace(0.005, 0.19, 150)
gvals = np.linspace(0.45, 0.95, 150)
J, G = np.meshgrid(js, gvals)
RU = np.full_like(J, np.nan, dtype=float)
RP = np.full_like(J, np.nan, dtype=float)
qmap = np.linspace(0, 2*np.pi, 720, endpoint=False)
for iy, g in enumerate(gvals):
    for ix, j in enumerate(js):
        s = 1-g+4*j*j
        physical = 0 <= g < 1 and abs(delta0)*s <= 1-g + 1e-14
        if not physical:
            continue
        ru = centered_dimensionless_uniform_reserve(j, g, delta0)
        rp = np.min(centered_dimensionless_pointwise_reserve(qmap, j, g, delta0))
        RU[iy, ix] = ru if np.isfinite(ru) else -1.0
        RP[iy, ix] = rp if np.isfinite(rp) else -1.0

j0, g0 = coords["j"], coords["g"]
for ax, R, title, letter in [
    (axes[1,0], RU, "Global dimensionless certificate", "c"),
    (axes[1,1], RP, "Pointwise dimensionless certificate", "d"),
]:
    mask = np.ma.masked_invalid(R)
    vmax = np.nanpercentile(R[np.isfinite(R)], 95)
    vmin = -max(0.02, abs(np.nanpercentile(R[np.isfinite(R)], 10)))
    im = ax.pcolormesh(J, G, mask, cmap="RdBu", shading="auto", vmin=vmin, vmax=vmax)
    ax.contour(J, G, np.where(np.isfinite(R), R, -1), levels=[0], colors="black", linewidths=1.2)
    ax.scatter([j0], [g0], marker="*", s=95, color="white", edgecolor="black", zorder=5,
               label="main-text parameters")
    ax.set_xlabel(r"$j=|J|/\kappa$"); ax.set_ylabel(r"$g=\gamma_\phi/\kappa$")
    ax.set_title(title + fr" ($\delta={delta0:.3f}$)")
    ax.legend(loc="lower left", fontsize=7.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.03)
    cb.set_label("dimensionless reserve")
    panel_label(ax, letter)

fig.tight_layout()
path = os.path.join(OUT, "fig2_dimensionless_certificate.png")
fig.savefig(path)
print(f"finite-N global threshold = {thr_finite:.6f}")
print(f"all-N global threshold = {thr_uniform:.6f}")
print(f"all-N pointwise threshold = {thr_point:.6f}")
print("interval validation at 4.33:", validated)
print("saved", path)
