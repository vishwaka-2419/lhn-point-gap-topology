"""FIGURE 2 -- Exact reduction and analytical quantum-to-classical certificate.

Panels:
(a) Liouvillian spectra as dephasing separates coherence-dominated branches.
(b) Exact finite-N scalar Schur complement and the second-order Markov symbol at
    the fixed reference point lambda_0.
(c) Actual scalar error, finite-N and uniform-in-N analytical bounds, and the
    exact classical point-gap margin.
(d) Full-Liouvillian and effective-Markov windings at the fixed reference point,
    together with the full-Liouvillian minimum-singular-value margin. The
    vertical line marks the conservative uniform-in-N certificate.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from lhn import (
    LHNParams,
    analytical_error_bound,
    classical_point_gap_margin,
    effective_markov_symbol,
    exact_schur_scalar,
    liouvillian,
    winding_certificate,
    winding_number,
)
from lhn.models import liouvillian_block_q
from lhn.topology import point_gap_margin
from lhn.style import use_style, C, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)   # so a fresh clone works even without figures/
LAM = -0.4
BASE = dict(N=10, J=1.0, G_R=1.0, G_L=0.35, pbc=True)


def uniform_threshold(lo: float = 4.0, hi: float = 30.0) -> float:
    """Bisection for the first uniform-in-N certified dephasing value."""
    def f(g: float) -> float:
        p = LHNParams(gphi=g, **BASE)
        cert = winding_certificate(p, LAM, uniform_in_N=True)
        return float(cert["classical_margin"] - cert["error_bound"])

    # Move lo above the resolvent-convergence threshold if needed.
    xs = np.linspace(lo, hi, 2000)
    vals = np.array([f(x) for x in xs])
    idx = np.where(np.isfinite(vals) & (vals > 0))[0]
    if len(idx) == 0:
        raise RuntimeError("No certified region found in the requested interval.")
    j = idx[0]
    a = xs[max(0, j - 1)]
    b = xs[j]
    for _ in range(70):
        m = 0.5 * (a + b)
        if f(m) > 0:
            b = m
        else:
            a = m
    return b


threshold_all = uniform_threshold()
fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.1))

# ---- (a) spectra vs dephasing -------------------------------------------
ax = axes[0, 0]
gphis = [0.0, 2.0, 8.0, 30.0]
cols = plt.cm.viridis(np.linspace(0.05, 0.8, len(gphis)))
for g, col in zip(gphis, cols):
    p = LHNParams(gphi=g, **BASE)
    ev = np.linalg.eigvals(liouvillian(p))
    ax.plot(ev.real, ev.imag, "o", ms=3.0, color=col, mec="k", mew=0.2,
            label=fr"$\gamma_\phi={g:g}$", alpha=0.88)
ax.set_xlim(-36, 1.5)
ax.set_xlabel(r"Re $\lambda$")
ax.set_ylabel(r"Im $\lambda$")
ax.set_title("Separation of coherence-dominated branches")
ax.legend(loc="lower left", fontsize=7.5)
panel_label(ax, "a")

# ---- (b) exact scalar reduction and Markov symbol -----------------------
ax = axes[0, 1]
qs = np.linspace(0.0, 2.0 * np.pi, 700)
for g, col in zip([8.0, 12.0, 25.0], plt.cm.viridis(np.linspace(0.1, 0.75, 3))):
    p = LHNParams(gphi=g, **BASE)
    exact = np.array([exact_schur_scalar(q, p, LAM) for q in qs])
    markov = np.array([effective_markov_symbol(q, p) - LAM for q in qs])
    ax.plot(exact.real, exact.imag, lw=1.9, color=col,
            label=fr"exact $S_N$, $\gamma_\phi={g:g}$")
    ax.plot(markov.real, markov.imag, "--", lw=1.3, color=col, alpha=0.95)
ax.plot(0.0, 0.0, "x", ms=8, mew=2.0, color=C["accent"], label="reference point")
ax.set_xlabel(r"Re $S_N(q,\lambda_0)$")
ax.set_ylabel(r"Im $S_N(q,\lambda_0)$")
ax.set_title("Exact scalar reduction and Markov limit")
ax.legend(loc="center", fontsize=7.1)
panel_label(ax, "b")

# ---- (c) analytical certificate ----------------------------------------
ax = axes[1, 0]
gs_cert = np.linspace(4.2, 30.0, 170)
actual, finite_bound, uniform_bound, margins_cl = [], [], [], []
qerr = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
for g in gs_cert:
    p = LHNParams(gphi=float(g), **BASE)
    actual.append(max(
        abs(exact_schur_scalar(q, p, LAM) - (effective_markov_symbol(q, p) - LAM))
        for q in qerr
    ))
    finite_bound.append(analytical_error_bound(p, LAM))
    uniform_bound.append(analytical_error_bound(p, LAM, uniform_in_N=True))
    margins_cl.append(classical_point_gap_margin(p, LAM))
actual = np.asarray(actual)
finite_bound = np.asarray(finite_bound)
uniform_bound = np.asarray(uniform_bound)
margins_cl = np.asarray(margins_cl)
ax.semilogy(gs_cert, actual, lw=1.9, color=C["topo"], label="actual scalar error")
ax.semilogy(gs_cert, finite_bound, "--", lw=1.7, color=C["classical"],
            label=r"finite-$N$ bound")
ax.semilogy(gs_cert, uniform_bound, ":", lw=2.0, color=C["accent"],
            label=r"uniform-in-$N$ bound")
ax.semilogy(gs_cert, margins_cl, lw=2.0, color=C["triv"],
            label="classical point-gap margin")
ax.axvspan(threshold_all, gs_cert[-1], color=C["gray"], alpha=0.12,
           label=fr"certified for all finite $N\geq4$ ($\gamma_\phi\geq{threshold_all:.2f}$)")
ax.set_xlabel(r"dephasing rate $\gamma_\phi$")
ax.set_ylabel("magnitude")
ax.set_ylim(1e-4, 8)
ax.set_title("Gap-versus-error homotopy certificate")
ax.legend(loc="upper right", fontsize=6.9)
panel_label(ax, "c")

# ---- (d) winding continuation and full point-gap margin -----------------
ax = axes[1, 1]
gs = np.concatenate([[0.0], np.logspace(-2, 3, 35)])
w_full, w_eff, full_margins = [], [], []
for g in gs:
    p = LHNParams(gphi=float(g), **BASE)
    full_fn = lambda q, p=p: liouvillian_block_q(q, p)
    eff_fn = lambda q, p=p: np.array([[effective_markov_symbol(q, p)]], dtype=complex)
    w_full.append(winding_number(full_fn, lam=LAM, n_q=1024))
    w_eff.append(winding_number(eff_fn, lam=LAM, n_q=1024))
    full_margins.append(point_gap_margin(full_fn, lam=LAM, n_q=512))
x = np.maximum(gs, 1e-3)
ax.plot(x, w_full, "o-", ms=3.8, lw=1.8, color=C["topo"], mec="k", mew=0.25,
        label="full Liouvillian")
ax.plot(x, w_eff, "s--", ms=3.2, lw=1.4, color=C["classical"], mec="k", mew=0.2,
        label="effective Markov generator")
ax.axvline(threshold_all, color=C["accent"], ls=":", lw=2.0,
           label=fr"uniform certificate $\gamma_\phi={threshold_all:.2f}$")
ax.set_xscale("log")
ax.set_ylim(-1.4, 0.3)
ax.set_yticks([-1, 0])
ax.set_xlabel(r"dephasing rate $\gamma_\phi$")
ax.set_ylabel(r"winding $w(\lambda_0)$")
ax.set_title(fr"Fixed reference point $\lambda_0={LAM:.1f}$")
ax.legend(loc="lower left", fontsize=6.9)
ax2 = ax.twinx()
ax2.plot(x, full_margins, "-.", lw=1.4, color=C["gray"])
ax2.set_ylabel(r"$\min_q\sigma_{\min}(\mathcal{L}_q-\lambda_0)$", color=C["gray"])
ax2.tick_params(axis="y", colors=C["gray"])
ax2.set_ylim(0.0, max(full_margins) * 1.25)
ax2.grid(False)
panel_label(ax, "d")

fig.suptitle("Analytically controlled continuation from Lindblad to Markov dynamics",
             fontsize=12.6, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
path = os.path.join(OUT, "fig2_superdecoherence.png")
fig.savefig(path)
print(f"uniform-in-N certificate threshold: gamma_phi = {threshold_all:.6f}")
print(f"minimum full-Liouvillian point-gap margin: {min(full_margins):.6f}")
print("full-Liouvillian windings:", sorted(set(w_full)))
print("effective-Markov windings:", sorted(set(w_eff)))
print("saved", path)
