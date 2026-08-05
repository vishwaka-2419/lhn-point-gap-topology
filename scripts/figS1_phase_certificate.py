"""SUPPLEMENTARY FIGURE S1 -- analytical winding certificate over (gamma_phi, lambda).

The exact finite-N winding is computed from the closed-form determinant, while
the effective winding and the sufficient all-finite-N certificate are evaluated
analytically for real reference points.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from lhn import LHNParams
from lhn.analytical import (
    analytical_error_bound,
    classical_point_gap_margin,
    classical_winding_real_reference,
)
from lhn.style import use_style, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def determinant_curve(qs: np.ndarray, p: LHNParams, lam: float) -> np.ndarray:
    """Vectorized exact determinant from the finite-size recursion."""
    u = -1j * p.J * (1.0 - np.exp(1j * qs))
    v = -1j * p.J * (1.0 - np.exp(-1j * qs))
    alpha = p.gphi + p.G_R + p.G_L + lam
    th0 = np.ones_like(qs, dtype=complex)
    th1 = -alpha * np.ones_like(qs, dtype=complex)
    for _m in range(2, p.N):
        th = -alpha * th1 - u * v * th0
        th0, th1 = th1, th
    A = (-(p.G_R + p.G_L) + p.G_R * np.exp(-1j * qs)
         + p.G_L * np.exp(1j * qs) - lam)
    return A * th1 - 2.0 * u * v * th0 - ((-1) ** p.N) * (u**p.N + v**p.N)


def winding(curve: np.ndarray) -> int:
    closed = np.concatenate([curve, curve[:1]])
    phase = np.unwrap(np.angle(closed))
    return int(np.rint((phase[-1] - phase[0]) / (2.0 * np.pi)))


def main() -> None:
    gammas = np.logspace(-1, 2, 55)
    lambdas = np.linspace(-2.4, -0.04, 80)
    qs = np.linspace(0.0, 2.0 * np.pi, 768, endpoint=False)

    full = np.zeros((gammas.size, lambdas.size))
    eff = np.zeros_like(full)
    cert = np.zeros_like(full, dtype=bool)
    ratio = np.full_like(full, np.nan, dtype=float)

    for ig, gamma in enumerate(gammas):
        p = LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.35,
                      gphi=float(gamma), pbc=True)
        for il, lam in enumerate(lambdas):
            full[ig, il] = winding(determinant_curve(qs, p, float(lam)))
            eff[ig, il] = classical_winding_real_reference(p, float(lam))
            margin = classical_point_gap_margin(p, float(lam))
            bound = analytical_error_bound(p, float(lam), uniform_in_N=True)
            cert[ig, il] = np.isfinite(bound) and margin > bound
            if np.isfinite(bound) and margin > 0:
                ratio[ig, il] = bound / margin

    L, G = np.meshgrid(lambdas, gammas)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)

    im = axes[0].pcolormesh(L, G, full, shading="auto", vmin=-1, vmax=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"real reference point $\lambda$")
    axes[0].set_ylabel(r"dephasing rate $\gamma_\phi$")
    axes[0].set_title(r"Exact finite-$N$ winding")
    fig.colorbar(im, ax=axes[0], label=r"$w_{\mathcal{L}}$")
    panel_label(axes[0], "a")

    im = axes[1].pcolormesh(L, G, eff, shading="auto", vmin=-1, vmax=1)
    axes[1].contour(L, G, cert.astype(float), levels=[0.5], linewidths=1.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"real reference point $\lambda$")
    axes[1].set_title("Effective Markov winding\ncontour: uniform certificate")
    fig.colorbar(im, ax=axes[1], label=r"$w_{\rm eff}$")
    panel_label(axes[1], "b")

    masked = np.ma.masked_invalid(ratio)
    im = axes[2].pcolormesh(L, G, masked, shading="auto", vmin=0, vmax=1.5)
    axes[2].contour(L, G, cert.astype(float), levels=[0.5], linewidths=1.8)
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"real reference point $\lambda$")
    axes[2].set_title(r"Bound-to-margin ratio $\epsilon/m_{\rm cl}$")
    fig.colorbar(im, ax=axes[2], label=r"$\epsilon/m_{\rm cl}$")
    panel_label(axes[2], "c")

    fig.tight_layout()
    path = os.path.join(OUT, "figS1_phase_certificate.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    agreement = float(np.mean(full == eff))
    certified_agreement = float(np.mean((full == eff)[cert])) if np.any(cert) else float("nan")
    print(f"saved {path}")
    print(f"full/effective agreement over grid: {agreement:.6f}")
    print(f"analytically certified grid fraction: {np.mean(cert):.6f}")
    print(f"agreement inside certified region: {certified_agreement:.6f}")


if __name__ == "__main__":
    main()
