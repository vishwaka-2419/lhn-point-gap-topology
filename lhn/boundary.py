"""Boundary-geometry and eigenoperator diagnostics.

The physical Liouvillian acts on an N x N operator lattice.  The slow
population-associated sector has dimension N, whereas the full Liouville space
has dimension N^2.  In the strong-dephasing regime this module identifies and
quantifies the resulting subextensive skin sector.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig

from .models import LHNParams, liouvillian, liouvillian_block_q

__all__ = [
    "liouville_bloch_blocks",
    "synthetic_liouville_obc",
    "physical_eigenoperator_metrics",
    "slow_skin_sector",
]


def liouville_bloch_blocks(p: LHNParams, n_q: int = 16):
    """Return B0, Bplus, Bminus for L(q)=B0+Bplus e^{iq}+Bminus e^{-iq}."""
    if not p.pbc:
        raise ValueError("p must use periodic boundaries")
    qs = np.linspace(0.0, 2.0 * np.pi, int(n_q), endpoint=False)
    mats = np.array([liouvillian_block_q(q, p) for q in qs])
    b0 = mats.mean(axis=0)
    bplus = np.mean(mats * np.exp(-1j * qs)[:, None, None], axis=0)
    bminus = np.mean(mats * np.exp(1j * qs)[:, None, None], axis=0)
    residual = max(
        np.linalg.norm(mats[i] - (b0 + bplus * np.exp(1j * q) + bminus * np.exp(-1j * q)))
        for i, q in enumerate(qs)
    )
    return b0, bplus, bminus, float(residual)


def synthetic_liouville_obc(p: LHNParams, cells: int) -> np.ndarray:
    """Open the coordinate conjugate to q while retaining a fixed internal r-space."""
    cells = int(cells)
    if cells < 2:
        raise ValueError("cells must be at least 2")
    b0, bp, bm, residual = liouville_bloch_blocks(p)
    if residual > 1e-10:
        raise RuntimeError(f"unexpected higher Fourier harmonics: residual={residual:g}")
    rdim = p.N
    out = np.zeros((cells * rdim, cells * rdim), dtype=complex)
    for x in range(cells):
        sx = slice(x * rdim, (x + 1) * rdim)
        out[sx, sx] = b0
        if x + 1 < cells:
            sy = slice((x + 1) * rdim, (x + 2) * rdim)
            out[sy, sx] = bm
            out[sx, sy] = bp
    return out


def physical_eigenoperator_metrics(L: np.ndarray, N: int) -> dict[str, np.ndarray]:
    """Diagonalise L and return normalized right-eigenoperator diagnostics."""
    values, vectors = eig(L)
    a, b = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    centre = (a + b) / 2.0
    diagonal = a == b
    com = np.empty(N * N)
    ipr = np.empty(N * N)
    diag_weight = np.empty(N * N)
    right_edge_weight = np.empty(N * N)
    for k in range(N * N):
        X = vectors[:, k].reshape((N, N), order="F")
        weight = np.abs(X) ** 2
        norm = weight.sum()
        if norm <= 0:
            weight[:] = 0.0
        else:
            weight /= norm
        com[k] = float((centre * weight).sum() / max(1, N - 1))
        ipr[k] = float((weight**2).sum())
        diag_weight[k] = float(weight[diagonal].sum())
        right_edge_weight[k] = float(weight[centre >= 0.8 * (N - 1)].sum())
    return {
        "eigenvalues": values,
        "eigenvectors": vectors,
        "centre_of_mass": com,
        "ipr": ipr,
        "diagonal_weight": diag_weight,
        "right_edge_weight": right_edge_weight,
    }


def slow_skin_sector(p: LHNParams, diagonal_threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Return metrics and mask for the population-associated OBC skin sector."""
    if p.pbc:
        p = p.copy(pbc=False)
    metrics = physical_eigenoperator_metrics(liouvillian(p), p.N)
    metrics["slow_mask"] = metrics["diagonal_weight"] > float(diagonal_threshold)
    return metrics
