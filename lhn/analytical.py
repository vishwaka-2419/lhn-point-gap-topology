"""Exact finite-size reduction and winding certificate.

This module provides analytical functions for the momentum-resolved
single-excitation Lindblad Hatano--Nelson model:

* an exact finite-N characteristic determinant;
* an exact scalar Schur complement after eliminating coherences;
* the second-order effective Markov symbol;
* a uniform error bound and sufficient homotopy criterion proving equality of
  the full-Liouvillian and effective-Markov point-gap windings.

The certificate is stated for a real reference point and N >= 4.
"""
from __future__ import annotations

import math
import numpy as np

from .models import LHNParams

__all__ = [
    "coherent_entries",
    "population_entry",
    "coherence_eigenvalues",
    "theta_sequence",
    "exact_characteristic_determinant",
    "exact_schur_scalar",
    "effective_markov_symbol",
    "classical_winding_real_reference",
    "classical_point_gap_margin",
    "analytical_error_bound",
    "winding_certificate",
]


def _validate(p: LHNParams) -> None:
    if p.N < 4:
        raise ValueError("The analytical remainder bound is stated for N >= 4.")
    if not p.pbc:
        raise ValueError("The momentum-block analytical reduction requires pbc=True.")


def coherent_entries(q: float, p: LHNParams) -> tuple[complex, complex]:
    """Return the coherent relative-coordinate entries u(q) and v(q)."""
    u = -1j * p.J * (1.0 - np.exp(1j * q))
    v = -1j * p.J * (1.0 - np.exp(-1j * q))
    return complex(u), complex(v)


def population_entry(q: float, p: LHNParams, lam: float) -> complex:
    """Return A(q,lambda) = [L_q]_{00} - lambda."""
    return (
        -(p.G_R + p.G_L)
        + p.G_R * np.exp(-1j * q)
        + p.G_L * np.exp(1j * q)
        - lam
    )


def coherence_eigenvalues(q: float, p: LHNParams, lam: float) -> np.ndarray:
    """Exact eigenvalues of the (N-1)-dimensional coherence block.

    The block is D(q,lambda)=-(kappa+lambda)I+K(q), where K is
    anti-Hermitian. Its eigenvalues are

        -(kappa+lambda) + 4 i J sin(q/2) cos(m pi/N), m=1,...,N-1.
    """
    _validate(p)
    m = np.arange(1, p.N)
    alpha = p.gphi + p.G_R + p.G_L + lam
    return -alpha + 4j * p.J * np.sin(q / 2.0) * np.cos(m * np.pi / p.N)


def theta_sequence(q: float, p: LHNParams, lam: float) -> np.ndarray:
    """Leading-principal determinants of the coherence Toeplitz block."""
    _validate(p)
    alpha = p.gphi + p.G_R + p.G_L + lam
    u, v = coherent_entries(q, p)
    theta = np.empty(p.N, dtype=complex)  # theta_0,...,theta_{N-1}
    theta[0] = 1.0
    theta[1] = -alpha
    for m in range(2, p.N):
        theta[m] = -alpha * theta[m - 1] - u * v * theta[m - 2]
    return theta


def exact_characteristic_determinant(q: float, p: LHNParams, lam: float) -> complex:
    """Exact det[L_q-lambda I] for finite N >= 4."""
    u, v = coherent_entries(q, p)
    theta = theta_sequence(q, p, lam)
    A = population_entry(q, p, lam)
    return (
        A * theta[p.N - 1]
        - 2.0 * u * v * theta[p.N - 2]
        - ((-1) ** p.N) * (u**p.N + v**p.N)
    )


def exact_schur_scalar(q: float, p: LHNParams, lam: float) -> complex:
    """Exact scalar Schur complement controlling the determinant winding."""
    u, v = coherent_entries(q, p)
    theta = theta_sequence(q, p, lam)
    denominator = theta[p.N - 1]
    if abs(denominator) < 1e-14:
        raise ZeroDivisionError("The coherence block is singular at this q and lambda.")
    correction = (
        2.0 * u * v * theta[p.N - 2]
        + ((-1) ** p.N) * (u**p.N + v**p.N)
    ) / denominator
    return population_entry(q, p, lam) - correction


def effective_markov_symbol(q: float, p: LHNParams) -> complex:
    """Second-order effective biased-walk symbol W_eff(q)."""
    kappa = p.gphi + p.G_R + p.G_L
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    D = 2.0 * p.J**2 / kappa
    rate_R = p.G_R + D
    rate_L = p.G_L + D
    return (
        rate_R * np.exp(-1j * q)
        + rate_L * np.exp(1j * q)
        - (rate_R + rate_L)
    )


def classical_winding_real_reference(p: LHNParams, lam: float) -> int:
    """Exact winding of W_eff(q) about a real reference point."""
    _validate(p)
    kappa = p.gphi + p.G_R + p.G_L
    D = 2.0 * p.J**2 / kappa
    sigma = p.G_R + p.G_L + 2.0 * D
    delta = p.G_R - p.G_L
    if delta == 0:
        return 0
    if -2.0 * sigma < lam < 0.0:
        return -1 if delta > 0 else 1
    return 0


def classical_point_gap_margin(p: LHNParams, lam: float) -> float:
    """Exact min_q |W_eff(q)-lambda| for a real reference point."""
    _validate(p)
    kappa = p.gphi + p.G_R + p.G_L
    D = 2.0 * p.J**2 / kappa
    sigma = p.G_R + p.G_L + 2.0 * D
    delta = p.G_R - p.G_L
    x = -float(lam)

    denominator = sigma**2 - delta**2
    if denominator <= 0:
        qs = np.linspace(0.0, 2.0 * np.pi, 200_001, endpoint=False)
        values = np.array([effective_markov_symbol(q, p) - lam for q in qs])
        return float(np.min(np.abs(values)))

    t_star = (x * sigma - delta**2) / denominator
    t = float(np.clip(t_star, 0.0, 2.0))
    margin_sq = (x - sigma * t) ** 2 + delta**2 * (2.0 * t - t**2)
    return math.sqrt(max(0.0, margin_sq))


def analytical_error_bound(
    p: LHNParams,
    lam: float,
    *,
    uniform_in_N: bool = False,
) -> float:
    """Bound sup_q |S_N(q,lambda)-[W_eff(q)-lambda]|.

    For alpha=kappa+lambda and c_N=cos(pi/N), the finite-N expression is

      8 J^2 |lambda|/(kappa alpha)
      + 128 J^4 c_N^2/[alpha^3(1-4|J|c_N/alpha)].

    Setting ``uniform_in_N=True`` replaces c_N by one and yields a sufficient
    bound valid for every finite N >= 4.
    """
    _validate(p)
    kappa = p.gphi + p.G_R + p.G_L
    alpha = kappa + lam
    if alpha <= 0:
        return math.inf
    c_n = 1.0 if uniform_in_N else math.cos(math.pi / p.N)
    eta = 4.0 * abs(p.J) * c_n / alpha
    if eta >= 1.0:
        return math.inf
    frequency_term = 8.0 * p.J**2 * abs(lam) / (kappa * alpha)
    remainder_term = 128.0 * p.J**4 * c_n**2 / (
        alpha**3 * (1.0 - eta)
    )
    return frequency_term + remainder_term


def winding_certificate(
    p: LHNParams,
    lam: float,
    *,
    uniform_in_N: bool = False,
) -> dict[str, float | int | bool]:
    """Return a sufficient certificate for w_L(lambda)=w_eff(lambda)."""
    _validate(p)
    kappa = p.gphi + p.G_R + p.G_L
    alpha = kappa + lam
    c_n = 1.0 if uniform_in_N else math.cos(math.pi / p.N)
    margin = classical_point_gap_margin(p, lam)
    bound = analytical_error_bound(p, lam, uniform_in_N=uniform_in_N)
    coherence_condition = alpha > 4.0 * abs(p.J) * c_n
    certified = bool(coherence_condition and margin > bound)
    return {
        "alpha": alpha,
        "coherence_condition": coherence_condition,
        "classical_margin": margin,
        "error_bound": bound,
        "certified": certified,
        "predicted_winding": classical_winding_real_reference(p, lam),
    }
