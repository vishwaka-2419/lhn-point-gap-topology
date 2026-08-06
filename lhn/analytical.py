"""Exact reduction and topology-preservation certificates.

The module implements the finite-size Schur reduction of the momentum-resolved
single-excitation Liouvillian Hatano--Nelson chain.  It provides both the
original momentum-uniform error estimate and a sharper momentum-resolved
estimate.  The latter can be validated on the full Brillouin zone by interval
subdivision, so no momentum-grid assumption enters the certificate.
"""
from __future__ import annotations

import math
from typing import Any

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
    "pointwise_error_bound",
    "pointwise_gap_reserve",
    "winding_certificate",
    "interval_validated_pointwise_certificate",
    "dimensionless_coordinates",
    "centered_dimensionless_uniform_reserve",
    "centered_dimensionless_pointwise_reserve",
]


def _validate(p: LHNParams) -> None:
    if p.N < 4:
        raise ValueError("The analytical remainder bound is stated for N >= 4.")
    if not p.pbc:
        raise ValueError("The momentum-block analytical reduction requires pbc=True.")


def coherent_entries(q: float, p: LHNParams) -> tuple[complex, complex]:
    """Return coherent relative-coordinate entries u(q) and v(q)."""
    u = -1j * p.J * (1.0 - np.exp(1j * q))
    v = -1j * p.J * (1.0 - np.exp(-1j * q))
    return complex(u), complex(v)


def population_entry(q: float, p: LHNParams, lam: float) -> complex:
    """Return A(q,lambda)=[L_q]_{00}-lambda."""
    return (
        -(p.G_R + p.G_L)
        + p.G_R * np.exp(-1j * q)
        + p.G_L * np.exp(1j * q)
        - lam
    )


def coherence_eigenvalues(q: float, p: LHNParams, lam: float) -> np.ndarray:
    """Exact eigenvalues of the (N-1)-dimensional coherence block."""
    _validate(p)
    m = np.arange(1, p.N)
    alpha = p.gphi + p.G_R + p.G_L + lam
    return -alpha + 4j * p.J * np.sin(q / 2.0) * np.cos(m * np.pi / p.N)


def theta_sequence(q: float, p: LHNParams, lam: float) -> np.ndarray:
    """Leading-principal determinants of the coherence Toeplitz block."""
    _validate(p)
    alpha = p.gphi + p.G_R + p.G_L + lam
    u, v = coherent_entries(q, p)
    theta = np.empty(p.N, dtype=complex)
    theta[0] = 1.0
    theta[1] = -alpha
    for m in range(2, p.N):
        theta[m] = -alpha * theta[m - 1] - u * v * theta[m - 2]
    return theta


def exact_characteristic_determinant(q: float, p: LHNParams, lam: float) -> complex:
    """Exact det[L_q-lambda I] for finite N>=4."""
    u, v = coherent_entries(q, p)
    theta = theta_sequence(q, p, lam)
    A = population_entry(q, p, lam)
    return (
        A * theta[p.N - 1]
        - 2.0 * u * v * theta[p.N - 2]
        - ((-1) ** p.N) * (u**p.N + v**p.N)
    )


def exact_schur_scalar(q: float, p: LHNParams, lam: float) -> complex:
    """Exact scalar Schur complement controlling determinant winding."""
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
    """Momentum-uniform bound on the Schur-to-Markov error."""
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


def pointwise_error_bound(
    q: float | np.ndarray,
    p: LHNParams,
    lam: float,
    *,
    uniform_in_N: bool = False,
) -> float | np.ndarray:
    """Momentum-resolved Schur-to-Markov error bound.

    This retains the factors sin^2(q/2) and sin^4(q/2) that are discarded by
    the global supremum.  It is therefore typically much sharper near the
    momentum at which the classical point-gap margin is smallest.
    """
    _validate(p)
    kappa = p.gphi + p.G_R + p.G_L
    alpha = kappa + lam
    q_arr = np.asarray(q, dtype=float)
    if alpha <= 0:
        out = np.full_like(q_arr, np.inf, dtype=float)
        return float(out) if out.ndim == 0 else out
    c_n = 1.0 if uniform_in_N else math.cos(math.pi / p.N)
    s = np.abs(np.sin(q_arr / 2.0))
    eta = 4.0 * abs(p.J) * c_n * s / alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (
            8.0 * p.J**2 * abs(lam) * s**2 / (kappa * alpha)
            + 128.0 * p.J**4 * c_n**2 * s**4
            / (alpha**3 * (1.0 - eta))
        )
    out = np.where(eta < 1.0, out, np.inf)
    return float(out) if out.ndim == 0 else out


def pointwise_gap_reserve(
    q: float | np.ndarray,
    p: LHNParams,
    lam: float,
    *,
    uniform_in_N: bool = False,
) -> float | np.ndarray:
    """Return |W_eff(q)-lambda|-epsilon(q). Positive for all q certifies winding equality."""
    q_arr = np.asarray(q, dtype=float)
    markov = np.array([effective_markov_symbol(float(x), p) for x in q_arr.ravel()])
    markov = markov.reshape(q_arr.shape)
    reserve = np.abs(markov - lam) - pointwise_error_bound(
        q_arr, p, lam, uniform_in_N=uniform_in_N
    )
    return float(reserve) if reserve.ndim == 0 else reserve


def winding_certificate(
    p: LHNParams,
    lam: float,
    *,
    uniform_in_N: bool = False,
) -> dict[str, float | int | bool]:
    """Return the original momentum-uniform sufficient certificate."""
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
        "reserve": margin - bound,
        "certified": certified,
        "predicted_winding": classical_winding_real_reference(p, lam),
    }


def _trig_range(kind: str, a: float, b: float) -> tuple[float, float]:
    fn = math.sin if kind == "sin" else math.cos
    vals = [fn(a), fn(b)]
    if kind == "sin":
        k0 = math.ceil((a - math.pi / 2.0) / math.pi)
        k1 = math.floor((b - math.pi / 2.0) / math.pi)
        vals.extend(fn(math.pi / 2.0 + k * math.pi) for k in range(k0, k1 + 1))
    else:
        k0 = math.ceil(a / math.pi)
        k1 = math.floor(b / math.pi)
        vals.extend(fn(k * math.pi) for k in range(k0, k1 + 1))
    return min(vals), max(vals)


def _distance_from_zero(lo: float, hi: float) -> float:
    if lo <= 0.0 <= hi:
        return 0.0
    return min(abs(lo), abs(hi))


def _interval_reserve_lower_bound(
    p: LHNParams,
    lam: float,
    a: float,
    b: float,
    c_n: float,
) -> float:
    """Rigorous lower bound on the pointwise reserve on q in [a,b]."""
    kappa = p.gphi + p.G_R + p.G_L
    alpha = kappa + lam
    if alpha <= 0:
        return -math.inf

    x = -float(lam)
    D = 2.0 * p.J**2 / kappa
    sigma = p.G_R + p.G_L + 2.0 * D
    delta = p.G_R - p.G_L

    sin_lo, sin_hi = _trig_range("sin", a, b)
    cos_lo, cos_hi = _trig_range("cos", a, b)

    re_lo = x - sigma * (1.0 - cos_lo)
    re_hi = x - sigma * (1.0 - cos_hi)
    if re_lo > re_hi:
        re_lo, re_hi = re_hi, re_lo
    im_lo, im_hi = -delta * sin_hi, -delta * sin_lo
    if im_lo > im_hi:
        im_lo, im_hi = im_hi, im_lo
    modulus_lower = math.hypot(
        _distance_from_zero(re_lo, re_hi),
        _distance_from_zero(im_lo, im_hi),
    )

    # epsilon is monotone in s=|sin(q/2)| on its convergence interval.
    s2_max = max(0.0, (1.0 - cos_lo) / 2.0)
    s_max = math.sqrt(s2_max)
    eta = 4.0 * abs(p.J) * c_n * s_max / alpha
    if eta >= 1.0:
        return -math.inf
    epsilon_upper = (
        8.0 * p.J**2 * abs(lam) * s2_max / (kappa * alpha)
        + 128.0 * p.J**4 * c_n**2 * s2_max**2
        / (alpha**3 * (1.0 - eta))
    )
    return modulus_lower - epsilon_upper


def interval_validated_pointwise_certificate(
    p: LHNParams,
    lam: float,
    *,
    uniform_in_N: bool = False,
    initial_intervals: int = 256,
    max_depth: int = 20,
) -> dict[str, Any]:
    """Validate the pointwise homotopy criterion on the full Brillouin zone.

    Each interval receives a lower bound on |W_eff-lambda| and an upper bound on
    epsilon(q). Intervals that are not immediately positive are bisected. A
    positive result is independent of a momentum grid.
    """
    _validate(p)
    if not isinstance(initial_intervals, int) or initial_intervals < 4:
        raise ValueError("initial_intervals must be an integer >=4")
    c_n = 1.0 if uniform_in_N else math.cos(math.pi / p.N)
    stack = [
        (2.0 * math.pi * i / initial_intervals,
         2.0 * math.pi * (i + 1) / initial_intervals,
         0)
        for i in range(initial_intervals)
    ]
    minimum_bound = math.inf
    checked = 0
    unresolved: tuple[float, float] | None = None
    while stack:
        a, b, depth = stack.pop()
        lower = _interval_reserve_lower_bound(p, lam, a, b, c_n)
        minimum_bound = min(minimum_bound, lower)
        checked += 1
        if lower > 0.0:
            continue
        if depth >= max_depth:
            unresolved = (a, b)
            break
        mid = (a + b) / 2.0
        stack.append((a, mid, depth + 1))
        stack.append((mid, b, depth + 1))

    certified = unresolved is None
    return {
        "certified": certified,
        "minimum_interval_reserve": minimum_bound,
        "unresolved_interval": unresolved,
        "intervals_checked": checked,
        "uniform_in_N": uniform_in_N,
        "predicted_winding": classical_winding_real_reference(p, lam),
    }


def dimensionless_coordinates(p: LHNParams, lam: float) -> dict[str, float]:
    """Return the dimensionless variables used in the revised theorem."""
    _validate(p)
    kappa = p.gphi + p.G_R + p.G_L
    sigma = p.G_R + p.G_L + 4.0 * p.J**2 / kappa
    return {
        "j": abs(p.J) / kappa,
        "g": p.gphi / kappa,
        "xi": -float(lam) / kappa,
        "s": sigma / kappa,
        "delta": (p.G_R - p.G_L) / sigma,
        "kappa": kappa,
        "sigma": sigma,
    }


def centered_dimensionless_uniform_reserve(j: float, g: float, delta: float) -> float:
    """Uniform all-N reserve at the centered reference point lambda=-Sigma.

    The point is physically realisable only when |delta| s <= 1-g, where
    s=1-g+4j^2. The function returns -inf outside the convergence or physical
    domain.
    """
    j, g, delta = float(abs(j)), float(g), float(delta)
    s = 1.0 - g + 4.0 * j**2
    a = g - 4.0 * j**2
    if not (0.0 <= g < 1.0) or s <= 0.0 or abs(delta) * s > 1.0 - g + 1e-15:
        return -math.inf
    if a <= 4.0 * j:
        return -math.inf
    error = (
        8.0 * j**2 * s / a
        + 128.0 * j**4 / (a**3 * (1.0 - 4.0 * j / a))
    )
    return abs(delta) * s - error


def centered_dimensionless_pointwise_reserve(
    q: float | np.ndarray,
    j: float,
    g: float,
    delta: float,
) -> float | np.ndarray:
    """Dimensionless pointwise reserve at lambda=-Sigma."""
    j, g, delta = float(abs(j)), float(g), float(delta)
    s = 1.0 - g + 4.0 * j**2
    a = g - 4.0 * j**2
    q_arr = np.asarray(q, dtype=float)
    if (
        not (0.0 <= g < 1.0)
        or s <= 0.0
        or abs(delta) * s > 1.0 - g + 1e-15
        or a <= 0.0
    ):
        out = np.full_like(q_arr, -np.inf, dtype=float)
        return float(out) if out.ndim == 0 else out
    sh = np.abs(np.sin(q_arr / 2.0))
    eta = 4.0 * j * sh / a
    modulus = s * np.sqrt(np.cos(q_arr) ** 2 + delta**2 * np.sin(q_arr) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = (
            8.0 * j**2 * s * sh**2 / a
            + 128.0 * j**4 * sh**4 / (a**3 * (1.0 - eta))
        )
    reserve = np.where(eta < 1.0, modulus - error, -np.inf)
    return float(reserve) if reserve.ndim == 0 else reserve
