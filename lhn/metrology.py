"""
Estimation theory for Lindbladian sensors.

The calculations below concern stationary states of time-independent
Liouvillians. They do not include finite preparation time, finite observation
time, detector noise, or output-field measurements.

For a parameter theta:

    L(theta) rho_ss(theta) = 0 ,      Tr rho_ss = 1
    L d_theta rho_ss = -(d_theta L) rho_ss ,   Tr d_theta rho_ss = 0

    F_Q  = 2 sum_{mn} |<m| d_theta rho |n>|^2 / (p_m + p_n)     (quantum Fisher info)
    F_C  = sum_j (d_theta p_j)^2 / p_j                          (site-resolved readout)

F_C is the Fisher information associated with an ideal site-occupation
measurement and therefore lower-bounds F_Q. Equality is model- and
parameter-dependent and is not assumed by the implementation.
"""

import numpy as np

from .models import (LHNParams, hamiltonian, jump_operators, liouvillian)

__all__ = [
    "liouvillian_from_ops",
    "steady_state",
    "d_steady_state",
    "quantum_fisher_information",
    "classical_fisher_information",
    "theta_derivative_superoperator",
    "lhn_sensor_fisher",
    "ep_liouvillian",
    "ep_splitting",
    "ep_sensor_fisher",
]


def liouvillian_from_ops(H, ops):
    """Generic Liouvillian from a Hamiltonian and a list of jump operators."""
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for A in ops:
        AdA = A.conj().T @ A
        Lsup += np.kron(A.conj(), A)
        Lsup -= 0.5 * np.kron(I, AdA)
        Lsup -= 0.5 * np.kron(AdA.T, I)
    return Lsup


def _augment(Lsup, d):
    """Stack the trace functional under L so the singular system becomes solvable."""
    tr_row = np.eye(d, dtype=complex).reshape(-1, order="F").conj()[None, :]
    return np.vstack([Lsup, tr_row])


def steady_state(Lsup, d, rcond=None):
    """Steady-state density matrix (d x d) from the kernel of the Liouvillian."""
    A = _augment(Lsup, d)
    b = np.zeros(A.shape[0], dtype=complex)
    b[-1] = 1.0
    x, *_ = np.linalg.lstsq(A, b, rcond=rcond)
    rho = x.reshape((d, d), order="F")
    rho = 0.5 * (rho + rho.conj().T)          # enforce hermiticity
    return rho / np.trace(rho).real


def d_steady_state(Lsup, dLsup, rho, d, rcond=None):
    """d rho_ss / d theta by solving the constrained linear system."""
    rhs = -(dLsup @ rho.reshape(-1, order="F"))
    A = _augment(Lsup, d)
    b = np.concatenate([rhs, [0.0]])
    x, *_ = np.linalg.lstsq(A, b, rcond=rcond)
    drho = x.reshape((d, d), order="F")
    return 0.5 * (drho + drho.conj().T)


def quantum_fisher_information(rho, drho, eps=1e-12):
    """QFI via the spectral decomposition of rho (SLD form)."""
    p, V = np.linalg.eigh(rho)
    p = np.clip(p, 0.0, None)
    M = V.conj().T @ drho @ V
    F = 0.0
    n = len(p)
    for m in range(n):
        for k in range(n):
            s = p[m] + p[k]
            if s > eps:
                F += 2.0 * abs(M[m, k]) ** 2 / s
    return float(np.real(F))


def classical_fisher_information(rho, drho, eps=1e-14):
    """Fisher information of the site-occupation distribution p_j = rho_jj."""
    p = np.real(np.diag(rho))
    dp = np.real(np.diag(drho))
    mask = p > eps
    return float(np.sum(dp[mask] ** 2 / p[mask]))


def theta_derivative_superoperator(p):
    """d L / d theta for the weak boundary link.

    The link enters through the jump operator A = sqrt(theta) |0><N-1|, and every
    superoperator term built from it is exactly LINEAR in theta. So the derivative
    is just the same construction evaluated at unit rate -- no finite differences,
    no truncation error.
    """
    N = p.N
    I = np.eye(N, dtype=complex)
    A = np.zeros((N, N), dtype=complex)
    A[0, N - 1] = 1.0
    AdA = A.conj().T @ A
    return (np.kron(A.conj(), A)
            - 0.5 * np.kron(I, AdA)
            - 0.5 * np.kron(AdA.T, I))


def lhn_sensor_fisher(p, theta0=1e-3, J_disorder=None, rate_disorder=None):
    """Fisher information of the LHN chain for estimating the boundary link theta.

    The chain is open and theta is a weak return path from the last site to the
    first. Directional transport produces an exponentially nonuniform
    stationary distribution. For the chosen rate parameter and site-occupation
    readout, the one-sided local Fisher information at theta = 0 can grow
    exponentially with system size. This is a model-specific boundary-response
    effect, not a topologically quantized or resource-normalized sensitivity.

    Returns F_Q, F_C, the stationary profile, and the density matrix.
    """
    pp = p.copy(pbc=False, theta=theta0)
    Lsup = liouvillian(pp, J_disorder=J_disorder, rate_disorder=rate_disorder)
    dL = theta_derivative_superoperator(pp)
    rho = steady_state(Lsup, pp.N)
    drho = d_steady_state(Lsup, dL, rho, pp.N)
    prof = np.array(np.real(np.diag(rho)), dtype=float)
    return dict(
        F_Q=quantum_fisher_information(rho, drho),
        F_C=classical_fisher_information(rho, drho),
        profile=prof / prof.sum(),
        rho=rho,
    )


# ----------------------------------------------------------------------------
# Exceptional-point reference sensor: resonantly driven, damped qubit
# ----------------------------------------------------------------------------

def ep_liouvillian(Omega, gamma, delta=0.0):
    """Liouvillian of a driven damped qubit in the rotating frame.

        H = (Omega/2) sigma_x + (delta/2) sigma_z ,   jump = sqrt(gamma) sigma_-

    On resonance the Bloch eigenvalues are -gamma/2 and
    -3 gamma/4 +- sqrt((gamma/4)^2 - Omega^2), giving a genuine LIOUVILLIAN
    exceptional point at Omega = gamma/4. delta is the parameter to be estimated.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1| lowering
    H = 0.5 * Omega * sx + 0.5 * delta * sz
    return liouvillian_from_ops(H, [np.sqrt(gamma) * sm])


def ep_splitting(Omega, gamma, delta):
    """Splitting of the two Liouvillian eigenvalues that collide at the EP.

    Taken as the minimum pairwise separation among the non-zero eigenvalues.
    This identifies the coalescing pair in the local neighbourhood of the
    exceptional point used in the perturbative calculations. Far from the
    exceptional point, another pair can become closer; figures that display a
    wide drive range therefore use the analytic coalescing-pair expression.
    """
    ev = np.linalg.eigvals(ep_liouvillian(Omega, gamma, delta))
    ev = ev[np.abs(ev) > 1e-9]        # drop the zero (steady-state) eigenvalue
    D = np.abs(ev[:, None] - ev[None, :]) + np.eye(len(ev)) * 1e18
    return float(D.min())


def ep_sensor_fisher(Omega, gamma, delta, h=1e-6):
    """Fisher information of the driven-qubit steady state for the detuning delta.

    dL/d delta is obtained by central differences on the (analytically linear in
    delta) Liouvillian, so the step size is not a source of error.
    """
    L0 = ep_liouvillian(Omega, gamma, delta)
    dL = (ep_liouvillian(Omega, gamma, delta + h)
          - ep_liouvillian(Omega, gamma, delta - h)) / (2.0 * h)
    rho = steady_state(L0, 2)
    drho = d_steady_state(L0, dL, rho, 2)
    return dict(F_Q=quantum_fisher_information(rho, drho),
                F_C=classical_fisher_information(rho, drho),
                rho=rho)


# ----------------------------------------------------------------------------
# Realistic driven-qubit Liouvillian with independent T1 and T2 (for ESR-STM)
# ----------------------------------------------------------------------------

def driven_qubit_liouvillian(Omega, gamma1, gamma_z, delta=0.0):
    """Driven, damped, dephased qubit.

        H = (Omega/2) sigma_x + (delta/2) sigma_z
        jumps: sqrt(gamma1) sigma_-  (relaxation),  sqrt(gamma_z) sigma_z (dephasing)

    All rates in MHz. Relates to the measured times as
        1/T1 = gamma1 ,      1/T2 = gamma1/2 + 2 gamma_z .
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    H = 0.5 * Omega * sx + 0.5 * delta * sz
    return liouvillian_from_ops(H, [np.sqrt(gamma1) * sm, np.sqrt(gamma_z) * sz])


def gamma_z_from_times(T1_us, T2_us):
    """Pure-dephasing rate gamma_z from compatible Markovian T1 and T2 data.

    The relation 1/T2 = 1/(2 T1) + 2 gamma_z requires gamma_z >= 0. Values of
    T1 and T2 obtained under different experimental conditions or refocusing
    protocols need not satisfy this time-independent Markovian relation.
    """
    g1 = 1.0 / T1_us
    gz = 0.5 * (1.0 / T2_us - 0.5 * g1)
    if gz < -1e-12:
        raise ValueError("T1 and T2 are incompatible with non-negative Markovian pure dephasing")
    return max(0.0, float(gz))


def liouvillian_gap_pair(L):
    """Minimum separation between non-zero Liouvillian eigenvalues."""
    ev = np.linalg.eigvals(L)
    ev = ev[np.abs(ev) > 1e-9 * max(1.0, np.abs(ev).max())]
    D = np.abs(ev[:, None] - ev[None, :]) + np.eye(len(ev)) * 1e18
    return float(D.min())


def petermann_factor(L):
    """Largest eigenvector condition number of a matrix.

    Diverges at an exceptional point (eigenvectors coalesce) and stays O(1) at an
    ordinary degeneracy, so it is the correct EP diagnostic. Eigenvalue proximity
    alone is NOT: the driven qubit has a trivial degeneracy at Omega = 0 where the
    decoupled sigma_x and sigma_y modes share a decay rate without any coalescence.
    """
    ev, R = np.linalg.eig(L)
    Linv = np.linalg.inv(R)
    K = np.array([np.linalg.norm(R[:, i]) * np.linalg.norm(Linv[i, :])
                  for i in range(len(ev))])
    return float(K.max())


def ep_drive_strength(T1_us, T2_us):
    """Analytic Liouvillian EP of the resonantly driven qubit.

    The Bloch equations decouple <sigma_x> from the (<sigma_y>, <sigma_z>) block

        M = [[-1/T2, -Omega], [Omega, -1/T1]]

    whose eigenvalues -(1/T1 + 1/T2)/2 +- sqrt( ((1/T2 - 1/T1)/2)^2 - Omega^2 )
    coalesce at

        Omega_EP = |1/T2 - 1/T1| / 2      (angular units, rad/us)

    Returns (Omega_EP in rad/us, corresponding Rabi FREQUENCY Omega_EP/2pi in MHz).
    """
    om = 0.5 * abs(1.0 / T2_us - 1.0 / T1_us)
    return om, om / (2 * np.pi)
