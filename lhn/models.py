"""
Liouvillian Hatano-Nelson (LHN) chain -- single-excitation sector.

Model
-----
N-site ring/chain carrying one excitation. Basis |j>, j = 0..N-1.

    H          = J * sum_j ( |j+1><j| + |j><j+1| )          (coherent, reciprocal)
    R_j        = sqrt(G_R) |j+1><j|                          (incoherent hop right)
    L_j        = sqrt(G_L) |j><j+1|                          (incoherent hop left)
    D_j        = sqrt(gphi) |j><j|                           (dephasing)

The Lindbladian is number-conserving inside the one-excitation manifold, so the
density matrix rho is an N x N matrix and the Liouvillian is N^2 x N^2.

Key structural facts (all proved analytically in docs/derivation.md):

1.  The POSTSELECTED (no-jump) generator is
        H_eff = H - (i/2) sum_mu A_mu^dag A_mu
              = H - (i/2)(G_R + G_L + gphi) * Identity
    i.e. a reciprocal Hermitian hopping matrix plus a constant imaginary shift.
    Its Bloch spectrum is a line segment traced twice, and therefore has zero
    point-gap winding. Open boundaries may still produce non-extensive edge
    shifts because boundary jump channels are missing.

2.  The full Liouvillian can exhibit nonzero point-gap winding when G_R != G_L.
    In this construction, the non-reciprocal momentum dependence responsible
    for the winding enters through the recycling term sum_mu A rho A^dag, which
    is omitted in no-jump postselection.

This contrast is one of the structural results tested by the package.

Vectorisation convention: column-stacking, vec(A rho B) = (B.T kron A) vec(rho).
"""

import numpy as np

__all__ = [
    "LHNParams",
    "hamiltonian",
    "jump_operators",
    "liouvillian",
    "postselected_hamiltonian",
    "liouvillian_block_q",
    "classical_markov_generator",
    "effective_classical_generator",
]


class LHNParams:
    """Parameter container for the Liouvillian Hatano-Nelson chain.

    Parameters
    ----------
    N : int
        Number of sites.
    J : float
        Coherent (reciprocal) hopping amplitude.
    G_R, G_L : float
        Incoherent hopping rates to the right / left. G_R != G_L breaks
        reciprocity and is what generates the point-gap topology.
    gphi : float
        Local dephasing rate. Drives the quantum -> classical crossover.
    pbc : bool
        Periodic (True) or open (False) boundary conditions.
    theta : float
        Optional weak return link from N-1 to 0 for open-boundary diagnostics.
    """

    def __init__(self, N=12, J=1.0, G_R=1.0, G_L=0.4, gphi=0.0, pbc=True, theta=0.0):
        self.N = int(N)
        self.J = float(J)
        self.G_R = float(G_R)
        self.G_L = float(G_L)
        self.gphi = float(gphi)
        self.pbc = bool(pbc)
        self.theta = float(theta)
        if self.N < 3:
            raise ValueError("N must be at least 3")
        vals = {"J": self.J, "G_R": self.G_R, "G_L": self.G_L,
                "gphi": self.gphi, "theta": self.theta}
        if not all(np.isfinite(v) for v in vals.values()):
            raise ValueError("all model parameters must be finite")
        if self.G_R < 0 or self.G_L < 0 or self.gphi < 0 or self.theta < 0:
            raise ValueError("Lindblad rates G_R, G_L, gphi, and theta must be non-negative")

    def copy(self, **kw):
        d = dict(N=self.N, J=self.J, G_R=self.G_R, G_L=self.G_L,
                 gphi=self.gphi, pbc=self.pbc, theta=self.theta)
        d.update(kw)
        return LHNParams(**d)

    def __repr__(self):
        return (f"LHNParams(N={self.N}, J={self.J}, G_R={self.G_R}, G_L={self.G_L}, "
                f"gphi={self.gphi}, pbc={self.pbc}, theta={self.theta})")


def _bonds(N, pbc):
    """Bond list (j, j+1). Includes the wrap-around bond iff pbc."""
    b = [(j, j + 1) for j in range(N - 1)]
    if pbc and N > 2:
        b.append((N - 1, 0))
    return b


def hamiltonian(p, J_disorder=None):
    """Coherent Hermitian hopping matrix (N x N).

    J_disorder : optional array of per-bond hopping amplitudes overriding p.J.
    """
    N = p.N
    H = np.zeros((N, N), dtype=complex)
    bonds = _bonds(N, p.pbc)
    if J_disorder is not None:
        J_disorder = np.asarray(J_disorder, dtype=float)
        if J_disorder.shape != (len(bonds),) or not np.all(np.isfinite(J_disorder)):
            raise ValueError("J_disorder must be a finite one-dimensional array with one entry per bond")
    for b, (a, c) in enumerate(bonds):
        Jb = p.J if J_disorder is None else J_disorder[b]
        H[c, a] += Jb
        H[a, c] += Jb
    return H


def jump_operators(p, rate_disorder=None):
    """List of Lindblad jump operators (each N x N).

    rate_disorder : optional dict with keys 'R','L' giving per-bond multiplicative
        disorder factors, used for the robustness study.
    """
    N = p.N
    ops = []
    bonds = _bonds(N, p.pbc)
    if rate_disorder is not None:
        if set(rate_disorder) != {"R", "L"}:
            raise ValueError("rate_disorder must contain exactly the keys 'R' and 'L'")
        rate_disorder = {key: np.asarray(rate_disorder[key], dtype=float)
                         for key in ("R", "L")}
        for key in ("R", "L"):
            arr = rate_disorder[key]
            if arr.shape != (len(bonds),) or not np.all(np.isfinite(arr)):
                raise ValueError(f"rate_disorder['{key}'] must contain one finite factor per bond")
            if np.any(arr < 0):
                raise ValueError("multiplicative rate-disorder factors must be non-negative")

    for b, (a, c) in enumerate(bonds):
        fR = 1.0 if rate_disorder is None else rate_disorder["R"][b]
        fL = 1.0 if rate_disorder is None else rate_disorder["L"][b]
        if p.G_R * fR > 0:
            A = np.zeros((N, N), dtype=complex)
            A[c, a] = np.sqrt(p.G_R * fR)     # |a+1><a| : hop right
            ops.append(A)
        if p.G_L * fL > 0:
            A = np.zeros((N, N), dtype=complex)
            A[a, c] = np.sqrt(p.G_L * fL)     # |a><a+1| : hop left
            ops.append(A)

    # optional weak return link; only for open chains
    if (not p.pbc) and p.theta != 0.0 and N > 2:
        A = np.zeros((N, N), dtype=complex)
        A[0, N - 1] = np.sqrt(p.theta)        # |0><N-1| : return path
        ops.append(A)

    if p.gphi > 0:
        for j in range(N):
            A = np.zeros((N, N), dtype=complex)
            A[j, j] = np.sqrt(p.gphi)
            ops.append(A)

    return ops


def liouvillian(p, J_disorder=None, rate_disorder=None):
    """Full Liouvillian superoperator, shape (N^2, N^2), column-stacking vec."""
    N = p.N
    I = np.eye(N, dtype=complex)
    H = hamiltonian(p, J_disorder=J_disorder)
    ops = jump_operators(p, rate_disorder=rate_disorder)

    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for A in ops:
        AdA = A.conj().T @ A
        Lsup += np.kron(A.conj(), A)
        Lsup -= 0.5 * np.kron(I, AdA)
        Lsup -= 0.5 * np.kron(AdA.T, I)
    return Lsup


def postselected_hamiltonian(p, J_disorder=None, rate_disorder=None):
    """No-jump (postselected) non-Hermitian Hamiltonian H - (i/2) sum A^dag A."""
    H = hamiltonian(p, J_disorder=J_disorder)
    ops = jump_operators(p, rate_disorder=rate_disorder)
    G = np.zeros_like(H)
    for A in ops:
        G += A.conj().T @ A
    return H - 0.5j * G


def liouvillian_block_q(q, p):
    """Momentum-resolved Liouvillian block L_q (N x N), acting on the relative
    coordinate r = j - l (mod N), at centre-of-mass momentum q.

    Derived analytically:
        [L_q]_{r+1,r} = -i J (1 - e^{+iq})
        [L_q]_{r-1,r} = -i J (1 - e^{-iq})
        [L_q]_{r,r}   = -(G_R + G_L + gphi)                     for r != 0
        [L_q]_{0,0}   = -(G_R + G_L) + G_R e^{-iq} + G_L e^{+iq}

    Requires pbc=True. The union of spec(L_q) over q = 2*pi*n/N reproduces the
    full PBC Liouvillian spectrum exactly (checked in tests).
    """
    if not p.pbc:
        raise ValueError("liouvillian_block_q requires periodic boundary conditions")
    N = p.N
    Lq = np.zeros((N, N), dtype=complex)
    up = -1j * p.J * (1.0 - np.exp(1j * q))     # r -> r+1
    dn = -1j * p.J * (1.0 - np.exp(-1j * q))    # r -> r-1
    for r in range(N):
        Lq[(r + 1) % N, r] += up
        Lq[(r - 1) % N, r] += dn
        Lq[r, r] += -(p.G_R + p.G_L + p.gphi)
    Lq[0, 0] += p.gphi + p.G_R * np.exp(-1j * q) + p.G_L * np.exp(1j * q)
    return Lq


def classical_markov_generator(N, rate_R, rate_L, pbc=True, theta=0.0):
    """Classical biased-random-walk generator W (N x N), acting on populations.

    dp/dt = W p, with columns summing to zero. In the strong-dephasing,
    low-frequency limit this is the effective generator obtained after
    eliminating Liouvillian coherences.
    """
    N = int(N)
    rate_R = float(rate_R)
    rate_L = float(rate_L)
    theta = float(theta)
    if N < 3:
        raise ValueError("N must be at least 3")
    if min(rate_R, rate_L, theta) < 0 or not np.all(np.isfinite([rate_R, rate_L, theta])):
        raise ValueError("classical transition rates must be finite and non-negative")
    W = np.zeros((N, N), dtype=complex)
    bonds = [(j, j + 1) for j in range(N - 1)]
    if pbc and N > 2:
        bonds.append((N - 1, 0))
    for (a, c) in bonds:
        W[c, a] += rate_R
        W[a, a] -= rate_R
        W[a, c] += rate_L
        W[c, c] -= rate_L
    if (not pbc) and theta != 0.0 and N > 2:
        W[0, N - 1] += theta
        W[N - 1, N - 1] -= theta
    return W


def coherence_decay_rate(p):
    """Leading decay rate kappa of the r = +-1 coherence sectors.

        kappa = gphi + G_R + G_L

    This is the correct denominator for adiabatic elimination. Using gphi alone
    is valid only in the strict gphi >> G_R + G_L limit and is quantitatively
    inaccurate otherwise -- see docs/derivation.md section 4.
    """
    return p.gphi + p.G_R + p.G_L


def symmetric_rate_D(p):
    """Symmetric (diffusive) rate generated by coherent hopping under dephasing.

        D = 2 J^2 / kappa ,    kappa = gphi + G_R + G_L

    Reduces to the often-quoted 2 J^2 / gphi only when gphi >> G_R + G_L.
    """
    k = coherence_decay_rate(p)
    if k <= 0:
        raise ValueError("adiabatic elimination requires gphi + G_R + G_L > 0")
    return 2.0 * p.J ** 2 / k


def effective_classical_generator(p):
    """Adiabatic-elimination prediction for the strong-dephasing regime.

    Second-order elimination of the coherences (r = +-1) gives a classical biased
    walk with rates

        rate_R = G_R + D ,   rate_L = G_L + D ,   D = 2 J^2 / (gphi + G_R + G_L)

    Coherent hopping is converted into a SYMMETRIC incoherent rate, which dilutes
    the rate RATIO but leaves the rate DIFFERENCE G_R - G_L unchanged at this
    order. Persistence of the winding additionally requires the chosen point gap
    to stay open; it does not follow from a nonzero bias alone.
    """
    D = symmetric_rate_D(p)
    return classical_markov_generator(p.N, p.G_R + D, p.G_L + D,
                                      pbc=p.pbc, theta=p.theta)


def schur_r0(q, p):
    """Zero-frequency Schur complement of the population coordinate r = 0.

    The algebraic complement

        L_eff(q; z=0) = L_00 - L_0r (L_rr)^{-1} L_r0

    eliminates all coherence coordinates at zero spectral frequency. It is
    exact as a zero-frequency Schur complement, but it is not an exact
    frequency-independent population generator at finite coherent hopping.
    The exact spectral reduction contains the resolvent (z - L_rr)^{-1}.

    As gphi grows, this quantity approaches the classical Markov symbol with
    rates G_R + D and G_L + D, where
    D = 2 J^2 / (gphi + G_R + G_L).
    """
    Lq = liouvillian_block_q(q, p)
    idx = np.arange(p.N)
    rest = idx[idx != 0]
    A00 = Lq[0, 0]
    A0r = Lq[0, rest]
    Ar0 = Lq[rest, 0]
    Arr = Lq[np.ix_(rest, rest)]
    return complex(A00 - A0r @ np.linalg.solve(Arr, Ar0))


def disordered_markov_generator(N, rate_R, rate_L, W_dis, rng=None, phi=0.0,
                                  base_R=None, base_L=None):
    """Classical biased walk with multiplicative bond disorder and a twist.

    A fixed disorder realization is specified by arrays ``base_R`` and
    ``base_L`` whose entries lie in [-1, 1]:

        r_R,b = rate_R * (1 + W_dis * base_R[b]),
        r_L,b = rate_L * (1 + W_dis * base_L[b]).

    Supplying the same base arrays while varying ``W_dis`` defines a continuous
    disorder path. If the arrays are omitted, they are drawn once from ``rng``.
    The function rejects non-positive rates instead of silently flooring them.
    The wrap-around bond carries the boundary twist used for the real-space
    winding number.
    """
    if W_dis < 0 or not np.isfinite(W_dis):
        raise ValueError("W_dis must be finite and non-negative")
    if base_R is None or base_L is None:
        if rng is None:
            raise ValueError("provide rng or fixed base_R/base_L arrays")
        base_R = rng.uniform(-1.0, 1.0, N)
        base_L = rng.uniform(-1.0, 1.0, N)
    base_R = np.asarray(base_R, dtype=float)
    base_L = np.asarray(base_L, dtype=float)
    if base_R.shape != (N,) or base_L.shape != (N,):
        raise ValueError("base_R and base_L must each have shape (N,)")
    rates_R = rate_R * (1.0 + W_dis * base_R)
    rates_L = rate_L * (1.0 + W_dis * base_L)
    if np.any(rates_R <= 0.0) or np.any(rates_L <= 0.0):
        raise ValueError("disorder realization contains a non-positive rate")

    Wm = np.zeros((N, N), dtype=complex)
    for b in range(N):
        a, c = b, (b + 1) % N
        phase = np.exp(-1j * phi) if b == N - 1 else 1.0
        Wm[c, a] += rates_R[b] * phase
        Wm[a, a] -= rates_R[b]
        Wm[a, c] += rates_L[b] / phase
        Wm[c, c] -= rates_L[b]
    return Wm
