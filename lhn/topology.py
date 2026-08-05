"""
Spectral topology of dissipative generators.

The single invariant used throughout is the point-gap winding number

    w(lam) = (1 / 2 pi i) * closed-integral dq  d/dq  ln det[ M(q) - lam ]
           = (1 / 2 pi) * (total change in arg det[M(q) - lam] as q: 0 -> 2 pi)

It is computed by exactly the same routine for

    * the quantum Liouvillian superoperator   L_q      (models.liouvillian_block_q)
    * the classical Markov generator          W(k)     (bloch_classical)
    * the postselected non-Hermitian Hamiltonian H_eff(k)  (bloch_postselected)

which supplies a common spectral diagnostic for the three generator families.
The existence and value of a winding still depend on the chosen point gap.
"""

import numpy as np

__all__ = [
    "winding_number",
    "winding_map",
    "point_gap_margin",
    "bloch_classical",
    "bloch_postselected",
    "pbc_spectrum_from_blocks",
    "spectrum",
    "skin_profile",
    "mode_centre_of_mass",
    "steady_state_profile",
]


def winding_number(matrix_fn, lam, n_q=512, return_curve=False):
    """Point-gap winding number of the family matrix_fn(q) about the point lam.

    Parameters
    ----------
    matrix_fn : callable
        q (float) -> square complex ndarray.
    lam : complex
        Reference point in the complex plane.
    n_q : int
        Number of quadrature points on the momentum circle.

    Returns
    -------
    w : int
        Winding number (rounded; the raw value is returned too if requested).
    """
    qs = np.linspace(0.0, 2.0 * np.pi, n_q, endpoint=False)
    phases = np.empty(n_q)
    logabs = np.empty(n_q)
    for i, q in enumerate(qs):
        M = matrix_fn(q)
        sign, la = np.linalg.slogdet(M - lam * np.eye(M.shape[0]))
        phases[i] = np.angle(sign)
        logabs[i] = la
    # close the loop, unwrap, measure total phase accumulation
    ph = np.unwrap(np.concatenate([phases, phases[:1]]))
    raw = (ph[-1] - ph[0]) / (2.0 * np.pi)
    w = int(np.rint(raw))
    if return_curve:
        return w, raw, qs, phases, logabs
    return w


def winding_map(matrix_fn, re_lim, im_lim, n_re=90, n_im=90, n_q=192):
    """Winding number on a grid of reference points -> reveals the point gap.

    Returns (RE, IM, W) suitable for pcolormesh.
    """
    res = np.linspace(*re_lim, n_re)
    ims = np.linspace(*im_lim, n_im)
    RE, IM = np.meshgrid(res, ims)
    W = np.zeros_like(RE)

    qs = np.linspace(0.0, 2.0 * np.pi, n_q, endpoint=False)
    mats = [matrix_fn(q) for q in qs]
    n = mats[0].shape[0]
    eig = np.array([np.linalg.eigvals(M) for M in mats])   # (n_q, n)

    # det(M - lam) = prod_i (e_i - lam); winding = sum over the eigenvalue branches
    # of the winding of each branch. Summing arg over all eigenvalues is
    # branch-ordering independent, which makes this both fast and robust.
    for a in range(n_re):
        for b in range(n_im):
            lam = res[a] + 1j * ims[b]
            ang = np.angle(eig - lam).sum(axis=1)
            ang = np.unwrap(np.concatenate([ang, ang[:1]]))
            W[b, a] = np.rint((ang[-1] - ang[0]) / (2.0 * np.pi))
    return RE, IM, W



def point_gap_margin(matrix_fn, lam, n_q=512):
    """Minimum singular value of ``M(q) - lam I`` on the momentum circle.

    This is the appropriate finite-grid diagnostic for an open point gap of a
    non-normal matrix. Eigenvalue distance alone can overestimate spectral
    stability and is therefore not used here.
    """
    qs = np.linspace(0.0, 2.0 * np.pi, n_q, endpoint=False)
    margin = np.inf
    for q in qs:
        M = matrix_fn(q)
        A = M - lam * np.eye(M.shape[0], dtype=complex)
        smin = np.linalg.svd(A, compute_uv=False)[-1]
        margin = min(margin, float(smin))
    return margin

def bloch_classical(rate_R, rate_L):
    """Bloch 'matrix' (1x1) of the classical biased random walk generator."""
    def fn(k):
        val = rate_R * np.exp(-1j * k) + rate_L * np.exp(1j * k) - (rate_R + rate_L)
        return np.array([[val]], dtype=complex)
    return fn


def bloch_postselected(p):
    """Bloch matrix of the postselected (no-jump) Hamiltonian.

    H_eff(k) = 2 J cos k - (i/2)(G_R + G_L + gphi)

    A straight horizontal segment traced twice in the complex plane. For any
    reference point outside the spectrum, its point-gap winding vanishes. In
    this model the no-jump dynamics is therefore point-gap topologically trivial.
    """
    shift = -0.5j * (p.G_R + p.G_L + p.gphi)

    def fn(k):
        return np.array([[2.0 * p.J * np.cos(k) + shift]], dtype=complex)
    return fn


def pbc_spectrum_from_blocks(p, block_fn):
    """Union of spec(L_q) over the N allowed momenta -- must equal spec(L_PBC)."""
    vals = []
    for n in range(p.N):
        q = 2.0 * np.pi * n / p.N
        vals.append(np.linalg.eigvals(block_fn(q, p)))
    return np.concatenate(vals)


def spectrum(M):
    """Eigenvalues sorted by real part (descending: slowest-decaying first)."""
    ev = np.linalg.eigvals(M)
    return ev[np.argsort(-ev.real)]


def mode_centre_of_mass(M, N):
    """Centre of mass (in site index) of each Liouvillian eigenmode.

    Each eigenvector is vec(sigma) for an N x N operator sigma; the weight of the
    mode on site j is taken as sum_l |sigma_{jl}|^2. A skin effect shows up as the
    bulk of the modes piling up near one edge.
    """
    _, V = np.linalg.eig(M)
    coms = np.empty(V.shape[1])
    sites = np.arange(N)
    for i in range(V.shape[1]):
        sig = V[:, i].reshape((N, N), order="F")
        wgt = (np.abs(sig) ** 2).sum(axis=1)
        s = wgt.sum()
        coms[i] = (sites * wgt).sum() / s if s > 0 else np.nan
    return coms


def skin_profile(M, N):
    """Site-resolved weight summed over all eigenmodes of the superoperator."""
    _, V = np.linalg.eig(M)
    prof = np.zeros(N)
    for i in range(V.shape[1]):
        sig = V[:, i].reshape((N, N), order="F")
        w = (np.abs(sig) ** 2).sum(axis=1)
        s = w.sum()
        if s > 0:
            prof += w / s
    return prof / prof.sum()


def steady_state_profile(rho):
    """Population profile p_j = rho_{jj} of a density matrix."""
    d = np.array(np.real(np.diag(rho)), dtype=float)
    return d / d.sum()


def real_space_winding(builder, lam, n_phi=256):
    """Winding number of det[M(phi) - lam] under a boundary twist phi.

    This is the definition that survives disorder: it needs no translation
    invariance, only a ring geometry with one twisted bond. For a clean chain it
    reproduces the Bloch winding number exactly.
    """
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    ang = np.empty(n_phi)
    for i, ph in enumerate(phis):
        M = builder(ph)
        sign, _ = np.linalg.slogdet(M - lam * np.eye(M.shape[0]))
        ang[i] = np.angle(sign)
    a = np.unwrap(np.concatenate([ang, ang[:1]]))
    return int(np.rint((a[-1] - a[0]) / (2.0 * np.pi)))
