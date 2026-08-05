"""Regression and consistency checks for the numerical package.

Passing this suite verifies the implemented identities and selected parameter
regimes; it is not a substitute for independent physical review.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.optimize import linear_sum_assignment
from lhn import *
from lhn.models import liouvillian_block_q

ok = lambda name, cond, extra="": print(
    f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))
results = []


def check(name, cond, extra=""):
    ok(name, cond, extra)
    results.append(bool(cond))


# --------------------------------------------------------------------------
print("\n--- 1. Structural checks on the Liouvillian ---")
p = LHNParams(N=8, J=1.0, G_R=1.0, G_L=0.4, gphi=0.3, pbc=True)
L = liouvillian(p)

# trace preservation: vec(I)^dag L = 0
tr_row = np.eye(p.N, dtype=complex).reshape(-1, order="F").conj()
check("Liouvillian is trace-annihilating", np.abs(tr_row @ L).max() < 1e-10,
      f"max |vec(I)^dag L| = {np.abs(tr_row @ L).max():.2e}")

# there is exactly one zero eigenvalue (unique steady state)
ev = np.linalg.eigvals(L)
n_zero = np.sum(np.abs(ev) < 1e-9)
check("unique steady state (single zero mode)", n_zero == 1, f"n_zero = {n_zero}")

# --------------------------------------------------------------------------
print("\n--- 2. Momentum blocks reproduce the full PBC spectrum ---")
# compare as SETS via optimal matching -- naive sorting is unstable when
# eigenvalues share a real part up to floating-point noise
full = np.linalg.eigvals(L)
blocks = pbc_spectrum_from_blocks(p, liouvillian_block_q)
Cst = np.abs(full[:, None] - blocks[None, :])
ri, ci = linear_sum_assignment(Cst)
err = Cst[ri, ci].max()
check("spec(L_PBC) == union_q spec(L_q)", err < 1e-9, f"max abs error = {err:.2e}")

# --------------------------------------------------------------------------
print("\n--- 3. Winding numbers ---")
# classical biased walk
w_bias = winding_number(bloch_classical(1.0, 0.4), lam=-0.7 + 0.0j)
w_unbias = winding_number(bloch_classical(1.0, 1.0), lam=-2.0 + 0.0j)
check("biased classical walk has |w| = 1", abs(w_bias) == 1, f"w = {w_bias}")
check("unbiased classical walk has w = 0", w_unbias == 0, f"w = {w_unbias}")

# postselected Hamiltonian: BULK line spectrum -> always zero winding
w_ps = winding_number(bloch_postselected(p), lam=-0.85j)
check("postselected H_eff bulk has w = 0", w_ps == 0, f"w = {w_ps}")

# ...but OBC and PBC spectra of H_eff are NOT identical: open edges are missing
# jump channels, so the imaginary onsite terms are non-uniform. The correct
# claim is "zero bulk winding, no extensive skin effect", not "identical spectra".
from lhn.models import postselected_hamiltonian
ev_pb = np.linalg.eigvals(postselected_hamiltonian(p))
ev_ob = np.linalg.eigvals(postselected_hamiltonian(p.copy(pbc=False)))
im_spread_pb = np.ptp(ev_pb.imag)
im_spread_ob = np.ptp(ev_ob.imag)
check("H_eff: PBC imaginary parts are uniform", im_spread_pb < 1e-9,
      f"spread = {im_spread_pb:.2e}")
check("H_eff: OBC shows edge shifts (NOT identical to PBC)", im_spread_ob > 1e-3,
      f"spread = {im_spread_ob:.4f}")

# Full Liouvillian: use the winding MAP rather than a single reference point,
# because a hand-picked lambda may land on the spectrum (where w is undefined).
_, _, Wnr = winding_map(lambda q: liouvillian_block_q(q, p),
                        (-4.0, 0.5), (-4.0, 4.0), n_re=60, n_im=60, n_q=128)
frac_nr = np.mean(np.abs(Wnr) >= 1)
check("non-reciprocal Liouvillian has a point gap", frac_nr > 0.01,
      f"fraction of plane with |w|>=1 = {frac_nr:.3f}")

p_rec = p.copy(G_R=0.7, G_L=0.7)
_, _, Wr = winding_map(lambda q: liouvillian_block_q(q, p_rec),
                       (-4.0, 0.5), (-4.0, 4.0), n_re=60, n_im=60, n_q=128)
frac_r = np.mean(np.abs(Wr) >= 1)
check("reciprocal Liouvillian has w = 0 everywhere", frac_r == 0.0,
      f"fraction of plane with |w|>=1 = {frac_r:.3f}")

# real-space (twist) winding must agree with the Bloch winding for a clean chain
from lhn.models import disordered_markov_generator
from lhn.topology import real_space_winding
rng0 = np.random.default_rng(0)
uR0, uL0 = rng0.uniform(-1, 1, 20), rng0.uniform(-1, 1, 20)
w_rs = real_space_winding(
    lambda ph: disordered_markov_generator(20, 1.0, 0.4, 0.0, phi=ph,
                                           base_R=uR0, base_L=uL0),
    lam=-1.4 + 0.0j)
w_bl = winding_number(bloch_classical(1.0, 0.4), lam=-1.4 + 0.0j)
check("real-space twist winding == Bloch winding", w_rs == w_bl,
      f"twist {w_rs} vs Bloch {w_bl}")

# --------------------------------------------------------------------------
print("\n--- 4. Strong-dephasing reduction to a classical Markov generator ---")
# The correct denominator is kappa = gphi + G_R + G_L, NOT gphi alone.
# These tolerances are tight enough to FAIL for D = 2 J^2 / gphi -- that is the
# point. The original loose test (5/gphi) did not reject the superseded formula.
from lhn.models import schur_r0, symmetric_rate_D, coherence_decay_rate

def _pred(q, p, D):
    return (-(p.G_R + p.G_L) + p.G_R * np.exp(-1j * q) + p.G_L * np.exp(1j * q)
            - 2.0 * D * (1.0 - np.cos(q)))

for gphi, tol in [(20.0, 4e-3), (100.0, 4e-5), (500.0, 4e-7)]:
    pg = LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.4, gphi=gphi, pbc=True)
    D_ok = symmetric_rate_D(pg)
    D_bad = 2.0 * pg.J ** 2 / gphi                      # the superseded formula
    e_ok = e_bad = 0.0
    for q in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        ex = schur_r0(q, pg)
        e_ok = max(e_ok, abs(ex - _pred(q, pg, D_ok)))
        e_bad = max(e_bad, abs(ex - _pred(q, pg, D_bad)))
    rel_ok = e_ok / (pg.G_R + pg.G_L)
    check(f"adiabatic elimination, kappa = gphi+GR+GL, gphi={gphi:g}", rel_ok < tol,
          f"rel. error = {rel_ok:.2e} < {tol:.0e}")
    check(f"...and this tolerance rejects D=2J^2/gphi at gphi={gphi:g}",
          e_bad / (pg.G_R + pg.G_L) > tol,
          f"superseded formula rel. error = {e_bad/(pg.G_R+pg.G_L):.2e}")

# kappa must reduce to gphi only in the strict limit
p_lim = LHNParams(N=8, J=1.0, G_R=1e-6, G_L=1e-6, gphi=50.0)
check("D -> 2J^2/gphi when G_R+G_L << gphi",
      abs(symmetric_rate_D(p_lim) - 2.0 / 50.0) < 1e-6,
      f"D = {symmetric_rate_D(p_lim):.8f} vs 2J^2/gphi = {2.0/50.0:.8f}")

# The crossover claim concerns the FULL Liouvillian at one fixed reference point.
# The zero-frequency scalar Schur complement is not a substitute for det[L_q-lambda].
from lhn.topology import point_gap_margin
LAM_CROSS = -0.4 + 0.0j
ws_full, ws_eff, margins = [], [], []
for gphi in [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    pg = LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.35, gphi=gphi, pbc=True)
    full_fn = lambda q, pg=pg: liouvillian_block_q(q, pg)
    D = symmetric_rate_D(pg)
    eff_fn = bloch_classical(pg.G_R + D, pg.G_L + D)
    ws_full.append(winding_number(full_fn, lam=LAM_CROSS, n_q=768))
    ws_eff.append(winding_number(eff_fn, lam=LAM_CROSS, n_q=768))
    margins.append(point_gap_margin(full_fn, lam=LAM_CROSS, n_q=512))
check("full-Liouvillian winding is constant at one fixed reference point",
      ws_full == [-1] * len(ws_full), f"w = {ws_full}")
check("effective Markov winding agrees in the controlled strong-dephasing regime",
      ws_eff[3:] == ws_full[3:], f"full {ws_full}, effective {ws_eff}")
check("the fixed full-Liouvillian point gap remains open",
      min(margins) > 0.15, f"minimum singular-value margin = {min(margins):.4f}")

# --------------------------------------------------------------------------
print("\n--- 5. Exact finite-size reduction and analytical winding certificate ---")
from lhn.analytical import (
    analytical_error_bound,
    classical_point_gap_margin,
    coherence_eigenvalues,
    effective_markov_symbol,
    exact_characteristic_determinant,
    exact_schur_scalar,
    winding_certificate,
)

rng_a = np.random.default_rng(20260805)
max_det_rel = 0.0
max_schur_rel = 0.0
max_coh_err = 0.0
for N in (4, 5, 7, 10, 14):
    for _ in range(8):
        pa = LHNParams(
            N=N,
            J=float(rng_a.uniform(0.1, 1.3)),
            G_R=float(rng_a.uniform(0.4, 1.5)),
            G_L=float(rng_a.uniform(0.1, 0.8)),
            gphi=float(rng_a.uniform(0.2, 15.0)),
            pbc=True,
        )
        lam = float(rng_a.uniform(-0.7, -0.05))
        q = float(rng_a.uniform(0.0, 2.0 * np.pi))
        M = liouvillian_block_q(q, pa) - lam * np.eye(N)
        direct_det = np.linalg.det(M)
        formula_det = exact_characteristic_determinant(q, pa, lam)
        max_det_rel = max(max_det_rel,
                          abs(direct_det - formula_det) / max(1.0, abs(direct_det)))

        Dblock = M[1:, 1:]
        direct_schur = M[0, 0] - (M[0:1, 1:] @ np.linalg.solve(Dblock, M[1:, 0:1]))[0, 0]
        formula_schur = exact_schur_scalar(q, pa, lam)
        max_schur_rel = max(max_schur_rel,
                            abs(direct_schur - formula_schur) / max(1.0, abs(direct_schur)))

        direct_coh = np.linalg.eigvals(Dblock)
        formula_coh = coherence_eigenvalues(q, pa, lam)
        cost = np.abs(direct_coh[:, None] - formula_coh[None, :])
        rr, cc = linear_sum_assignment(cost)
        max_coh_err = max(max_coh_err, float(cost[rr, cc].max()))

check("exact finite-N characteristic determinant", max_det_rel < 2e-11,
      f"maximum relative error = {max_det_rel:.2e}")
check("exact scalar Schur complement", max_schur_rel < 2e-12,
      f"maximum relative error = {max_schur_rel:.2e}")
check("closed-form coherence-block eigenvalues", max_coh_err < 2e-11,
      f"maximum matching error = {max_coh_err:.2e}")

# Closed-form ellipse margin against dense deterministic minimisation.
max_margin_err = 0.0
for gphi in (8.0, 10.0, 30.0, 100.0):
    pa = LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.35, gphi=gphi, pbc=True)
    qs = np.linspace(0.0, 2.0 * np.pi, 120_001, endpoint=False)
    dense = np.min(np.abs(np.array([effective_markov_symbol(q, pa) for q in qs]) + 0.4))
    closed = classical_point_gap_margin(pa, -0.4)
    max_margin_err = max(max_margin_err, abs(dense - closed))
check("closed-form effective-Markov point-gap margin", max_margin_err < 1e-8,
      f"maximum absolute error = {max_margin_err:.2e}")

# Actual scalar error lies below both finite-N and uniform bounds in their domains.
max_ratio_finite = 0.0
for gphi in (8.0, 8.1, 10.0, 20.0, 100.0):
    pa = LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.35, gphi=gphi, pbc=True)
    qs = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    actual = max(abs(exact_schur_scalar(q, pa, -0.4)
                     - (effective_markov_symbol(q, pa) + 0.4)) for q in qs)
    bound = analytical_error_bound(pa, -0.4)
    max_ratio_finite = max(max_ratio_finite, actual / bound)
check("actual scalar error is below the analytical finite-N bound",
      max_ratio_finite < 1.0, f"maximum actual/bound ratio = {max_ratio_finite:.3f}")

cert_n10 = winding_certificate(
    LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.35, gphi=8.0, pbc=True), -0.4)
cert_all = winding_certificate(
    LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.35, gphi=8.1, pbc=True),
    -0.4, uniform_in_N=True)
check("finite-N winding certificate holds at gamma_phi=8", cert_n10["certified"],
      f"margin={cert_n10['classical_margin']:.6f}, bound={cert_n10['error_bound']:.6f}")
check("uniform-in-N certificate holds for every finite N>=4 at gamma_phi=8.1",
      cert_all["certified"],
      f"margin={cert_all['classical_margin']:.6f}, bound={cert_all['error_bound']:.6f}")

# --------------------------------------------------------------------------
print("\n--- 6. Open-boundary stationary state ---")
# For J = 0 the open chain is a pure biased walk: p_j proportional to (G_R/G_L)^j
p_ob = LHNParams(N=12, J=0.0, G_R=1.0, G_L=0.4, gphi=0.0, pbc=False)
L_ob = liouvillian(p_ob)
rho = steady_state(L_ob, p_ob.N)
prof = steady_state_profile(rho)
ratio = p_ob.G_R / p_ob.G_L
pred = ratio ** np.arange(p_ob.N)
pred = pred / pred.sum()
check("OBC steady state = (G_R/G_L)^j", np.abs(prof - pred).max() < 1e-8,
      f"max dev = {np.abs(prof - pred).max():.2e}")

# PBC steady state is uniform
p_pb = p_ob.copy(pbc=True)
rho_pb = steady_state(liouvillian(p_pb), p_pb.N)
prof_pb = steady_state_profile(rho_pb)
check("PBC steady state is uniform",
      np.abs(prof_pb - 1.0 / p_pb.N).max() < 1e-9,
      f"max dev = {np.abs(prof_pb - 1.0/p_pb.N).max():.2e}")

# --------------------------------------------------------------------------
print("\n--- 7. Exceptional point of the driven qubit ---")
gamma = 1.0
Om_ep = gamma / 4.0
s_at = ep_splitting(Om_ep, gamma, 0.0)
check("Liouvillian eigenvalues coalesce at Omega = gamma/4", s_at < 1e-6,
      f"splitting = {s_at:.2e}")

# Square-root response requires a perturbation that acts INSIDE the Jordan block.
# Perturbing Omega does; perturbing the detuning does NOT (it only couples the
# block to the third, well-separated Liouvillian mode, giving a linear response).
eps = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
sp = np.array([ep_splitting(Om_ep + e, gamma, 0.0) for e in eps])
slope = np.polyfit(np.log(eps), np.log(sp), 1)[0]
check("splitting ~ eps^(1/2) for a perturbation of Omega", abs(slope - 0.5) < 0.02,
      f"fitted exponent = {slope:.4f}")

eps_det = np.logspace(-6, -2, 5)
spd = np.array([ep_splitting(Om_ep, gamma, e) for e in eps_det])
sld = np.polyfit(np.log(eps_det), np.log(spd), 1)[0]
check("splitting ~ delta^1 for a detuning perturbation", abs(sld - 1.0) < 0.02,
      f"fitted exponent = {sld:.4f}")

# the QFI must not be enhanced at the EP
from lhn.metrology import (ep_liouvillian, steady_state, d_steady_state,
                           quantum_fisher_information)
def _qfi(Om, h=1e-6):
    L0 = ep_liouvillian(Om, gamma, 0.0)
    dL = (ep_liouvillian(Om + h, gamma, 0.0) - ep_liouvillian(Om - h, gamma, 0.0)) / (2 * h)
    r = steady_state(L0, 2)
    return quantum_fisher_information(r, d_steady_state(L0, dL, r, 2))
f_ep, f_lo, f_hi = _qfi(Om_ep), _qfi(Om_ep - 0.05), _qfi(Om_ep + 0.05)
check("QFI has no local maximum at the EP", not (f_ep > f_lo and f_ep > f_hi),
      f"F_Q(below)={f_lo:.4f}, F_Q(EP)={f_ep:.4f}, F_Q(above)={f_hi:.4f}")

# --------------------------------------------------------------------------
print("\n--- 8. Sensing scaling ---")
from lhn.metrology import lhn_sensor_fisher
Fs_t = [lhn_sensor_fisher(LHNParams(N=N, J=0.0, G_R=1.0, G_L=0.4, gphi=0.2),
                          theta0=0.0)["F_C"] for N in [18, 20, 22, 24]]
ratio = np.mean([Fs_t[i+1] / Fs_t[i] for i in range(3)])
check("J=0 Fisher info grows as (G_R/G_L)^N", abs(ratio - 6.25) < 1e-3,
      f"measured ratio per +2 sites = {ratio:.5f}  (expected 6.25)")

Fs_r = np.array([lhn_sensor_fisher(LHNParams(N=N, J=0.5, G_R=0.7, G_L=0.7, gphi=0.2),
                                   theta0=0.0)["F_C"] for N in [18, 20, 22, 24]])
pw = np.polyfit(np.log([18, 20, 22, 24]), np.log(Fs_r), 1)[0]
check("reciprocal Fisher info grows as N^2", abs(pw - 2.0) < 0.1,
      f"fitted power = {pw:.4f}")

# --------------------------------------------------------------------------
print("\n--- 9. Unit conventions (regression guard) ---")
from lhn.physical import (to_angular, to_cyclic, hopping_MHz, ep_rabi_MHz,
                          dephasing_from_current_MHz, T1_RANGE_US, T2_STAR_NS,
                          T2_ECHO_NS, RABI_MAX_MHZ, ep_rabi_range_MHz)

check("to_angular / to_cyclic round-trip", abs(to_cyclic(to_angular(13.7)) - 13.7) < 1e-12)
check("to_angular applies 2*pi", abs(to_angular(1.0) - 2 * np.pi) < 1e-12,
      f"to_angular(1 MHz) = {to_angular(1.0):.4f} rad/us")

# A measured coupling must be converted before entering the Hamiltonian.
J_cyc = hopping_MHz(0.72, quiet=True)
check("dipolar hopping converted to angular differs by 2*pi",
      abs(to_angular(J_cyc) / J_cyc - 2 * np.pi) < 1e-9,
      f"{J_cyc:.1f} MHz -> {to_angular(J_cyc):.1f} rad/us")

# The Fe-derived dephasing law must refuse to run without acknowledgement.
try:
    dephasing_from_current_MHz(10.0)
    guarded = False
except ValueError:
    guarded = True
check("Fe-derived dephasing law is opt-in only", guarded)

# EP estimates must be reported as a range, and all must sit below the ceiling.
eps = [f for _, _, _, f in ep_rabi_range_MHz()]
check("EP estimates span >5x across T1/T2 conventions", max(eps) / min(eps) > 5,
      f"{min(eps):.3f} - {max(eps):.3f} MHz  (spread {max(eps)/min(eps):.1f}x)")
check("all EP estimates lie below the Rabi ceiling", max(eps) < RABI_MAX_MHZ,
      f"max {max(eps):.3f} MHz < {RABI_MAX_MHZ:.0f} MHz")
check("all displayed T1/T2 pairs satisfy T2 <= 2 T1",
      all(t2 <= 2.0 * t1 + 1e-12 for _, t1, t2, _ in ep_rabi_range_MHz()),
      "central values are compatible with non-negative Markovian pure dephasing")

from lhn.metrology import gamma_z_from_times
try:
    gamma_z_from_times(0.142, 0.292)
    incompatible_guard = False
except ValueError:
    incompatible_guard = True
check("incompatible T1/T2 pair is rejected by the Markovian dephasing model",
      incompatible_guard)

# --------------------------------------------------------------------------
print("\n--- 10. Regression guards against overstatement ---")
from lhn.topology import mode_centre_of_mass

# (a) H_eff bulk spectrum is a LINE -> zero winding, but points off it ARE
#     point-gapped. "No point gap" is incorrect; "zero winding" is the appropriate statement.
pl = LHNParams(N=10, J=0.6, G_R=1.0, G_L=0.35, gphi=0.25, pbc=True)
ks = np.linspace(0, 2 * np.pi, 400)
line = np.array([bloch_postselected(pl)(k)[0, 0] for k in ks])
lam_off = line.mean() - 0.6j
check("H_eff: reference point off the line IS point-gapped",
      np.abs(line - lam_off).min() > 0.1,
      f"min |lam - spectrum| = {np.abs(line - lam_off).min():.3f} > 0")
check("H_eff: ...and its winding is nonetheless zero",
      winding_number(bloch_postselected(pl), lam=lam_off) == 0)

# (b) eigenoperator accumulation is NOT extensive -- fraction must DECREASE
fr = []
for n in [8, 12, 16]:
    pn = LHNParams(N=n, J=0.6, G_R=1.0, G_L=0.35, gphi=0.25, pbc=False)
    com = mode_centre_of_mass(liouvillian(pn), n)
    com = com[~np.isnan(com)]
    fr.append(np.mean(com > 0.7 * (n - 1)))
check("edge-localized mode fraction decreases over the sampled sizes",
      fr[-1] < fr[0], f"N=8,12,16 -> {[round(f,3) for f in fr]}")

# (c) but the directional shift IS real
pn = LHNParams(N=12, J=0.6, G_R=1.0, G_L=0.35, gphi=0.25, pbc=False)
com_nr = mode_centre_of_mass(liouvillian(pn), 12); com_nr = com_nr[~np.isnan(com_nr)]
pr = pn.copy(G_R=0.675, G_L=0.675)
com_r = mode_centre_of_mass(liouvillian(pr), 12); com_r = com_r[~np.isnan(com_r)]
check("directional shift in eigenoperator weight is real",
      com_nr.mean() > com_r.mean() + 0.3,
      f"<COM> {com_nr.mean()/11:.3f} vs {com_r.mean()/11:.3f} (normalised)")

# (d) no anomalous relaxation: OBC gap must not close faster than PBC
def _gap(pp):
    e = np.linalg.eigvals(liouvillian(pp))
    e = e[np.abs(e) > 1e-9 * max(1.0, np.abs(e).max())]
    return -e.real.max()
Ns_ = [8, 12, 16]
sO = np.polyfit(np.log(Ns_), np.log([_gap(LHNParams(N=n, J=0.6, G_R=1.0, G_L=0.35,
                gphi=0.25, pbc=False)) for n in Ns_]), 1)[0]
sP = np.polyfit(np.log(Ns_), np.log([_gap(LHNParams(N=n, J=0.6, G_R=1.0, G_L=0.35,
                gphi=0.25, pbc=True)) for n in Ns_]), 1)[0]
check("OBC gap closes more slowly than PBC over the sampled sizes",
      sO > sP, f"OBC ~ N^{sO:.2f}, PBC ~ N^{sP:.2f}")

# (e) Regression guard for the superseded Figure 2(c): at lambda=-2 the
# zero-frequency Schur complement winds at weak dephasing while the FULL
# Liouvillian does not. A scalar-only test would therefore give a false result.
LAM_OLD = -2.0 + 0.0j
pg0 = LHNParams(N=10, J=1.0, G_R=1.0, G_L=0.35, gphi=0.0, pbc=True)
w_old_full = winding_number(lambda q: liouvillian_block_q(q, pg0), LAM_OLD, n_q=768)
w_old_schur = winding_number(lambda q: np.array([[schur_r0(q, pg0)]]), LAM_OLD, n_q=768)
check("regression: scalar Schur winding is not the full Liouvillian winding",
      w_old_full == 0 and w_old_schur == -1,
      f"full={w_old_full}, zero-frequency Schur={w_old_schur}")

# --------------------------------------------------------------------------
n_pass, n_tot = sum(results), len(results)
print(f"\n================  {n_pass}/{n_tot} checks passed  ================")
sys.exit(0 if n_pass == n_tot else 1)
