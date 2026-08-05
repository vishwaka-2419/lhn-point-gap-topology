"""
Literature parameters for Ti spin qubits on 2 ML MgO/Ag(100).

Two conventions requiring explicit treatment
------------------------------------------------------------------------------
(1) UNITS. Rates are in MHz = us^-1 and times in microseconds. The Lindblad
    equation rho' = -i[H, rho] takes Hamiltonian coefficients as ANGULAR rates
    (rad/us), whereas experiments report couplings as CYCLIC frequencies
    (J/h, Omega/2pi, in MHz). Converting requires a factor of 2 pi. Use
    `to_angular()` before putting any measured coupling into LHNParams(J=...).
    Decay RATES (1/T1, 1/T2) are convention-free and need no conversion.

(2) WHICH T2. A Markovian Lindbladian has one transverse rate. Experiments quote
    several, and they are not interchangeable:
      - An unrefocused Rabi-envelope decay time of order 40 ns provides an
        indicative driven-coherence scale, but is not by itself a calibrated
        Markovian T2.
      - Hahn echo (189 ns) and dynamical decoupling (277-292 ns) are refocused
        sequence-dependent coherence times.
    These quantities should not be interchanged in a time-independent
    Bloch-Liouvillian without a noise model. Greule et al. (NJP 28, 083501,
    2026) found an approximately 30 ns two-delay Hahn-echo scale for FePc/MgO
    and showed how a conventional echo can contain relaxation contributions.
    That result is methodological evidence only: it is not Ti data and is not
    used in the Ti parameter calculations below.
    Functions below therefore require T2 explicitly and do not select a default.

------------------------------------------------------------------------------
Quantity                    Value                     Source / caveat
------------------------------------------------------------------------------
Spin                        S = 1/2 (Ti_O, Ti_B)      Yang PRL 119, 227206 (2017)
g-factor                    ~1.79                     from f0/B below
External field              0.82-0.9 T                Yang Science 366 (2019)
Larmor frequency            20.55 GHz @ 0.82 T        Yang Science 366 (2019)
Temperature                 0.4-1.2 K                 various
Rabi-envelope decay          ~40 ns                    Yang Science 366 (2019); indicative
T2 (Hahn echo)              189 +- 23 ns              Yang Science 366 (2019)
T2 (dyn. decoupling)        277 +- 20, 292 +- 23 ns   Wang npj QI 9, 48 (2023)
T1 (spin-flop)              ~1.1 us                   Yang PRL 119, 227206 (2017)
T1 (inferred, T2 ~ 2T1)     ~140 ns                   Wang npj QI 9, 48 (2023)
Rabi rate                   Omega/2pi <~ 25 MHz       Yang Science 366 (2019)
pi/2 pulse                  6-13 ns                   Wang npj QI 9, 48 (2023)
Tunnel current              1-100 pA                  various
Ti-Ti spacings              0.72, 0.92 nm             Bae Sci. Adv. 4, eaau4159
Exchange (strong dimer)     J_ex/h = 28.9 +- 1.3 GHz  Bae Sci. Adv. 4, eaau4159
------------------------------------------------------------------------------

T1 IS CONDITION-DEPENDENT. Both ~1.1 us (Yang 2017, spin-flop) and ~140 ns
(inferred from T2 ~ 2T1 in Wang 2023) appear in the literature for Ti on MgO;
T1 depends strongly on tip proximity, bias and current. This module exposes the
range rather than picking a value, and downstream estimates are reported as
intervals.

DEPHASING PER TUNNELLING ELECTRON: the widely quoted ~0.64 events/electron of
Willke et al., Sci. Adv. 4, eaaq1543 (2018) was measured on an **Fe** atom, not
Ti. It is provided here only as an order-of-magnitude reference and must not be
used as a calibrated Ti dephasing law. `dephasing_from_current_MHz` therefore
requires an explicit opt-in.

INTERATOMIC COUPLING: at sub-nanometre separation the Ti-Ti interaction is not
given by the bare dipolar formula -- ESR-STM finds a crossover to
exchange-dominated coupling, with J_ex/h ~ 29 GHz reported for a strongly
coupled dimer. `hopping_MHz` is only an order-of-magnitude secular dipolar estimate for
the stated geometry, and warns below ~1 nm.
"""

import warnings
import numpy as np

MU_0 = 4e-7 * np.pi
MU_B = 9.2740100783e-24          # J/T
H_PLANCK = 6.62607015e-34        # J s
K_B = 1.380649e-23               # J/K
E_CHARGE = 1.602176634e-19       # C

G_FACTOR = 1.79
B_EXT_T = 0.82
F_LARMOR_GHZ = 20.55
TEMP_K = (0.4, 1.2)

# transverse times (ns), labelled by what they actually measure
T2_STAR_NS = 40.0            # unrefocused Rabi-envelope decay; indicative only
T2_ECHO_NS = 189.0           # Hahn echo -- refocused
T2_DD_NS = (277.0, 292.0)    # dynamical decoupling -- refocused

# longitudinal time (us): literature range, condition-dependent
T1_RANGE_US = (0.142, 1.1)

RABI_MAX_MHZ = 25.0
SPACINGS_NM = (0.72, 0.92)
J_EXCHANGE_DIMER_GHZ = 28.9
DEPHASING_PER_ELECTRON_FE = 0.64     # Fe, NOT Ti


# ---------------------------------------------------------------- conversions
def to_angular(f_MHz):
    """Cyclic frequency (MHz, i.e. E/h) -> angular rate (rad/us) for use in H."""
    return 2.0 * np.pi * np.asarray(f_MHz, dtype=float)


def to_cyclic(omega_rad_per_us):
    """Angular rate (rad/us) -> cyclic frequency (MHz), for comparison to data."""
    return np.asarray(omega_rad_per_us, dtype=float) / (2.0 * np.pi)


# ------------------------------------------------------------------ couplings
def dipolar_coupling_MHz(r_nm, g=G_FACTOR, S=0.5, theta_deg=90.0, quiet=False):
    """Secular dipolar prefactor for two spin-1/2 centres (cyclic MHz).

    The returned value is the coefficient
    D_dd = (mu_0/4pi)(g mu_B)^2 |1-3 cos^2(theta)|/(h r^3).
    The single-excitation flip-flop matrix element is D_dd/4 in the standard
    high-field secular Hamiltonian. ``S`` is retained only for API compatibility
    and must equal 1/2.

    This estimate is meaningful only in a dipolar-dominated geometry. At the
    sub-nanometre Ti separations considered experimentally, exchange can dominate.
    """
    if S != 0.5:
        raise ValueError("the implemented secular flip-flop convention assumes S=1/2")
    if r_nm <= 0:
        raise ValueError("r_nm must be positive")
    if r_nm < 1.0 and not quiet:
        warnings.warn(
            f"r = {r_nm} nm is below ~1 nm: measured Ti-Ti coupling is "
            "exchange-dominated here (J_ex/h ~ 29 GHz for a strongly coupled "
            "dimer). The dipolar value is only an order-of-magnitude estimate "
            "for the specified orientation.", stacklevel=2)
    r = r_nm * 1e-9
    ang = abs(1.0 - 3.0 * np.cos(np.deg2rad(theta_deg)) ** 2)
    energy = (MU_0 / (4.0 * np.pi)) * (g * MU_B) ** 2 * ang / r ** 3
    return energy / H_PLANCK * 1e-6


def hopping_MHz(r_nm, **kw):
    """High-field secular flip-flop matrix element (cyclic MHz).

    For two spin-1/2 centres the off-diagonal matrix element in the
    single-excitation sector is one quarter of the dipolar prefactor returned by
    :func:`dipolar_coupling_MHz`. Apply :func:`to_angular` before using this value
    as a Hamiltonian coefficient.
    """
    return 0.25 * dipolar_coupling_MHz(r_nm, **kw)


def exchange_hopping_MHz(J_ex_GHz=J_EXCHANGE_DIMER_GHZ):
    """Flip-flop matrix element J_ex/2 for an isotropic Heisenberg pair (MHz)."""
    return 0.5 * J_ex_GHz * 1e3


# ------------------------------------------------------------------ dephasing
def dephasing_from_current_MHz(I_pA, eta=DEPHASING_PER_ELECTRON_FE,
                               offset_MHz=3.3, i_understand_this_is_Fe=False):
    """Order-of-magnitude dephasing rate vs tunnel current.

    Willke et al. measured, on an **Fe** atom, a decoherence rate equal to a
    term linear in tunnel current (of order one dephasing event per tunnelling
    electron) plus a current-independent offset. Applying that coefficient to Ti
    is an unvalidated extrapolation, so this function refuses to run unless the
    caller acknowledges it. One pA delivers I/e = 6.24 electrons per microsecond.
    """
    if not i_understand_this_is_Fe:
        raise ValueError(
            "The 0.64 events/electron coefficient was measured on Fe, not Ti. "
            "Pass i_understand_this_is_Fe=True to use it as an order-of-"
            "magnitude reference only, and label it as such in any figure.")
    return offset_MHz + eta * (np.asarray(I_pA, float) * 1e-12 / E_CHARGE) * 1e-6


# -------------------------------------------------------------------- thermal
def thermal_energy_GHz(T_K):
    """k_B T as a cyclic frequency, in GHz."""
    return K_B * T_K / H_PLANCK * 1e-9


def detailed_balance_ratio(delta_MHz, T_K):
    """Boltzmann ratio exp(h*delta / k_B T) for a Zeeman step delta (cyclic MHz)."""
    return np.exp(delta_MHz * 1e6 * H_PLANCK / (K_B * T_K))


# ---------------------------------------------------------- exceptional point
def ep_rabi_MHz(T1_us, T2_us):
    """Cyclic Rabi frequency at the driven-qubit Liouvillian EP.

    Omega_EP = |1/T2 - 1/T1| / 2  (rad/us), returned as Omega_EP/2pi (MHz) so it
    can be compared directly with experimentally reported Rabi rates.
    """
    return to_cyclic(0.5 * abs(1.0 / T2_us - 1.0 / T1_us))


def ep_rabi_range_MHz():
    """Illustrative EP frequencies from reported, condition-dependent times.

    The listed T1 and T2 values were not necessarily measured under identical
    conditions, and echo/DD times are sequence-dependent. The resulting range is
    therefore an order-of-magnitude comparison with demonstrated drive rates,
    not a platform-specific prediction.
    """
    T1_lo, T1_hi = 1e3 * T1_RANGE_US[0], 1e3 * T1_RANGE_US[1]
    cases = [
        ("Rabi-envelope scale, T1 inferred", T1_lo, T2_STAR_NS),
        ("Rabi-envelope scale, T1 spin-flop", T1_hi, T2_STAR_NS),
        ("Hahn echo refocused, T1 inferred", T1_lo, T2_ECHO_NS),
        ("Hahn echo refocused, T1 spin-flop", T1_hi, T2_ECHO_NS),
        ("Dyn. decoupling refocused, T1 inferred", T1_lo, T2_DD_NS[0]),
    ]
    return [(lab, t1, t2, float(ep_rabi_MHz(t1 * 1e-3, t2 * 1e-3)))
            for lab, t1, t2 in cases]


def summary():
    a = []
    a.append(f"g = {G_FACTOR},  B = {B_EXT_T} T,  f_Larmor = {F_LARMOR_GHZ} GHz")
    a.append("")
    a.append("couplings (cyclic MHz; multiply by 2*pi for the Hamiltonian):")
    for r in SPACINGS_NM:
        h = hopping_MHz(r, quiet=True)
        a.append(f"  r = {r} nm: dipolar hopping {h:7.1f} MHz "
                 f"= {to_angular(h):8.1f} rad/us   [bare dipolar estimate]")
    a.append(f"  exchange dimer:  flip-flop  {exchange_hopping_MHz():8.0f} MHz "
             "(J_ex/2, strongly coupled pair)")
    a.append("")
    a.append("decay rates (convention-free):")
    for lab, t in [("Rabi env. decay", T2_STAR_NS), ("T2 Hahn echo", T2_ECHO_NS),
                   ("T2 dyn. decoup.", T2_DD_NS[1])]:
        a.append(f"  1/{lab:16s} = {1e3/t:7.2f} MHz   (T2 = {t:.0f} ns)")
    for t in T1_RANGE_US:
        a.append(f"  1/{'T1':16s} = {1.0/t:7.2f} MHz   (T1 = {t*1e3:.0f} ns)")
    a.append("")
    a.append("Liouvillian EP, cyclic Rabi frequency:")
    for lab, t1, t2, f in ep_rabi_range_MHz():
        a.append(f"  {lab:40s} {f:6.3f} MHz")
    a.append(f"  experimental ceiling ~{RABI_MAX_MHZ:.0f} MHz")
    a.append("")
    for T in TEMP_K:
        a.append(f"k_B T at {T} K = {thermal_energy_GHz(T):.2f} GHz")
    return "\n".join(a)


if __name__ == "__main__":
    print(summary())
