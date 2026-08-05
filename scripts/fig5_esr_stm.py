"""
FIGURE 5 -- Comparison with reported ESR-STM parameter scales.

This figure deliberately does not assert that the model is implementable on
Ti/MgO today. It reports what is measured, what is uncertain, and which dynamical process remains unavailable.

(a) Measured Ti coherence and control scales. Note that Hahn-echo and
    dynamical-decoupling times are REFOCUSED and are not the transverse rate of
    a time-independent Lindbladian. The Rabi-envelope decay is an
    unrefocused coherence scale, but is not a calibrated Markovian transverse rate.
(b) Liouvillian EP in cyclic Rabi frequency, computed for selected reported
    (T1, T2) pairing. The spread is approximately 14-fold, so a single
    quoted value would be unjustifiably precise. The calculation is illustrative
    because the reported coherence times are protocol- and condition-dependent.
(c) Direction-selective dissipation is required. At the representative
    dipolar hopping scale, equilibrium thermal bias is much smaller than the
    rate asymmetry considered in the model.

All couplings are handled as CYCLIC frequencies and converted with to_angular()
before entering any Hamiltonian.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt

from lhn.physical import (hopping_MHz, exchange_hopping_MHz, to_angular,
                          ep_rabi_range_MHz, RABI_MAX_MHZ, T2_STAR_NS,
                          T2_ECHO_NS, T2_DD_NS, T1_RANGE_US, SPACINGS_NM,
                          thermal_energy_GHz, detailed_balance_ratio, TEMP_K)
from lhn.style import use_style, C, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)   # so a fresh clone works even without figures/
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))

# ---- (a) measured time scales -------------------------------------------
ax = axes[0]
entries = [
    ("Rabi-envelope decay", T2_STAR_NS, C["topo"], "unrefocused; indicative"),
    ("$T_2$ (Hahn echo)", T2_ECHO_NS, C["accent"], "refocused"),
    ("$T_2$ (dyn. decoupl.)", T2_DD_NS[1], C["accent"], "refocused"),
    ("$T_1$ (inferred)", 1e3 * T1_RANGE_US[0], C["triv"], "from $T_2\\simeq2T_1$"),
    ("$T_1$ (spin-flop)", 1e3 * T1_RANGE_US[1], C["triv"], "measured"),
]
y = np.arange(len(entries))[::-1]
for yi, (lab, t, col, note) in zip(y, entries):
    ax.barh(yi, t, height=0.55, color=col, alpha=0.75, edgecolor="k", linewidth=0.5)
    ax.text(t * 1.12, yi, f"{t:.0f} ns  ({note})", va="center", fontsize=7.8)
ax.set_yticks(y)
ax.set_yticklabels([e[0] for e in entries], fontsize=8.5)
ax.set_xscale("log")
ax.set_xlim(20, 1e4)
ax.set_xlabel("time (ns)")
ax.set_title("Reported Ti coherence and relaxation times")
panel_label(ax, "a")

# ---- (b) EP estimate spread ---------------------------------------------
ax = axes[1]
rows = ep_rabi_range_MHz()
vals = [r[3] for r in rows]
yy = np.arange(len(rows))[::-1]
cols = [C["topo"] if "Rabi-envelope" in r[0] else C["accent"] for r in rows]
ax.barh(yy, vals, height=0.55, color=cols, alpha=0.8, edgecolor="k", linewidth=0.5)
for yi, r in zip(yy, rows):
    ax.text(r[3] * 1.15, yi, f"{r[3]:.2f}", va="center", fontsize=8)
ax.set_yticks(yy)
ax.set_yticklabels([r[0].replace(", ", ",\n") for r in rows], fontsize=6.9)
ax.set_xscale("log")
ax.set_xlim(0.05, 200)
ax.axvline(RABI_MAX_MHZ, color=C["gray"], ls="--", lw=1.6)
ax.text(RABI_MAX_MHZ * 1.1, len(rows) - 1.4,
        f"demonstrated\nRabi ceiling\n({RABI_MAX_MHZ:.0f} MHz)",
        fontsize=7.6, color=C["gray"])
ax.set_xlabel("cyclic Rabi frequency at the EP,  $\\Omega_{\\rm EP}/2\\pi$ (MHz)")
ax.set_title(f"Illustrative two-level EP scale: {min(vals):.2f}$-${max(vals):.2f} MHz "
             f"({max(vals)/min(vals):.0f}$\\times$ spread)")
panel_label(ax, "b")

# ---- (c) requirement for direction-selective dissipation -------------
ax = axes[2]
deltas = np.logspace(0, 4.2, 200)          # Zeeman step, cyclic MHz
for T, ls in zip(TEMP_K, ["-", "--"]):
    ax.loglog(deltas, detailed_balance_ratio(deltas, T) - 1.0, ls, lw=2.2,
              color=C["topo"], label=f"$T = {T}$ K")
for target, lab in [(0.6, "$\\Gamma_R/\\Gamma_L = 1.6$"),
                    (1.5, "$\\Gamma_R/\\Gamma_L = 2.5$")]:
    ax.axhline(target, color=C["triv"], ls=":", lw=1.5)
    ax.text(1.3, target * 1.15, lab, fontsize=8, color=C["triv"])
J_dip = hopping_MHz(SPACINGS_NM[0], quiet=True)
ax.axvline(J_dip, color=C["accent"], lw=1.8)
ax.text(J_dip * 1.15, 3e-5, f"representative hopping\n$J\\approx{J_dip:.0f}$ MHz",
        fontsize=7.8, color=C["accent"])
ax.set_xlabel("Zeeman step between sites,  $\\delta$ (MHz)")
ax.set_ylabel("thermal bias  $\\Gamma_R/\\Gamma_L - 1$")
ax.set_ylim(1e-5, 20)
ax.set_title("Equilibrium thermal bias is insufficient")
ax.legend(loc="upper left", fontsize=8)
panel_label(ax, "c")

fig.suptitle("Ti / 2 ML MgO / Ag(100): reported scales and implementation requirement",
             fontsize=12.5, y=1.005)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_esr_stm.png"))

# ---- console report ------------------------------------------------------
print("couplings (cyclic -> angular):")
for r in SPACINGS_NM:
    h = hopping_MHz(r, quiet=True)
    print(f"  r={r} nm: dipolar hopping {h:6.1f} MHz = {to_angular(h):7.1f} rad/us"
          "  [BARE DIPOLAR ESTIMATE: exchange dominates below ~1 nm]")
print(f"  exchange dimer flip-flop: {exchange_hopping_MHz():.0f} MHz")
print(f"\nEP spread: {min(vals):.3f} - {max(vals):.3f} MHz "
      f"({max(vals)/min(vals):.1f}x), ceiling {RABI_MAX_MHZ:.0f} MHz")
print("\nthermal bias at delta = J (representative hopping scale):")
for T in TEMP_K:
    print(f"  T={T} K: kT/h = {thermal_energy_GHz(T)*1e3:8.0f} MHz, "
          f"ratio = {detailed_balance_ratio(J_dip, T):.4f}")
print("=> the equilibrium thermal bias is insufficient at the representative hopping scale.")
print("saved figures/fig5_esr_stm.png")
