"""
FIGURE 4 -- Quantized phase label and parameter-sensitive response.

(a) Real-space winding of a disordered classical ring at one fixed reference
    point. Each curve follows one fixed disorder realization continuously as W
    is varied; all rates remain strictly positive because W <= 0.95.
(b) Illustrative lifting of a second-order exceptional point under drive
    mistuning. This is not a like-for-like disorder comparison with panel (a).
(c) Distribution of the one-sided boundary-rate Fisher information under bond
    disorder. The winding is integer-valued; the response magnitude is not.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt

from lhn import LHNParams
from lhn.models import disordered_markov_generator
from lhn.topology import real_space_winding
from lhn.metrology import lhn_sensor_fisher, ep_splitting
from lhn.style import use_style, C, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)   # so a fresh clone works even without figures/
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

N_CH, GR, GL = 18, 1.0, 0.4
N_SAMP = 24
N_SENS = 12
N_SAMP_C = 32
LAM_REF = -(GR + GL) + 0.0j

# Fixed base disorder arrays define continuous paths as W is varied.
base_disorder = []
for s in range(N_SAMP):
    rng = np.random.default_rng(1000 + s)
    base_disorder.append((rng.uniform(-1, 1, N_CH), rng.uniform(-1, 1, N_CH)))

# ---- (a) quantized winding under disorder -------------------------------
ax = axes[0]
Ws = np.linspace(0.0, 0.95, 20)
allw = np.zeros((len(Ws), N_SAMP))
for i, W in enumerate(Ws):
    for s, (uR, uL) in enumerate(base_disorder):
        allw[i, s] = real_space_winding(
            lambda ph, W=W, uR=uR, uL=uL: disordered_markov_generator(
                N_CH, GR, GL, W, phi=ph, base_R=uR, base_L=uL),
            lam=LAM_REF, n_phi=160)
med = np.median(allw, axis=1)
for s in range(N_SAMP):
    ax.plot(Ws, allw[:, s], "-", color=C["topo"], alpha=0.18, lw=1.0)
ax.plot(Ws, med, "o-", color=C["topo"], ms=5, mec="k", mew=0.35,
        label=f"median of {N_SAMP} paths")
ax.set_ylim(-1.55, 0.55)
ax.set_yticks([-1, 0])
ax.set_xlabel("bond-disorder strength $W$")
ax.set_ylabel("winding $w(\\lambda_0)$")
ax.set_title(f"Fixed $\\lambda_0={LAM_REF.real:.2f}$, positive rates ($N={N_CH}$)")
ax.legend(loc="center left", fontsize=8)
panel_label(ax, "a")

# ---- (b) illustrative EP mistuning --------------------------------------
ax = axes[1]
gamma = 1.0
Om_ep = gamma / 4.0
Wd = np.logspace(-6, -0.5, 34)
lift = []
for W in Wd:
    vals = []
    for s in range(120):
        rng = np.random.default_rng(7000 + s)
        Om = Om_ep * (1.0 + W * rng.uniform(-1, 1))
        vals.append(ep_splitting(Om, gamma, 0.0))
    lift.append(np.median(vals))
lift = np.array(lift)
sl = np.polyfit(np.log(Wd), np.log(lift), 1)[0]
ax.loglog(Wd, lift, "o-", color=C["accent"], ms=4.2, mec="k", mew=0.3,
          label="median eigenvalue splitting")
ax.loglog(Wd, 2 * np.sqrt(gamma * Om_ep * Wd / 4), "--", color="k", lw=1.3,
          label="$\\sqrt{W}$ guide")
ax.set_xlabel("fractional drive mistuning $W$")
ax.set_ylabel("residual eigenvalue splitting")
ax.set_title(f"Exceptional-point mistuning (slope ${sl:.2f}$)")
ax.legend(loc="upper left", fontsize=8)
panel_label(ax, "b")

# ---- (c) boundary-rate response under disorder --------------------------
ax = axes[2]
Wl = np.array([0.0, 0.2, 0.4, 0.6])
for tag, gr, gl_, col, off, seed0 in [
        ("biased chain", 1.0, 0.4, C["topo"], -0.025, 5000),
        ("reciprocal chain", 0.7, 0.7, C["triv"], +0.025, 9000)]:
    # One base realization per sample, reused across W.
    uv = []
    for s in range(N_SAMP_C):
        rng = np.random.default_rng(seed0 + s)
        uv.append((rng.uniform(-1, 1, N_SENS - 1),
                   rng.uniform(-1, 1, N_SENS - 1)))
    meds, los, his = [], [], []
    for W in Wl:
        vals = []
        for uR, uL in uv:
            rd = {"R": 1 + W * uR, "L": 1 + W * uL}
            p = LHNParams(N=N_SENS, J=0.5, G_R=gr, G_L=gl_, gphi=0.2)
            vals.append(lhn_sensor_fisher(p, theta0=0.0, rate_disorder=rd)["F_C"])
        vals = np.asarray(vals)
        meds.append(np.median(vals))
        los.append(np.percentile(vals, 15))
        his.append(np.percentile(vals, 85))
    x = Wl + off
    ax.errorbar(x, meds, yerr=[np.asarray(meds)-los, np.asarray(his)-np.asarray(meds)],
                fmt="o-", color=col, ms=6, mec="k", mew=0.35, capsize=3, lw=1.8,
                label=tag)
ax.set_yscale("log")
ax.set_xlabel("bond-disorder strength $W$")
ax.set_ylabel("one-sided $F_C(\\theta=0^+)$")
ax.set_title(f"Boundary-rate response ($N={N_SENS}$, {N_SAMP_C} paths)")
ax.legend(loc="center left", fontsize=8)
panel_label(ax, "c")

fig.suptitle("Quantised winding and non-quantised response", fontsize=12.5, y=1.005)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_robustness.png"))
print("winding medians:", med.astype(int))
print(f"EP lifting exponent = {sl:.3f}")
print("saved figures/fig4_robustness.png")
