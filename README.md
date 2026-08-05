# Point-gap topology from Lindblad to Markov dynamics — simulation code

Simulation and analysis code accompanying the manuscript
*Analytically Controlled Point-Gap Topology from Lindblad to Markov Dynamics*.

This repository contains **code only**. It reproduces every figure and every
numerical value quoted in the paper from scratch. Pure `numpy` / `scipy` /
`matplotlib` — no compiled dependencies, no cluster, under two minutes on a
laptop.

## Quick start

```bash
git clone https://github.com/<USERNAME>/lhn-point-gap-topology.git
cd lhn-point-gap-topology
pip install -r requirements.txt

python scripts/validate.py    # 50 checks against closed-form results
python scripts/run_all.py     # regenerates all figures into figures/
```

`validate.py` is the entry point that matters. Every quoted number is checked
against an analytic result derived in `docs/derivation.md`. Several checks exist
specifically to fail if a superseded formula or an unsupported claim is
reintroduced.

## Layout

```
lhn/
  models.py       Lindblad Hatano-Nelson model, Liouvillian, momentum blocks,
                  classical Markov generators
  analytical.py   exact finite-N characteristic determinant, scalar Schur
                  complement, effective Markov symbol, and the uniform error
                  bound / homotopy certificate
  topology.py     point-gap winding numbers (Bloch and real-space twist),
                  localisation diagnostics
  metrology.py    stationary states, Fisher information, driven-qubit
                  exceptional-point reference
  physical.py     literature parameters for Ti on 2 ML MgO/Ag(100), with unit
                  conversions and per-value caveats
  style.py        figure styling

scripts/
  validate.py                 50 analytic and regression checks
  run_all.py                  regenerate every figure
  fig1_topology.py            point-gap winding, boundary diagnostics
  fig2_superdecoherence.py    quantum-to-classical reduction
  fig3_metrology.py           boundary-rate Fisher information vs EP response
  fig4_robustness.py          disordered winding, non-quantised response
  fig5_esr_stm.py             comparison with reported ESR-STM scales
  figS1_phase_certificate.py  supplementary winding certificate map

docs/derivation.md            analytics behind every figure panel
figures/                      generated output (not tracked)
```

## Reproducing specific claims

| Claim in the paper | Where |
|---|---|
| Exact finite-$N$ characteristic determinant | `lhn/analytical.py::exact_characteristic_determinant` |
| Homotopy criterion for winding equality | `lhn/analytical.py::winding_certificate` |
| Effective symmetric rate $D = 2J^2/(\gamma_\phi+\Gamma_R+\Gamma_L)$ | `lhn/models.py::symmetric_rate_D` |
| Winding at a fixed reference point across dephasing | `scripts/fig2_superdecoherence.py` |
| Boundary-rate Fisher information scaling | `lhn/metrology.py::lhn_sensor_fisher` |
| Liouvillian exceptional-point location | `lhn/physical.py::ep_rabi_MHz` |

## Two conventions that are easy to get wrong

Both are enforced by regression tests in `validate.py`:

1. **Angular vs cyclic frequency.** The Lindblad equation takes Hamiltonian
   coefficients as angular rates (rad/µs); experiments report cyclic
   frequencies ($J/h$, $\Omega/2\pi$, in MHz). Use `physical.to_angular()`
   before putting a measured coupling into a Hamiltonian. Decay rates
   ($1/T_1$, $1/T_2$) need no conversion.
2. **Reference points must be held fixed.** A winding number is only meaningful
   at a reference point excluded from the spectrum and held fixed along any
   parameter path. Recomputing it per parameter value tests nothing.

## Environment

Developed and tested with Python 3.11, `numpy` 2.x, `scipy` 1.1x,
`matplotlib` 3.1x. See `requirements.txt` for minimum versions. Results are
deterministic: all stochastic elements use seeded `numpy.random.default_rng`.

## Citation

If you use this code, please cite both the paper and the archived release
(see `CITATION.cff`).

## License

MIT — see `LICENSE`.
