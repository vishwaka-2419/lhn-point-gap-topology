# Certified point-gap topology under coherence elimination

This repository contains the analytical and numerical package for:

> **Certified Point-Gap Topology under Coherence Elimination in a Liouvillian Hatano-Nelson Chain**  
> Aishwarya Vishwakarma, University of Geneva

The model combines reciprocal coherent hopping, directionally biased Lindblad jumps, and local dephasing in the single-excitation sector.

## Main results

1. **Exact finite-size reduction.** The periodic Liouvillian decomposes into momentum blocks whose determinant is available in closed form. The complete point-gap winding reduces exactly to a scalar Schur complement.
2. **Topology-preserving elimination.** Global and momentum-resolved bounds compare the exact scalar with a second-order effective Markov symbol. A positive pointwise reserve proves equality of the full Liouvillian and effective Markov windings.
3. **Grid-independent certificate.** Interval subdivision validates the pointwise inequality over the complete Brillouin zone. For `J=1`, `Gamma_R=1`, `Gamma_L=0.35`, and `lambda=-0.4`, the all-finite-size certificate holds at `gamma_phi=4.33`; the global all-size threshold is approximately `8.01`.
4. **Dimensionless certified region.** The criterion is expressed through `j=|J|/kappa`, `g=gamma_phi/kappa`, `xi=-lambda/kappa`, and `delta=(Gamma_R-Gamma_L)/Sigma`.
5. **Subextensive skin sector.** At `J=0`, exactly `N` population modes carry a directional skin envelope within an `N^2`-mode Liouville space. At finite `J` and strong dephasing, the tested systems retain `N` localized population-associated modes while the coherence modes are extended.

The no-jump Hamiltonian is reciprocal and point-gap trivial in this model, but that contrast is supporting structure rather than the novelty claim.

## Reproduce the validation and figures

```bash
python -m pip install -r requirements.txt
python scripts/validate.py
python scripts/run_all.py
```

Expected validation summary:

```text
33/33 checks passed
```

The exact tested versions are recorded in `requirements-tested.txt`.

## Repository layout

```text
lhn/models.py          Lindblad, Bloch-block, and Markov generators
lhn/analytical.py      Exact determinant and global/pointwise certificates
lhn/boundary.py        Physical and synthetic boundary diagnostics
lhn/topology.py        Determinant winding and point-gap utilities
scripts/validate.py    Deterministic 33-check regression suite
scripts/fig*.py        Main and supplementary figure generation
```

## Code and archival record

Repository: <https://github.com/vishwaka-2419/lhn-point-gap-topology>  
Versioned Zenodo archive: <https://doi.org/10.5281/zenodo.21813150>

The Zenodo concept DOI resolves to the latest archived version. Cite the specific-version DOI shown on the Zenodo landing page when an immutable version-specific citation is required.

## Citation

Use `CITATION.cff` or cite the associated article and archived software release.

## License

Code is released under the MIT License. Manuscript text and figures remain subject to the journal or preprint licence selected by the author.
