# Certified point-gap topology under coherence elimination

This repository contains the analytical and numerical package for:

> **Certified Point-Gap Topology under Coherence Elimination in a Liouvillian Hatano--Nelson Chain**  
> Aishwarya Vishwakarma, University of Geneva

The model has reciprocal coherent hopping, directionally biased Lindblad jumps,
and local dephasing in the single-excitation sector.

## Main results

1. **Exact finite-size reduction.** The periodic Liouvillian decomposes into
   momentum blocks whose determinant is available in closed form. The complete
   point-gap winding reduces exactly to a scalar Schur complement.
2. **Topology-preserving elimination.** Global and momentum-resolved bounds
   compare the exact scalar with a second-order effective Markov symbol. A
   positive pointwise reserve proves equality of the full Liouvillian and
   effective Markov windings.
3. **Grid-independent certificate.** Interval subdivision validates the
   pointwise inequality over the complete Brillouin zone. For
   `J=1, Gamma_R=1, Gamma_L=0.35, lambda=-0.4`, the all-finite-size certificate
   holds at `gamma_phi=4.33`; the global all-size threshold is about `8.01`.
4. **Dimensionless certified region.** The criterion is expressed through
   `j=|J|/kappa`, `g=gamma_phi/kappa`, `xi=-lambda/kappa`, and
   `delta=(Gamma_R-Gamma_L)/Sigma`.
5. **Subextensive skin sector.** At `J=0`, exactly `N` population modes carry a
   directional skin envelope within an `N^2`-mode Liouville space. At finite
   `J` and strong dephasing, the tested systems retain `N` localized
   population-associated modes while the coherence modes are extended.

The no-jump Hamiltonian is reciprocal and point-gap trivial in this model, but
that contrast is supporting structure rather than the novelty claim.

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
lhn/analytical.py      exact determinant and global/pointwise certificates
lhn/boundary.py        physical and synthetic boundary diagnostics
lhn/topology.py        determinant winding and point-gap utilities
scripts/validate.py    deterministic 33-check regression suite
scripts/fig*.py        main and supplementary figure generation
figures/               generated manuscript figures
paper/                  manuscript and supplementary LaTeX sources
submission/             cover letter and portal-ready text
```

## Figure map

| Figure | Content |
|---|---|
| 1 | exact scalar winding reduction and Markov limit |
| 2 | global, pointwise, and dimensionless certificates |
| 3 | subextensive `N`-mode Liouvillian skin sector |
| S1 | two-parameter winding and certificate hierarchy |
| S2 | physical versus synthetic boundary counting |

## Code and archival record

Repository: <https://github.com/wgarching/lhn-point-gap-topology>  
Archived release: <https://doi.org/10.5281/zenodo.21813151>

The repository URL should be checked before release if the GitHub repository has
not yet been created under this exact name. The Zenodo record should be updated
with the revised release and manuscript title rather than replaced by an
unrelated new record.

## Citation

Use `CITATION.cff` or cite the associated article and archived software release.

## License

Code is released under the MIT License. Manuscript text and figures remain
subject to the journal/preprint licensing selected by the author.
