"""Certified point-gap topology under coherence elimination."""

from .models import (
    LHNParams,
    hamiltonian,
    jump_operators,
    liouvillian,
    postselected_hamiltonian,
    liouvillian_block_q,
    classical_markov_generator,
    effective_classical_generator,
    coherence_decay_rate,
    symmetric_rate_D,
    schur_r0,
)
from .topology import (
    winding_number,
    winding_map,
    point_gap_margin,
    bloch_classical,
    bloch_postselected,
    pbc_spectrum_from_blocks,
    steady_state_profile,
)
from .analytical import (
    coherent_entries,
    population_entry,
    coherence_eigenvalues,
    theta_sequence,
    exact_characteristic_determinant,
    exact_schur_scalar,
    effective_markov_symbol,
    classical_winding_real_reference,
    classical_point_gap_margin,
    analytical_error_bound,
    pointwise_error_bound,
    pointwise_gap_reserve,
    winding_certificate,
    interval_validated_pointwise_certificate,
    dimensionless_coordinates,
    centered_dimensionless_uniform_reserve,
    centered_dimensionless_pointwise_reserve,
)
from .boundary import (
    liouville_bloch_blocks,
    synthetic_liouville_obc,
    physical_eigenoperator_metrics,
    slow_skin_sector,
)

__version__ = "2.0.0"
