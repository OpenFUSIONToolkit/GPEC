module ErrorFields

"""
ErrorFields - Error-field sensitivity of a perturbed equilibrium to coil misalignment

Given one perturbed-equilibrium solve and its coil sets, the module linearizes the resonant
drive of each named coil set with respect to its rigid-body degrees of freedom, so that any
tolerance question downstream (Monte Carlo over manufacturing tolerances, locking risk,
allowable-tolerance scans) reduces to cheap linear algebra on a small table instead of a new
plasma solve or a new Biot-Savart integration.

## Module Structure

- `ErrorFieldsStructs.jl`: `ErrorFieldsControl` (the `[ErrorFields]` TOML section),
  `CoilSensitivities` (the cached linearization: nominal spectra and their derivatives),
  `SensitivityTable` (that linearization projected onto one dominant coupling mode)
- `Sensitivity.jl`: `compute_coil_sensitivities` (central-difference sweep of every rigid
  shift and tilt of every coil set on one shared boundary grid), `sensitivity_table`
- `Output.jl`: HDF5 writer under `ErrorFields/CoilSensitivities/` and the matching reader
- `ToleranceTOML.jl`: the tolerance input file — `read_tolerance_toml`, `ToleranceSet`,
  `validate_tolerances`, and the tilt unit conversion `tilt_tolerance_deg`
- `Sampling.jl`: random misalignment draws within a tolerance — the `RadialDistribution`
  shapes, `sample_disk`, `sample_uncertainty`, and the additive and cylinder tolerance models
- `MonteCarlo.jl`: `run_monte_carlo`, the batched, seeded recombination of a `SensitivityTable`
  with a `ToleranceSet` into intrinsic and corrected `|δ|` histograms
- `Risk.jl`: the ITPA penetration-threshold scalings, `locking_risk` (the overlap distribution
  convolved with the threshold distribution), `tolerance_scan` and `allowable_tolerance`

The stored primitive is the derivative of each coil set's root-area-weighted control-surface
spectrum b̃, not a scalar: the overlap with any dominant mode is linear in b̃, so the ψ_N window,
the mode index, and the field normalization stay post-hoc analysis choices.
"""

using LinearAlgebra
using HDF5
using Printf
using TOML
using Random
import Random: AbstractRNG

import ..Equilibrium
import ..ForcingTerms
import ..ForcingTerms: CoilSet, CoilSetConfig, CoilConfig, ForcingMode, CoilForcingGrid, coil_forcing_modes, apply_transforms
import ..PerturbedEquilibrium
import ..PerturbedEquilibrium: ResonantCoupling, DominantCoupling, dominant_coupling, rootarea_field, coupling_overlap
import ..Utilities

include("ErrorFieldsStructs.jl")
include("Sensitivity.jl")
include("ToleranceTOML.jl")
include("Sampling.jl")
include("MonteCarlo.jl")
include("Risk.jl")
include("Output.jl")

export ErrorFieldsControl, CoilSensitivities, SensitivityTable
export compute_coil_sensitivities, sensitivity_table, cancelling_offset
export ToleranceSet, CoilTolerance, CoherentGroupTolerance, OtherFieldBudget
export read_tolerance_toml, parse_tolerance_toml, validate_tolerances, tilt_tolerance_deg
export RadialDistribution, Flat, UniformArea, Hollow, Ring, PowerLaw, randpow, radial_distribution
export disk_radius, sample_disk, sample_uncertainty, sample_additive, sample_cylinder
export MonteCarloControl, MonteCarloResult, run_monte_carlo
export ThresholdScaling, ITPA_THRESHOLD_SCALINGS, threshold_scaling, ScenarioParameters, nominal_threshold, threshold_samples
export RiskControl, RiskResult, locking_risk, ToleranceScan, tolerance_scan, allowable_tolerance

end # module ErrorFields
