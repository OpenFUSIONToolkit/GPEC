"""
    ErrorFieldsControl

Control parameters for the `[ErrorFields]` TOML section. The sweep needs a coil-forced
perturbed-equilibrium solve (`forcing_data_format = "coil"` with the singular-coupling step on);
everything here concerns only how the coil linearization is taken and where it is written.
Analysis choices (which resonant surfaces count, which singular mode, which normalization) are
never inputs: the full spectra are stored and projected post hoc with [`sensitivity_table`](@ref).

## Fields

  - `fd_step_shift_m`: central-difference step for the rigid shifts, in metres
  - `fd_step_tilt_deg`: central-difference step for the rigid tilts, in degrees
  - `rotation_center`: pivot of the tilt taps — `"conductor"` rotates every conductor of a set
    about its own arc-length centre (the Fortran `coil_read` convention the OMFIT tolerance tool
    inherited), `"set"` rotates the whole set rigidly about its common arc-length centre (a
    winding pack moving as one body, which is what an axis-line tolerance constrains)
  - `tolerance_file`: manufacturing-tolerance TOML (see `ToleranceTOML`), relative to the run
    directory; empty means none. It is validated against the run's coil sets and echoed into
    `Input/RawInputs/ErrorFields/tolerance_toml_raw`
  - `output_filename`: HDF5 file the results are appended to; empty means the run's main output
  - `write_outputs_to_HDF5`: write `ErrorFields/CoilSensitivities/` when true
  - `verbose`: log per-coil-set progress and the linearity diagnostics
"""
Base.@kwdef struct ErrorFieldsControl
    fd_step_shift_m::Float64 = 1e-3
    fd_step_tilt_deg::Float64 = 0.1
    rotation_center::String = "conductor"
    tolerance_file::String = ""
    output_filename::String = ""
    write_outputs_to_HDF5::Bool = true
    verbose::Bool = false
end

"""
    CoilSensitivities

Linearization of each coil set's control-surface spectrum with respect to its six rigid-body
degrees of freedom, on the (m, n) column ordering of the [`ResonantCoupling`](@ref) it was built
against. Spectra are root-area-weighted fields b̃ in tesla, the vector the coupling matrix and
its singular vectors act on, so projecting any of them onto a mode is one inner product.

Shifts are Cartesian translations of the whole set; tilts are rotations about the machine
x, y, z axes through the pivot selected by `ErrorFieldsControl.rotation_center`. A coil set's
spectrum is proportional to its currents, so the table is specific to the current pattern it
was evaluated with; `peak_current` and `winding_multiplier` let a user renormalize per ampere-turn.

## Fields

  - `coil_names`: name of each coil set `[ncoil_set]`
  - `m_modes`, `n_modes`: poloidal and toroidal mode number of each spectrum entry `[numpert_total]`
  - `b_t0`: toroidal field magnitude on axis, tesla — the normalization that makes the overlap a
    dimensionless δ
  - `nominal_field`: b̃ of each set as built, tesla `[numpert_total × ncoil_set]`
  - `shift_sensitivity`: ∂b̃/∂(Δx, Δy, Δz), tesla per metre `[numpert_total × 3 × ncoil_set]`
  - `tilt_sensitivity`: ∂b̃/∂(θx, θy, θz), tesla per degree `[numpert_total × 3 × ncoil_set]`
  - `shift_linearity_residual`, `tilt_linearity_residual`: ‖b̃(+h) + b̃(−h) − 2b̃(0)‖ of each tap
    relative to the set's largest first difference ‖b̃(+h) − b̃(−h)‖ over all six taps, a
    window-independent measure of how much second-order response the linearization drops at
    the step taken `[3 × ncoil_set]`
  - `peak_current`: largest conductor current magnitude of each set, amperes `[ncoil_set]`
  - `winding_multiplier`: turns per conductor element of each set `[ncoil_set]`
"""
struct CoilSensitivities
    coil_names::Vector{String}
    m_modes::Vector{Int}
    n_modes::Vector{Int}
    b_t0::Float64
    nominal_field::Matrix{ComplexF64}
    shift_sensitivity::Array{ComplexF64,3}
    tilt_sensitivity::Array{ComplexF64,3}
    shift_linearity_residual::Matrix{Float64}
    tilt_linearity_residual::Matrix{Float64}
    peak_current::Vector{Float64}
    winding_multiplier::Vector{Float64}
end

"""
    SensitivityTable

A [`CoilSensitivities`](@ref) projected onto one singular mode of a [`DominantCoupling`](@ref)
and normalized by the axis toroidal field: the dimensionless overlap δ of each coil set and its
derivatives, which is all a tolerance Monte Carlo consumes. Built by [`sensitivity_table`](@ref).

With `S = (S_x, S_y)` the in-plane shift sensitivities, a rigid shift `(Δx, Δy)` moves the overlap
by `S_x·Δx + S_y·Δy`; the same holds for the tilts. Singular vectors carry an arbitrary phase, so
only magnitudes and relative phases within one table are meaningful.

## Fields

  - `coil_names`: name of each coil set `[ncoil_set]`
  - `mode`: index of the singular mode projected onto (1 is the dominant mode)
  - `delta_nominal`: overlap of each set as built `[ncoil_set]`
  - `shift`: ∂δ/∂(Δx, Δy, Δz), per metre `[3 × ncoil_set]`
  - `tilt`: ∂δ/∂(θx, θy, θz), per degree `[3 × ncoil_set]`
  - `shift_rms`, `tilt_rms`: direction-averaged in-plane magnitude `√((|S_x|² + |S_y|²)/2)` of the
    shift and tilt sensitivities `[ncoil_set]` — the single-number sensitivity the OMFIT tool used
  - `cancelling_shift`: the in-plane shift `(Δx, Δy)` that cancels `delta_nominal`, metres
    `[2 × ncoil_set]`, see [`cancelling_offset`](@ref)
  - `cancelling_tilt`: the in-plane tilt `(θx, θy)` that cancels `delta_nominal`, degrees `[2 × ncoil_set]`
"""
struct SensitivityTable
    coil_names::Vector{String}
    mode::Int
    delta_nominal::Vector{ComplexF64}
    shift::Matrix{ComplexF64}
    tilt::Matrix{ComplexF64}
    shift_rms::Vector{Float64}
    tilt_rms::Vector{Float64}
    cancelling_shift::Matrix{Float64}
    cancelling_tilt::Matrix{Float64}
end
