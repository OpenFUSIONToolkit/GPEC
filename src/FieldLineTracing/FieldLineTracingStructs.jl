"""
    FieldLineTracingControl

User-facing configuration for the `[FieldLineTracing]` section of `gpec.toml`.

Field-line tracing integrates magnetic field lines through the GPEC-computed field to
reveal 3D topology: magnetic islands at rational surfaces, stochastic layers where
islands overlap, connection lengths, and divertor footprints (the diagnostics common to
TRIP3D and MAFOT). The traced field is assembled from the equilibrium background plus a
selectable perturbation source.

## Fields

  - `tracing_coords::String`: Integration coordinate system.
      + `"flux"` (default): straight-field-line `(ψ,θ,ζ)`, interior only (`ψ ≤ ψ_control`).
        Uses GPEC's exact contravariant perturbed field; best island fidelity.
      + `"real"`: real-space cylindrical `(R,Z,φ)` (TRIP3D/MAFOT-style). Covers both
        inside and outside the LCFS/control surface.
  - `tracing_field::String`: Perturbation source.
      + `"total"` (default): applied field with plasma response, `Φ_tot = P·Φ_x`
        (ideal-screened at rationals).
      + `"vacuum"`: externally applied field only, `Φ_x` (unscreened — shows vacuum islands).
      + `"plasma"`: plasma-response part only (`total − vacuum`).
  - `region::String`: Where to launch/trace in `"real"` mode — `"inside"`, `"outside"`,
    or `"both"` (relative to the control surface). Ignored in `"flux"` mode.
  - `n_lines::Int`: Number of field lines launched.
  - `psi_start`, `psi_end`: Launch range in normalized flux (`"flux"` mode). Lines are
    seeded on a `ψ` grid at `θ = 0`.
  - `R_start`, `R_end`, `Z_launch`: Launch chord in metres (`"real"` mode). Lines are
    seeded along `R ∈ [R_start, R_end]` at `Z = Z_launch` (default: magnetic-axis height).
  - `n_transits::Int`: Number of toroidal transits (`ζ` or `φ` periods) integrated per line.
  - `phi_planes::Vector{Float64}`: Toroidal angles [rad] of the Poincaré section(s).
  - `tol::Float64`: Relative ODE tolerance (mirrors `eulerlagrange_tolerance`).
  - `flux_n_max::Int`: Highest toroidal harmonic in the `vacuum` flux drive (the coil field is
    fully 3D). Keeping `n = 1…flux_n_max` captures island bifurcation where `(2,1)` and `(4,2)`
    both resonate at `q=2`. `total`/`plasma` are single-`n` (the perturbed equilibrium is one `n`).
  - `compute_connection_length::Bool`: Produce the MAFOT-style connection-length/laminar map.
  - `compute_footprints::Bool`: Produce divertor/wall strike-point footprints (`"real"` mode).
  - `compute_island_width::Bool`: Auto-detect island half-widths and Chirikov overlap.
  - `write_outputs_to_HDF5::Bool`: Append results to the HDF5 output file.
  - `HDF5_filename::String`: Output filename (shared `gpec.h5`).
  - `verbose::Bool`: Verbose progress logging.
"""
Base.@kwdef mutable struct FieldLineTracingControl
    tracing_coords::String = "flux"
    tracing_field::String = "total"
    region::String = "inside"

    n_lines::Int = 40
    psi_start::Float64 = 0.05
    psi_end::Float64 = 0.98

    R_start::Float64 = 0.0
    R_end::Float64 = 0.0
    Z_launch::Float64 = NaN

    n_transits::Int = 400
    phi_planes::Vector{Float64} = [0.0]
    tol::Float64 = 1e-7

    # Highest toroidal harmonic carried by the vacuum flux drive. The coil field is fully 3D,
    # so `vacuum` flux tracing includes n = 1…flux_n_max to capture island bifurcation (both
    # (m,n)=(2,1) and (4,2) resonate at q=2). total/plasma stay single-n (perturbed eq. is one n).
    flux_n_max::Int = 3

    compute_connection_length::Bool = true
    compute_footprints::Bool = true
    compute_island_width::Bool = true

    write_outputs_to_HDF5::Bool = true
    HDF5_filename::String = "gpec.h5"
    verbose::Bool = true
end

"""
    FieldLineTracingInternal

Internal working state for a field-line-tracing run.

## Fields

  - `dir_path::String`: Working directory (for locating the output file).
  - `field_model::Any`: The assembled `FieldModel` (see `FieldModel.jl`); `nothing` until built.
  - `wall_R`, `wall_Z::Vector{Float64}`: Wall polygon in `(R,Z)` [m] for footprint detection
    (empty when no wall is available).
"""
Base.@kwdef mutable struct FieldLineTracingInternal
    dir_path::String = "./"
    field_model::Any = nothing
    wall_R::Vector{Float64} = Float64[]
    wall_Z::Vector{Float64} = Float64[]
end

"""
    FieldLineTracingState

Results container written to the `field_line_tracing/` HDF5 group.

## Fields

Metadata:
  - `tracing_coords::String`, `tracing_field::String`: Echo of the run configuration.
  - `nn::Int`: Toroidal mode number traced.

Poincaré punctures (flat, one entry per puncture; `line_id` groups them by launched line):
  - `punc_R`, `punc_Z::Vector{Float64}`: Puncture `(R,Z)` [m].
  - `punc_psi`, `punc_theta::Vector{Float64}`: Puncture `(ψ_N, θ)` (θ normalized; `NaN` outside).
  - `punc_line::Vector{Int}`: Source line index for each puncture.
  - `punc_phi::Vector{Float64}`: Section angle [rad] each puncture belongs to.

Connection-length / laminar map (one entry per launch point):
  - `cl_R`, `cl_Z::Vector{Float64}`: Launch point in metres (or mapped from a flux launch).
  - `cl_psi::Vector{Float64}`: Launch `ψ_N`.
  - `cl_length::Vector{Float64}`: Connection length in metres, or toroidal transits in flux mode, until termination.
  - `cl_class::Vector{Int}`: Topology class (0 laminar/closed, 1 stochastic, 2 wall-terminated).

Footprints (wall strike points; `"real"` mode):
  - `fp_R`, `fp_Z`, `fp_phi::Vector{Float64}`: Strike coordinates [m, m, rad].

Island diagnostics (one entry per resonant surface):
  - `island_m`, `island_n::Vector{Int}`: Resonant poloidal/toroidal mode numbers.
  - `island_psi::Vector{Float64}`: Rational-surface `ψ_N`.
  - `island_halfwidth::Vector{Float64}`: Island half-width in `ψ_N`.
  - `island_chirikov::Vector{Float64}`: Chirikov overlap with the next island outward.
"""
Base.@kwdef mutable struct FieldLineTracingState
    tracing_coords::String = "flux"
    tracing_field::String = "total"
    nn::Int = 0

    punc_R::Vector{Float64} = Float64[]
    punc_Z::Vector{Float64} = Float64[]
    punc_psi::Vector{Float64} = Float64[]
    punc_theta::Vector{Float64} = Float64[]
    punc_line::Vector{Int} = Int[]
    punc_phi::Vector{Float64} = Float64[]

    cl_R::Vector{Float64} = Float64[]
    cl_Z::Vector{Float64} = Float64[]
    cl_psi::Vector{Float64} = Float64[]
    cl_length::Vector{Float64} = Float64[]
    cl_class::Vector{Int} = Int[]

    fp_R::Vector{Float64} = Float64[]
    fp_Z::Vector{Float64} = Float64[]
    fp_phi::Vector{Float64} = Float64[]

    island_m::Vector{Int} = Int[]
    island_n::Vector{Int} = Int[]
    island_psi::Vector{Float64} = Float64[]
    island_halfwidth::Vector{Float64} = Float64[]
    island_chirikov::Vector{Float64} = Float64[]
end
