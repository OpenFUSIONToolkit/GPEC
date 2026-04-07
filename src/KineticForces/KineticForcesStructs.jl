"""
    KineticForcesControl

User-facing control parameters from the TOML `[KineticForces]` section.
Configures which NTV methods to run, species parameters, tolerances, and output options.

Constructed via keyword arguments or from a TOML dict:
```julia
ctrl = KineticForcesControl(; (Symbol(k) => v for (k, v) in inputs["KineticForces"])...)
```
"""
@kwdef mutable struct KineticForcesControl
    # Moment type
    moment::String = "pressure"     # "heat" or "pressure"

    # Method flags (which methods to calculate)
    fgar_flag::Bool = true          # Full general-aspect-ratio
    tgar_flag::Bool = false         # Trapped particle GAR
    pgar_flag::Bool = false         # Passing particle GAR
    rlar_flag::Bool = false         # Trapped particle large-aspect-ratio
    clar_flag::Bool = false         # Trapped cylindrical LAR
    fcgl_flag::Bool = false         # Fluid Chew-Goldberger-Low
    fwmm_flag::Bool = false         # Full energy via MXM E-L matrix
    twmm_flag::Bool = false         # Trapped energy via MXM E-L matrix
    pwmm_flag::Bool = false         # Passing energy via MXM E-L matrix
    ftmm_flag::Bool = false         # Full torque via MXM E-L matrix
    ttmm_flag::Bool = false         # Trapped torque via MXM E-L matrix
    ptmm_flag::Bool = false         # Passing torque via MXM E-L matrix
    fkmm_flag::Bool = false         # Full MXM E-L energy matrix norm
    tkmm_flag::Bool = false         # Trapped MXM E-L energy matrix norm
    pkmm_flag::Bool = false         # Passing MXM E-L energy matrix norm
    frmm_flag::Bool = false         # Full MXM E-L torque matrix norm
    trmm_flag::Bool = false         # Trapped MXM E-L torque matrix norm
    prmm_flag::Bool = false         # Passing MXM E-L torque matrix norm

    # Plasma species parameters
    zi::Int = 1                     # Ion charge (fundamental units)
    mi::Int = 2                     # Ion mass (proton masses)
    zimp::Int = 6                   # Impurity charge
    mimp::Int = 12                  # Impurity mass
    electron::Bool = false          # Include electron contribution

    # Mode numbers
    nn::Int = 1                     # Toroidal mode number
    nl::Int = 1                     # Bounce harmonic number

    # Tolerances
    atol_xlmda::Float64 = 1e-12    # Absolute tolerance for pitch angle integration
    rtol_xlmda::Float64 = 1e-9     # Relative tolerance for pitch angle integration

    # Scaling factors
    nfac::Float64 = 1.0            # Density scaling
    tfac::Float64 = 1.0            # Temperature scaling
    wefac::Float64 = 1.0           # ExB rotation scaling
    wdfac::Float64 = 1.0           # Magnetic drift scaling
    wpfac::Float64 = 1.0           # Indirect rotation scaling
    nufac::Float64 = 1.0           # Collisionality scaling
    divxfac::Float64 = 1.0         # div(xi_perp) scaling

    # Energy integration parameters
    nutype::String = "harmonic"     # Collision operator: "zero", "small", "krook", "harmonic"
    f0type::String = "maxwellian"   # Distribution function: "maxwellian", "jkp", "cgl"

    # Diagnostic parameters
    psilims::Vector{Float64} = [0.0, 1.0]  # Integration limits in psi

    # Diagnostic output flags
    eq_out::Bool = false            # Output equilibrium profiles
    theta_out::Bool = false         # Output theta-dependent quantities
    xlmda_out::Bool = false         # Output pitch angle profiles
    eqpsi_out::Bool = false         # Output equilibrium psi profiles

    # Output configuration
    write_outputs_to_HDF5::Bool = true
    HDF5_filename::String = "gpec.h5"
    save_records::Bool = false      # Save detailed ODE integration trajectories

    # Input files (relative to dir_path)
    kinetic_file::String = "kinetic.dat"
    gpec_file::String = "gpec.h5"

    # Diagnostic flags
    fnml_flag::Bool = false         # Fourier mode coupling diagnostics
    ellip_flag::Bool = false        # Elliptic integral diagnostics
    diag_flag::Bool = false         # General diagnostics
    force_xialpha::Bool = false     # Force xi_alpha formulation

    # Debugging
    verbose::Bool = false
end


# ============================================================================
# Internal State/Working Variables
# ============================================================================
"""
    KineticForcesInternal

Internal working state for KineticForces calculations.
Holds equilibrium-derived quantities, profile interpolants, and integration results.

Fields replacing former module-level globals:
- `ro`, `bo`, `chi1`: Equilibrium geometry parameters
- `mthsurf`, `mfac`: Poloidal grid info
- `sq`, `kin`, `geom`: Profile interpolants
- `dbob_m`, `divx_m`: Perturbation mode interpolants
"""
@kwdef mutable struct KineticForcesInternal
    dir_path::String = ""

    # Equilibrium-derived quantities (replaces former globals)
    ro::Float64 = 0.0              # Major radius [m]
    bo::Float64 = 0.0              # Toroidal field on axis [T]
    chi1::Float64 = 0.0            # 2π * ψ_o (flux normalization)
    mthsurf::Int = 0               # Number of poloidal grid points
    mfac::Vector{Int} = Int[]      # Poloidal mode numbers
    nn::Int = 0                    # Toroidal mode number

    # Profile interpolants (populated by initialize_from_equilibrium!)
    sq::Any = nothing              # Safety factor + flux profiles
    kin::Any = nothing             # Kinetic profiles (n, T per species)
    geom::Any = nothing            # Geometric profiles
    dbob_m::Any = nothing          # δB/B perturbation modes
    divx_m::Any = nothing          # ∇·ξ⊥ perturbation modes

    # Integration results
    tphi::ComplexF64 = 0.0 + 0.0im    # Total torque/energy
    tsurf::ComplexF64 = 0.0 + 0.0im   # Surface torque/energy
    teq::ComplexF64 = 0.0 + 0.0im     # Equilibrium grid result

    # Grid
    grid_type::String = "equil"
    psi_grid::Vector{Float64} = Float64[]

    # Method tracking
    method::String = ""
    wtw::Array{ComplexF64,3} = Array{ComplexF64}(undef, 0, 0, 0)
    psi_out_valid::Vector{Float64} = Float64[]
    nvalid::Int = 0

    # Status
    flags::Vector{Bool} = fill(false, 18)
    methods::Vector{String} = [
        "fgar", "tgar", "pgar", "rlar", "clar", "fcgl",
        "fwmm", "twmm", "pwmm", "ftmm", "ttmm", "ptmm",
        "fkmm", "tkmm", "pkmm", "frmm", "trmm", "prmm"]
    docs::Vector{String} = [
        "Full general-aspect-ratio calculation",
        "Trapped particle general-aspect-ratio calculation",
        "Passing particle general-aspect-ratio calculation",
        "Trapped particle large-aspect-ratio calculation",
        "Trapped particle cylindrical large-aspect-ratio calculation",
        "Fluid Chew–Goldberger–Low calculation",
        "Full energy calculation using MXM Euler–Lagrange matrix",
        "Trapped energy calculation using MXM Euler–Lagrange matrix",
        "Passing energy calculation using MXM Euler–Lagrange matrix",
        "Full torque calculation using MXM Euler–Lagrange matrix",
        "Trapped torque calculation using MXM Euler–Lagrange matrix",
        "Passing torque calculation using MXM Euler–Lagrange matrix",
        "Full MXM Euler–Lagrange energy matrix norm calculation",
        "Trapped MXM Euler–Lagrange energy matrix norm calculation",
        "Passing MXM Euler–Lagrange energy matrix norm calculation",
        "Full MXM Euler–Lagrange torque matrix norm calculation",
        "Trapped MXM Euler–Lagrange torque matrix norm calculation",
        "Passing MXM Euler–Lagrange torque matrix norm calculation"]

    verbose::Bool = false
end


# ============================================================================
# Result Structs
# ============================================================================

"""
    EnergyIntegrationResult

Results from a single energy-space ODE integration at one (ψ, λ, ℓ) point.
Trajectory fields are only populated when `ctrl.save_records=true`.
"""
@kwdef struct EnergyIntegrationResult
    psi::Float64 = 0.0
    lambda::Float64 = 0.0
    ell::Int = 0
    leff::Float64 = 0.0
    torque::ComplexF64 = 0.0 + 0.0im
    kinetic_energy::ComplexF64 = 0.0 + 0.0im
    x_trajectory::Vector{Float64} = Float64[]
    integrand_trajectory::Vector{ComplexF64} = ComplexF64[]
    integral_trajectory::Vector{ComplexF64} = ComplexF64[]
end

"""
    MethodResult

Results for one NTV computation method across all flux surfaces.
"""
@kwdef mutable struct MethodResult
    method::String = ""
    nn::Int = 0
    psi_grid::Vector{Float64} = Float64[]
    torque_vs_psi::Vector{ComplexF64} = ComplexF64[]
    energy_vs_psi::Vector{ComplexF64} = ComplexF64[]
    total_torque::ComplexF64 = 0.0 + 0.0im
    total_energy::ComplexF64 = 0.0 + 0.0im
    records::Vector{EnergyIntegrationResult} = EnergyIntegrationResult[]
end

"""
    KineticForcesState

Accumulated results from all KineticForces computations.
Written to gpec.h5 under the "kinetic_forces" group.
"""
@kwdef mutable struct KineticForcesState
    method_results::Dict{String, MethodResult} = Dict{String, MethodResult}()
    completed::Bool = false
end
