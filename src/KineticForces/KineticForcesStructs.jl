"""
    METHOD_REGISTRY

Single source of truth for the NTV calculation methods. Each entry is a NamedTuple
`(name, flag, kind, doc)`:
- `name`  — short method identifier used as the HDF5 group key and in `intr.method`
- `flag`  — the `KineticForcesControl` field symbol that enables the method
- `kind`  — dispatch routing tag consumed by `method_kind` / `Torque.jl`
            (`:gar` for the GAR/matrix family, `:fcgl`/`:rlar`/`:clar` for the
            three special-cased methods)
- `doc`   — one-line description printed in verbose output

The method names/docs and the `Compute.jl` enable list are all derived from this
tuple, and `Torque.jl` routes on `kind`, so the methods are enumerated in one place. 
To add a method: append an entry here and add the matching `*_flag` field
to `KineticForcesControl`.
"""
const METHOD_REGISTRY = (
    (name="fgar", flag=:fgar_flag, kind=:gar, doc="Full general-aspect-ratio calculation"),
    (name="tgar", flag=:tgar_flag, kind=:gar, doc="Trapped particle general-aspect-ratio calculation"),
    (name="pgar", flag=:pgar_flag, kind=:gar, doc="Passing particle general-aspect-ratio calculation"),
    (name="rlar", flag=:rlar_flag, kind=:rlar, doc="Trapped particle large-aspect-ratio calculation"),
    (name="clar", flag=:clar_flag, kind=:clar, doc="Trapped particle cylindrical large-aspect-ratio calculation"),
    (name="fcgl", flag=:fcgl_flag, kind=:fcgl, doc="Fluid Chew–Goldberger–Low calculation"),
    (name="fwmm", flag=:fwmm_flag, kind=:gar, doc="Full energy calculation using MXM Euler–Lagrange matrix"),
    (name="twmm", flag=:twmm_flag, kind=:gar, doc="Trapped energy calculation using MXM Euler–Lagrange matrix"),
    (name="pwmm", flag=:pwmm_flag, kind=:gar, doc="Passing energy calculation using MXM Euler–Lagrange matrix"),
    (name="ftmm", flag=:ftmm_flag, kind=:gar, doc="Full torque calculation using MXM Euler–Lagrange matrix"),
    (name="ttmm", flag=:ttmm_flag, kind=:gar, doc="Trapped torque calculation using MXM Euler–Lagrange matrix"),
    (name="ptmm", flag=:ptmm_flag, kind=:gar, doc="Passing torque calculation using MXM Euler–Lagrange matrix"),
    (name="fkmm", flag=:fkmm_flag, kind=:gar, doc="Full MXM Euler–Lagrange energy matrix norm calculation"),
    (name="tkmm", flag=:tkmm_flag, kind=:gar, doc="Trapped MXM Euler–Lagrange energy matrix norm calculation"),
    (name="pkmm", flag=:pkmm_flag, kind=:gar, doc="Passing MXM Euler–Lagrange energy matrix norm calculation"),
    (name="frmm", flag=:frmm_flag, kind=:gar, doc="Full MXM Euler–Lagrange torque matrix norm calculation"),
    (name="trmm", flag=:trmm_flag, kind=:gar, doc="Trapped MXM Euler–Lagrange torque matrix norm calculation"),
    (name="prmm", flag=:prmm_flag, kind=:gar, doc="Passing MXM Euler–Lagrange torque matrix norm calculation"))

"""
    method_kind(name) -> Symbol

Return the dispatch routing tag (`:gar`, `:fcgl`, `:rlar`, `:clar`) for the NTV
method `name`, looked up from [`METHOD_REGISTRY`]. Errors on an unknown method.
"""
function method_kind(name::AbstractString)
    for entry in METHOD_REGISTRY
        entry.name == name && return entry.kind
    end
    error("ERROR: torque - unknown method: $name")
end


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

    # Tolerances — debug defaults looser than Fortran PENTRC.
    # *_xlmda: shared tolerances for inner λ (pitch) and x (energy) integrations
    # *_psi:   tolerances for outer ψ quadrature
    atol_xlmda::Float64 = 1e-8     # Absolute tolerance for inner pitch + energy integrations
    rtol_xlmda::Float64 = 1e-5     # Relative tolerance for inner pitch + energy integrations
    # atol_psi=1e-2 N·m is small compared to typical tokamak torques (1-10 N·m) — one
    # of the main benefits of QuadGK over ODE is that tolerances are physically intuitive.
    atol_psi::Float64 = 1e-2       # Absolute tolerance for outer ψ quadrature
    rtol_psi::Float64 = 1e-2       # Relative tolerance for outer ψ quadrature

    # Scaling factors
    density_factor::Float64 = 1.0            # Density scaling (ni, ne)
    temperature_factor::Float64 = 1.0        # Temperature scaling (Ti, Te)
    ExB_rotation_factor::Float64 = 1.0       # ExB rotation scaling (omegaE)
    wdfac::Float64 = 1.0                     # Magnetic drift scaling
    toroidal_rotation_factor::Float64 = 1.0  # Total toroidal rotation scaling (wphi)

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
    save_records::Bool = false      # Save detailed integration trajectories

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
- `dbob_m`, `divx_m`: Perturbation mode interpolants

Equilibrium and kinetic profile data are read directly from the
`PlasmaEquilibrium` (`equil.profiles`, `equil.geometry`) and the
externally-loaded `KineticProfileSplines` — no shadow copies are kept on
this struct.
"""
@kwdef mutable struct KineticForcesInternal
    # Equilibrium-derived quantities
    ro::Float64 = 0.0              # Major radius [m]
    bo::Float64 = 0.0              # Toroidal field on axis [T]
    chi1::Float64 = 0.0            # 2π * ψ_o (flux normalization)
    mthsurf::Int = 0               # Number of poloidal grid points
    mfac::Vector{Int} = Int[]      # Poloidal mode numbers [mlow:mhigh]
    # Multi-n mode indexing (matching ForceFreeStatesInternal)
    nlow::Int = 0                  # Lowest toroidal mode number
    nhigh::Int = 0                 # Highest toroidal mode number
    npert::Int = 0                 # Number of toroidal modes
    mlow::Int = 0                  # Lowest poloidal mode number
    mhigh::Int = 0                 # Highest poloidal mode number
    mpert::Int = 0                 # Number of poloidal modes
    numpert_total::Int = 0         # Total modes: mpert × npert

    # Perturbation mode interpolants (populated from PerturbedEquilibriumState)
    dbob_m::Any = nothing          # δB/B perturbation modes (CubicSeriesInterpolant)
    divx_m::Any = nothing          # ∇·ξ⊥ perturbation modes (CubicSeriesInterpolant)

    # Sticky bracket-search hints for amortized lookup in `tpsi!`. When the outer
    # quadrature evaluates ψ non-monotonically, FastInterpolations falls back to a
    # broader search when the hint is stale — still a net win over a cold bracket
    # search on every call.
    dbob_m_hint::Base.RefValue{Int} = Ref(1)
    divx_m_hint::Base.RefValue{Int} = Ref(1)
    # 2D hints for `equil.eqfun_B` and `equil.rzphi_jac` evaluated on the
    # (ψ,θ) grid inside the θ loops of `tpsi!` / `calculate_fcgl`. Each call
    # site gets its own hint tuple so the ψ index is sticky across the
    # quadrature evaluation while the θ index updates monotonically within each θ loop.
    hint2d_eqfun_B::Tuple{Base.RefValue{Int},Base.RefValue{Int}} = (Ref(1), Ref(1))
    hint2d_rzphi_jac::Tuple{Base.RefValue{Int},Base.RefValue{Int}} = (Ref(1), Ref(1))

    # Upper ψ bound set by DCON/FFS (from ForceFreeStatesInternal.psilim).
    # The perturbation interpolants are only valid on [0, psilim]; extrapolation
    # beyond diverges. The outer ψ quadrature clips to this to match Fortran PENTRC.
    psilim::Float64 = 1.0

    # Pre-allocated θ-grid buffers for `tpsi!` — length mthsurf+1, reused per evaluation.
    tpsi_xs::Vector{Float64} = Float64[]
    tpsi_B::Vector{Float64} = Float64[]
    tpsi_dBdpsi::Vector{Float64} = Float64[]
    tpsi_dBdtheta::Vector{Float64} = Float64[]
    tpsi_jac::Vector{Float64} = Float64[]
    tpsi_djdpsi::Vector{Float64} = Float64[]

    # Raw geometric matrices for kinetic W vector construction
    # (Fortran dcon_interface.f fmodb s/t/x/y/z — NOT the DCON a/b/c/d/e/h matrices)
    smats::Any = nothing           # CubicSeriesInterpolant, mpert² series over ψ
    tmats::Any = nothing
    xmats::Any = nothing
    ymats::Any = nothing
    zmats::Any = nothing

    # Clebsch displacement vectors for mode-coupled dW contraction
    xs_m::Any = nothing            # Vector of 3 CubicSeriesInterpolants: [ξ_ψ, ξ_+, ξ_-]

    # Integration results
    tphi::ComplexF64 = 0.0 + 0.0im    # Total torque/energy
    tsurf::ComplexF64 = 0.0 + 0.0im   # Surface torque/energy
    teq::ComplexF64 = 0.0 + 0.0im     # Equilibrium grid result

    # Method tracking — available methods/docs come from `METHOD_REGISTRY`.
    method::String = ""
    wtw::Array{ComplexF64,3} = Array{ComplexF64}(undef, 0, 0, 0)

    verbose::Bool = false
end

"""
    KineticForcesInternal(equil; verbose=false)

Construct KineticForcesInternal from a PlasmaEquilibrium, extracting
the equilibrium geometry parameters needed for NTV calculations.
"""
function KineticForcesInternal(equil; verbose::Bool=false)
    mthsurf = length(equil.rzphi_ys) - 1
    nth = mthsurf + 1
    KineticForcesInternal(;
        ro      = equil.ro,
        bo      = equil.params.b0,
        chi1    = 2π * equil.psio,
        mthsurf,
        tpsi_xs       = collect(range(0.0, 1.0, length=nth)),
        tpsi_B        = Vector{Float64}(undef, nth),
        tpsi_dBdpsi   = Vector{Float64}(undef, nth),
        tpsi_dBdtheta = Vector{Float64}(undef, nth),
        tpsi_jac      = Vector{Float64}(undef, nth),
        tpsi_djdpsi   = Vector{Float64}(undef, nth),
        verbose,
    )
end

"""
    set_perturbation_data!(kf_intr, pe_state, ffs_intr, equil, metric)

Populate perturbation data from PerturbedEquilibriumState into KineticForcesInternal.

Builds three interpolant sets from PE Clebsch displacements:
1. `xs_m` — [ξ^ψ, ∂ξ^ψ/∂ψ, ξ^α] CubicSeriesInterpolants over ψ
2. `dbob_m` — δB/B Fourier modes via JBB deweighting (Fortran set_peq)
3. `divx_m` — ∇·ξ⊥ Fourier modes via JBB deweighting

The JBB deweighting algorithm (Fortran pentrc/inputs.f90:828-868):
1. Apply geometric matrices S,T,X,Y,Z in m-space
2. Inverse DFT to θ-space
3. Divide by J·B² at each θ
4. Forward DFT back to m-space
"""
function set_perturbation_data!(kf_intr::KineticForcesInternal, pe_state, ffs_intr,
                                equil::Equilibrium.PlasmaEquilibrium,
                                metric::ForceFreeStates.MetricData)
    # Copy mode numbers from FFS
    kf_intr.mlow = ffs_intr.mlow
    kf_intr.mhigh = ffs_intr.mhigh
    kf_intr.mpert = ffs_intr.mpert
    kf_intr.nlow = ffs_intr.nlow
    kf_intr.nhigh = ffs_intr.nhigh
    kf_intr.npert = ffs_intr.npert
    kf_intr.numpert_total = ffs_intr.numpert_total
    kf_intr.mfac = collect(ffs_intr.mlow:ffs_intr.mhigh)
    kf_intr.psilim = ffs_intr.psilim

    # Bail if no xi_modes available (PE didn't run or failed)
    if pe_state.xi_modes === nothing || isempty(pe_state.psi_grid)
        @warn "set_perturbation_data!: no xi_modes available, skipping perturbation build"
        return
    end

    xi_modes = pe_state.xi_modes
    psi_grid = pe_state.psi_grid
    npsi = length(psi_grid)
    mpert = ffs_intr.mpert
    chi1 = kf_intr.chi1

    # Build xs_m: 3 CubicSeriesInterpolants from Clebsch displacement matrices
    # xs_m[1] = ξ^ψ (unregularized), xs_m[2] = ∂ξ^ψ/∂ψ (regularized), xs_m[3] = ξ^α
    # Note: clebsch_alpha is stored as ξ^α/χ₁, multiply by chi1 to get ξ^α
    itp_opts = (; extrap=ExtendExtrap())
    xs_m_1 = cubic_interp(psi_grid, Series(xi_modes.clebsch_psi); itp_opts...)
    xs_m_2 = cubic_interp(psi_grid, Series(xi_modes.clebsch_psi1); itp_opts...)
    clebsch_alpha_raw = xi_modes.clebsch_alpha .* chi1
    xs_m_3 = cubic_interp(psi_grid, Series(clebsch_alpha_raw); itp_opts...)
    kf_intr.xs_m = [xs_m_1, xs_m_2, xs_m_3]

    # Build geometric matrices (S,T,X,Y,Z) for JBB deweighting
    geom_mats = ForceFreeStates.build_kinetic_metric_matrices(equil, ffs_intr, metric)

    # Build FourierTransform for the JBB deweighting DFT round-trip
    mthsurf = kf_intr.mthsurf
    ft = Utilities.FourierTransforms.FourierTransform(mthsurf, mpert, ffs_intr.mlow)

    # JBB deweighting: convert Clebsch modes → physical δB/B and ∇·ξ⊥ modes
    dbob_m_data = zeros(ComplexF64, npsi, mpert)
    divx_m_data = zeros(ComplexF64, npsi, mpert)

    # Pre-allocate buffers
    smat_flat = Vector{ComplexF64}(undef, mpert^2)
    tmat_flat = Vector{ComplexF64}(undef, mpert^2)
    xmat_flat = Vector{ComplexF64}(undef, mpert^2)
    ymat_flat = Vector{ComplexF64}(undef, mpert^2)
    zmat_flat = Vector{ComplexF64}(undef, mpert^2)
    jbb_kapx = Vector{ComplexF64}(undef, mpert)
    jbb_divx = Vector{ComplexF64}(undef, mpert)
    jbb_dbob = Vector{ComplexF64}(undef, mpert)
    theta_buf = Vector{ComplexF64}(undef, mthsurf)

    hint_s = Ref(1)
    hint_t = Ref(1)
    hint_x = Ref(1)
    hint_y = Ref(1)
    hint_z = Ref(1)

    for ipsi in 1:npsi
        psi = psi_grid[ipsi]

        # Get Clebsch displacement vectors at this ψ
        xsp  = view(xi_modes.clebsch_psi,  ipsi, :)       # ξ^ψ [mpert]
        xmp1 = view(xi_modes.clebsch_psi1, ipsi, :)       # ∂ξ^ψ/∂ψ [mpert]
        xms  = view(clebsch_alpha_raw, ipsi, :)            # ξ^α [mpert]

        # Evaluate geometric matrices at ψ → mpert² flat vectors, reshape to mpert×mpert
        geom_mats.smats(smat_flat, psi; hint=hint_s)
        geom_mats.tmats(tmat_flat, psi; hint=hint_t)
        geom_mats.xmats(xmat_flat, psi; hint=hint_x)
        geom_mats.ymats(ymat_flat, psi; hint=hint_y)
        geom_mats.zmats(zmat_flat, psi; hint=hint_z)

        smat = reshape(smat_flat, mpert, mpert)
        tmat = reshape(tmat_flat, mpert, mpert)
        xmat = reshape(xmat_flat, mpert, mpert)
        ymat = reshape(ymat_flat, mpert, mpert)
        zmat = reshape(zmat_flat, mpert, mpert)

        # Apply geometric matrices in m-space (Fortran set_peq lines 854-857).
        # Fortran xs_m(1)=∂ξ^ψ/∂ψ (xmp1), xs_m(2)=ξ^ψ (xsp), xs_m(3)=ξ^α (xms).
        #   jbb_kapx = smat · xs_m(2) + tmat · xs_m(3) = smat·xsp + tmat·xms
        #   jbb_divx = xmat · xs_m(1) + ymat · xs_m(2) + zmat · xs_m(3)
        #            = xmat·xmp1 + ymat·xsp + zmat·xms
        mul!(jbb_kapx, smat, xsp)
        mul!(jbb_kapx, tmat, xms, 1.0 + 0.0im, 1.0 + 0.0im)   # += tmat * xms
        mul!(jbb_divx, xmat, xmp1)
        mul!(jbb_divx, ymat, xsp,  1.0 + 0.0im, 1.0 + 0.0im)  # += ymat * xsp
        mul!(jbb_divx, zmat, xms,  1.0 + 0.0im, 1.0 + 0.0im)  # += zmat * xms
        @. jbb_dbob = -(jbb_divx + jbb_kapx)

        # Inverse DFT to θ-space, divide by J·B², forward DFT back
        _jbb_deweight!(view(dbob_m_data, ipsi, :), jbb_dbob, ft, psi, equil, mthsurf, theta_buf)
        _jbb_deweight!(view(divx_m_data, ipsi, :), jbb_divx, ft, psi, equil, mthsurf, theta_buf)
    end

    # Build CubicSeriesInterpolants over ψ for dbob_m and divx_m
    kf_intr.dbob_m = cubic_interp(psi_grid, Series(dbob_m_data); itp_opts...)
    kf_intr.divx_m = cubic_interp(psi_grid, Series(divx_m_data); itp_opts...)

    if kf_intr.verbose
        @info "set_perturbation_data!: built dbob_m/divx_m/xs_m interpolants " *
              "(npsi=$npsi, mpert=$mpert)"
    end
end

"""
    _jbb_deweight!(out, jbb_modes, ft, psi, equil, mthsurf, theta_buf)

JBB deweighting step: inverse DFT → divide by J(ψ,θ)·B(ψ,θ)² → forward DFT.

Matches Fortran set_peq lines 859-868: transforms JBB-weighted m-space data
to θ-space, removes the J·B² weighting at each poloidal angle, and transforms back.
"""
function _jbb_deweight!(out::AbstractVector{ComplexF64}, jbb_modes::Vector{ComplexF64},
                        ft::Utilities.FourierTransforms.FourierTransform,
                        psi::Float64, equil::Equilibrium.PlasmaEquilibrium,
                        mthsurf::Int, theta_buf::Vector{ComplexF64})
    # Inverse DFT: m-space → θ-space
    theta_buf .= Utilities.FourierTransforms.inverse(ft, jbb_modes)

    # Divide by J·B² at each θ point
    for j in 1:mthsurf
        theta_norm = (j - 1) / mthsurf   # θ ∈ [0, 1)
        pt = (psi, theta_norm)
        jac = equil.rzphi_jac(pt)
        B = equil.eqfun_B(pt)
        theta_buf[j] /= jac * B^2
    end

    # Forward DFT: θ-space → m-space
    out .= ft(theta_buf)
    return nothing
end


# ============================================================================
# Result Structs
# ============================================================================

"""
    EnergyIntegrationResult

Results from a single energy-space integration at one (ψ, λ, ℓ) point.
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
    torque_profile::Any = nothing     # Interpolant of dT/dψ(ψ) from ψ integration
    total_torque::ComplexF64 = 0.0 + 0.0im
    total_energy::ComplexF64 = 0.0 + 0.0im
    records::Vector{EnergyIntegrationResult} = EnergyIntegrationResult[]
    # Per-step ψ profile from outer quadrature
    psi_grid::Vector{Float64} = Float64[]
    dtdpsi::Vector{ComplexF64} = ComplexF64[]
    t_cumulative::Vector{ComplexF64} = ComplexF64[]
    psi_nsteps::Int = 0
end

"""
    KineticForcesState

Accumulated results from all KineticForces computations.
Written to gpec.h5 under the "kinetic_forces" group.
"""
@kwdef mutable struct KineticForcesState
    method_results::Dict{String, MethodResult} = Dict{String, MethodResult}()
    # Block-diagonal kinetic matrices: key=method, value=(numpert_total, numpert_total, 6)
    kinetic_matrices::Dict{String, Array{ComplexF64,3}} = Dict{String, Array{ComplexF64,3}}()
    completed::Bool = false
end
