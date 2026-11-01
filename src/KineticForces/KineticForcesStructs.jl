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
Configures which NTV methods to run, whether electrons join the species sum, and tolerances. The species
composition comes from the equilibrium's kinetic data (`equil.kinetic`).

Constructed via keyword arguments or from a TOML dict:

```julia
ctrl = KineticForcesControl(; (Symbol(k) => v for (k, v) in inputs["KineticForces"])...)
```

Immutable: vary a field by building a new control.

## Fields

  - `axis_validity_suppression::Bool` - Suppress the calculated kinetic terms below ψ_c, where the zero-orbit-width
    ordering fails (a thermal orbit width of the widest-orbit species reaches ⟨r⟩; moved clear of rational surfaces),
    rising to full strength at 2ψ_c through a C² envelope. No tuning parameters. Default `true`.
  - Quadrature tolerances form a ladder, outermost in: `rtol_psi` (default 1e-2) ⊃ `rtol_pitch` (default
    `psi_tolerance_margin × rtol_psi` = 1e-3) ⊃ `rtol_x` (default `nested_tolerance_margin × rtol_pitch` = 1e-5).
    Each inner level is resolved one margin tighter than the outer and no further. Set `rtol_pitch`/`rtol_x`
    explicitly to override (see `torque_pitch_rtol`, `matrix_pitch_rtol`).
"""
@kwdef struct KineticForcesControl
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

    # Species summed in the NTV; the composition itself is the equilibrium's (`equil.kinetic`)
    electron::Bool = false          # Add electron NTV in addition to the ion species

    # Mode numbers
    bounce_harmonic_max::Int = 1    # Bounce harmonics summed over ℓ = -max:max

    # Tolerances, outermost to innermost: ψ quadrature ⊃ λ (pitch) ⊃ x (energy). Each level derives from the one
    # enclosing it by a margin unless set explicitly (NaN ⇒ derived), so no integrator chases its integrand's noise.
    atol_pitch::Float64 = 1e-8      # Absolute tolerance for the pitch integration (median-normalized kernel)
    rtol_pitch::Float64 = NaN      # Relative tolerance for the pitch integration (NaN ⇒ psi_tolerance_margin × rtol_psi)
    psi_tolerance_margin::Float64 = 1e-1   # Factor relating the derived pitch tolerance to rtol_psi
    atol_x::Float64 = NaN          # Absolute tolerance for the energy integration (NaN ⇒ derived)
    rtol_x::Float64 = NaN          # Relative tolerance for the energy integration (NaN ⇒ derived)
    # Energy tolerances default to this fraction of the pitch ones (the pitch integrand is itself an energy integral).
    nested_tolerance_margin::Float64 = 1e-2   # Factor relating derived energy tolerances to the pitch ones
    # Primary convergence knob (~2 sig figs matches NTV model validity); the pitch and energy levels derive from it.
    rtol_psi::Float64 = 1e-2       # Relative tolerance for outer ψ quadrature
    # Amplitude-sensitive (NTV ∝ δB²); default 0 (rtol-only). Nonzero is an expert opt-out for near-zero-torque cases and warns when it dominates.
    atol_psi::Float64 = 0.0        # Absolute tolerance for outer ψ quadrature [N·m]
    # Runaway guard for rtol-only control (e.g. sign-cancelling torque density); warns when hit.
    maxevals_psi::Int = 2000       # Max integrand evaluations for outer ψ quadrature

    # Scaling factors
    density_factor::Float64 = 1.0            # Density scaling (ni, ne)
    temperature_factor::Float64 = 1.0        # Temperature scaling (Ti, Te)
    ExB_rotation_factor::Float64 = 1.0       # ExB rotation scaling (omegaE)
    magnetic_drift_factor::Float64 = 1.0 # Magnetic drift scaling
    toroidal_rotation_factor::Float64 = 1.0  # Total toroidal rotation scaling (wphi)

    collisionality_factor::Float64 = 1.0 # Collisionality scaling
    divxi_factor::Float64 = 1.0     # div(xi_perp) scaling
    axis_validity_suppression::Bool = true  # see the `## Fields` docstring

    # Energy integration parameters
    nutype::String = "harmonic"     # Collision operator: "zero", "small", "krook", "harmonic"
    f0type::String = "maxwellian"   # Distribution function: "maxwellian", "park", "cgl"

    # Diagnostic parameters
    psilims::Vector{Float64} = [0.0, 1.0]  # Integration limits in psi
end

"""
    torque_pitch_rtol(ctrl::KineticForcesControl) -> Float64

Relative tolerance of the pitch (λ) quadrature inside the ψ torque quadrature: `ctrl.rtol_pitch` when set,
otherwise `ctrl.psi_tolerance_margin × ctrl.rtol_psi`, so the λ integrand is resolved one margin tighter than
the ψ integral asks for and the ψ error estimate is not polluted by λ quadrature noise.
"""
torque_pitch_rtol(ctrl::KineticForcesControl) = isnan(ctrl.rtol_pitch) ? ctrl.psi_tolerance_margin * ctrl.rtol_psi : ctrl.rtol_pitch

# Pitch tolerance of the kinetic matrix build when `rtol_pitch` is left derived. There is no ψ quadrature to
# derive from; this value moves the free-boundary eigen-energies by ≤ 1e-4 relative to a 100× tighter build.
const MATRIX_RTOL_PITCH = 1e-3

"""
    matrix_pitch_rtol(ctrl::KineticForcesControl) -> Float64

Relative tolerance of the pitch (λ) quadrature in the calculated kinetic matrix build: `ctrl.rtol_pitch` when
set, otherwise `MATRIX_RTOL_PITCH` (1e-3). The matrices are splined over ψ and fed to the Euler–Lagrange
solve, so there is no outer quadrature tolerance to derive from.
"""
matrix_pitch_rtol(ctrl::KineticForcesControl) = isnan(ctrl.rtol_pitch) ? MATRIX_RTOL_PITCH : ctrl.rtol_pitch


# Internal state

# Concrete type of the `ForceFreeStates.build_kinetic_metric_matrices` interpolants, so the smats..zmats
# fields below are concretely typed and their per-ψ reads in `Torque.jl` dispatch statically.
const GeomInterp = typeof(cubic_interp(collect(0.0:0.25:1.0), Series(zeros(ComplexF64, 5, 1)); extrap=ExtendExtrap()))

"""
    KineticForcesInternal

Internal working state for KineticForces calculations: equilibrium-derived scalars, mode indexing, and
perturbation and geometric-matrix interpolants. Equilibrium and kinetic profiles are read from the
`PlasmaEquilibrium` and `KineticProfileSplines`, not copied here.

## Fields

  - `ro`, `bo`, `chi1`: Equilibrium geometry parameters
  - `mthsurf`, `mfac`: Poloidal grid info
  - `dbob_m`, `divx_m`: Perturbation mode interpolants
  - `sing_psis`: Sorted rational-surface ψ locations, used as panel boundaries of the outer ψ torque
    quadrature so resonant peaks fall on interval endpoints
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

    # Upper ψ bound (FFS psilim); the perturbation interpolants are invalid beyond it,
    # so the outer ψ quadrature clips here.
    psilim::Float64 = 1.0

    # Rational-surface ψ locations for ψ-quadrature paneling (see docstring).
    sing_psis::Vector{Float64} = Float64[]

    # θ knot grid (length mthsurf+1) read by the per-surface setup; never written after construction.
    tpsi_xs::Vector{Float64} = Float64[]

    # Geometric matrices s/t/x/y/z for the kinetic W vectors (not the DCON a/b/c/d/e/h matrices)
    smats::Union{Nothing,GeomInterp} = nothing   # CubicSeriesInterpolant over ψ, one mpert² block per n
    tmats::Union{Nothing,GeomInterp} = nothing
    xmats::Union{Nothing,GeomInterp} = nothing
    ymats::Union{Nothing,GeomInterp} = nothing
    zmats::Union{Nothing,GeomInterp} = nothing

    # Clebsch displacement vectors for mode-coupled dW contraction
    xs_m::Any = nothing            # Vector of 3 CubicSeriesInterpolants: [ξ^ψ, ∂ξ^ψ/∂ψ, ξ^α]

    # Method tracking — available methods/docs come from `METHOD_REGISTRY`.
    method::String = ""

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
    # Axis toroidal field F(0)/ro that normalizes λ = μ·bo/E; F_spline stores 2πF.
    bo_axis = abs(equil.profiles.F_spline(0.0)) / (2π * equil.ro)
    KineticForcesInternal(;
        ro=equil.ro,
        bo=bo_axis,
        chi1=2π * equil.psio,
        mthsurf,
        tpsi_xs=collect(range(0.0, 1.0; length=nth)),
        verbose
    )
end

"""
    set_perturbation_data!(kf_intr, pe_state, ffs, equil, metric)

Populate the perturbation interpolants of `kf_intr` from the PE Clebsch displacements: `xs_m`
([ξ^ψ, ∂ξ^ψ/∂ψ, ξ^α] over ψ), and the δB/B (`dbob_m`) and ∇·ξ⊥ (`divx_m`) Fourier modes, obtained by applying
the S,T,X,Y,Z geometric matrices and removing the J·B² weight (see `_jbb_deweight!`). Port of Fortran
`set_peq` (pentrc/inputs.f90). Errors unless the run has a single toroidal mode.
"""
function set_perturbation_data!(kf_intr::KineticForcesInternal, pe_state::PerturbedEquilibrium.PerturbedEquilibriumState,
    ffs::ForceFreeStates.ForceFreeStatesResult,
    equil::Equilibrium.PlasmaEquilibrium,
    metric::ForceFreeStates.MetricData)
    ffs.npert == 1 || error("KineticForces supports a single toroidal mode; this run has n = $(ffs.nlow):$(ffs.nhigh).")

    # Copy mode numbers from FFS
    kf_intr.mlow = ffs.mlow
    kf_intr.mhigh = ffs.mhigh
    kf_intr.mpert = ffs.mpert
    kf_intr.nlow = ffs.nlow
    kf_intr.nhigh = ffs.nhigh
    kf_intr.npert = ffs.npert
    kf_intr.numpert_total = ffs.numpert_total
    kf_intr.mfac = collect(ffs.mlow:ffs.mhigh)
    kf_intr.psilim = ffs.psilim

    # Rational-surface ψ locations (ideal + kinetic EL) become panel boundaries for the
    # outer ψ torque quadrature; dedupe against coincident points happens in psi_panel_points.
    kf_intr.sing_psis = sort!(vcat([s.psifac for s in ffs.surfaces], [s.psifac for s in ffs.kinetic.kinsing]))

    # Bail if no xi_modes available (PE didn't run or failed)
    if pe_state.xi_modes === nothing || isempty(pe_state.psi_grid)
        @warn "set_perturbation_data!: no xi_modes available, skipping perturbation build"
        return
    end

    xi_modes = pe_state.xi_modes
    psi_grid = pe_state.psi_grid
    npsi = length(psi_grid)
    mpert = ffs.mpert

    # Build xs_m: 3 CubicSeriesInterpolants from Clebsch displacement matrices
    # xs_m[1] = ξ^ψ (unregularized), xs_m[2] = ∂ξ^ψ/∂ψ (regularized), xs_m[3] = ξ^α
    itp_opts = (; extrap=ExtendExtrap())
    # clebsch_alpha is already the physical ξ^α = xms/χ₁ (as Fortran gpout_xclebsch writes it); use directly.
    clebsch_alpha_mat = xi_modes.clebsch_alpha
    xs_m_1 = cubic_interp(psi_grid, Series(xi_modes.clebsch_psi); itp_opts...)
    xs_m_2 = cubic_interp(psi_grid, Series(xi_modes.clebsch_psi1); itp_opts...)
    xs_m_3 = cubic_interp(psi_grid, Series(clebsch_alpha_mat); itp_opts...)
    kf_intr.xs_m = [xs_m_1, xs_m_2, xs_m_3]

    # Build geometric matrices (S,T,X,Y,Z) for JBB deweighting
    geom_mats = ForceFreeStates.build_kinetic_metric_matrices(equil, ffs, metric)

    # Build FourierTransform for the JBB deweighting DFT round-trip
    mthsurf = kf_intr.mthsurf
    ft = Utilities.FourierTransforms.FourierTransform(mthsurf, mpert, ffs.mlow)

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
        xsp = view(xi_modes.clebsch_psi, ipsi, :)       # ξ^ψ [mpert]
        xmp1 = view(xi_modes.clebsch_psi1, ipsi, :)       # ∂ξ^ψ/∂ψ [mpert]
        xms = view(clebsch_alpha_mat, ipsi, :)            # ξ^α [mpert]

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

        # jbb_kapx = smat·xsp + tmat·xms;  jbb_divx = xmat·xmp1 + ymat·xsp + zmat·xms
        mul!(jbb_kapx, smat, xsp)
        mul!(jbb_kapx, tmat, xms, 1.0 + 0.0im, 1.0 + 0.0im)   # += tmat * xms
        mul!(jbb_divx, xmat, xmp1)
        mul!(jbb_divx, ymat, xsp, 1.0 + 0.0im, 1.0 + 0.0im)  # += ymat * xsp
        mul!(jbb_divx, zmat, xms, 1.0 + 0.0im, 1.0 + 0.0im)  # += zmat * xms
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

Remove the J·B² weight from m-space modes: inverse DFT, divide by J(ψ,θ)·B(ψ,θ)², forward DFT into `out`.
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


# Result structs

"""
    MethodResult

Results for one NTV computation method across all flux surfaces.
"""
@kwdef mutable struct MethodResult
    method::String = ""
    nn::Int = 0
    torque_profile::Any = nothing     # Interpolant of dT/dψ(ψ) from ψ integration
    total_torque::ComplexF64 = 0.0 + 0.0im  # complex T: Re = torque T_φ [N·m], Im = 2n·δW_k (Logan 2013 Eq. 19)
    total_energy::Float64 = 0.0              # real kinetic energy δW_k = Im(T)/(2n) [J]
    # Per-step ψ profile from outer quadrature
    psi_grid::Vector{Float64} = Float64[]
    dtdpsi::Vector{ComplexF64} = ComplexF64[]
    t_cumulative::Vector{ComplexF64} = ComplexF64[]
    # dT/dψ resolved by bounce harmonic ℓ, shape (length(psi_grid), length(ell)) with ell = -nl:nl.
    # Summing over ℓ reproduces `dtdpsi`. Empty when no ψ profile was produced.
    dtdpsi_ell::Matrix{ComplexF64} = zeros(ComplexF64, 0, 0)
    ell::Vector{Int} = Int[]
    psi_nsteps::Int = 0
    # ψ-quadrature panel boundaries and the located kinetic-resonance surfaces (first n)
    panel_psis::Vector{Float64} = Float64[]
    resonance_psis::Vector{Float64} = Float64[]
end

"""
    KineticForcesState

Accumulated results from all KineticForces computations.
Written to gpec.h5 under the "KineticForces" group.
"""
@kwdef mutable struct KineticForcesState
    method_results::Dict{String,MethodResult} = Dict{String,MethodResult}()
    # Block-diagonal kinetic matrices: key=method, value=(numpert_total, numpert_total, 6)
    kinetic_matrices::Dict{String,Array{ComplexF64,3}} = Dict{String,Array{ComplexF64,3}}()
end

"""
    KineticForcesResult

The kinetic torque of a perturbed equilibrium, from [`kinetic_torque`](@ref).

## Fields

  - `total::KineticForcesState` - The torque by method, summed over species.
  - `species::Vector{Pair{String,KineticForcesState}}` - Each species' contribution by label; empty for a single-ion run.
  - `axis_psi_c::Float64` - The near-axis validity boundary the ψ quadrature used.
  - `pe::PerturbedEquilibrium.PerturbedEquilibriumState` - The perturbed equilibrium the torque was evaluated on.
  - `control::KineticForcesControl` - The kinetic model settings.
"""
struct KineticForcesResult
    total::KineticForcesState
    species::Vector{Pair{String,KineticForcesState}}
    axis_psi_c::Float64
    pe::PerturbedEquilibrium.PerturbedEquilibriumState
    control::KineticForcesControl
end
