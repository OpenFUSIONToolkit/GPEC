"""
    IonSpecies(; z, m, fraction=NaN, density="")

One main-ion species of the plasma. `z`/`m` are the charge (e) and mass (proton masses). The
density is given by exactly one of `fraction` or `density`, which select a fraction of the total
`n_i` profile or an explicit per-species profile (`n_*` dataset) in the kinetic file.
"""
struct IonSpecies
    z::Int
    m::Int
    fraction::Float64
    density::String
end

IonSpecies(; z::Integer, m::Integer, fraction=NaN, density="") =
    IonSpecies(Int(z), Int(m), Float64(fraction), String(density))
IonSpecies(d::AbstractDict) = IonSpecies(; (Symbol(k) => v for (k, v) in d)...)
Base.convert(::Type{Vector{IonSpecies}}, v::AbstractVector) =
    IonSpecies[x isa IonSpecies ? x : IonSpecies(x) for x in v]

"""
    EquilibriumConfig(...)

An immutable struct containing configuration parameters for equilibrium reconstruction
specified in the input.

## Fields

  - `eq_type::String` - Type of equilibrium file ("efit", "chease", "ifile", "imas", ...); an analytic config implies its own
  - `eq_filename::String` - Path to equilibrium input file
  - `jac_type::String` - Jacobian coordinate type ("hamada", "pest", "equal_arc", "boozer", "park", "custom")
  - `power_bp::Int` - Poloidal field power exponent for Jacobian (internal, derived from `jac_type`; not a deck key)
  - `power_b::Int` - Total field power exponent for Jacobian (internal, derived from `jac_type`; not a deck key)
  - `power_r::Int` - Major radius power exponent for Jacobian (internal, derived from `jac_type`; not a deck key)
  - `power_rc::Int` - Minor radius (rfac = √((R-R₀)²+(Z-Z₀)²)) power exponent for Jacobian (internal, derived from `jac_type`; not a deck key)
  - `jac_custom_power_bp::Int` - Poloidal-field exponent for a custom Jacobian `J = Bp^bp · B^b / (R^r · rfac^rc)`; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_b::Int` - Total-field exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_r::Int` - Major-radius exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_rc::Int` - Minor-radius (rfac) exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `r0_norm::Float64` - Major radius normalization for CHEASE/EQDSK [m]
  - `b0_norm::Float64` - On-axis toroidal field normalization for CHEASE/EQDSK [T]
  - `grid_type::String` - Grid type for flux surface discretization ("auto" — two-pass measured-curvature
    refinement when mpsi=0, three-region log layout when mpsi>0; "core_edge_packed" (sin² spacing, dense at axis and edge), "pow1", "uniform";
    inverse equilibria also "rho" (uniform in √ψ from psilow to psihigh) and "original" (the input
    file's own ψ nodes between psilow and psihigh; ignores mpsi))
  - `psilow::Float64` - Lower limit of normalized flux coordinate
  - `psihigh::Float64` - Requested upper limit of normalized flux coordinate; the value the
    equilibrium is actually formed on is `EquilibriumParameters.psihigh_resolved`.
  - `mpsi::Int` - Number of radial grid intervals; 0 with grid_type="auto" selects the two-pass auto
    grid. Standalone `PlasmaEquilibrium` callers get the coarse pass-1 grid unless they refine via
    `refined_psi_grid` + `override_psi_nodes`.
  - `psi_accuracy::Float64` - Target relative accuracy τ of splined profile derivatives for the
    two-pass auto grid (knot count scales as τ^(-1/3))
  - `mtheta::Int` - Number of poloidal grid points
  - `newq0::Float64` - Target on-axis safety factor q(0); the q and F profiles are rescaled to
    meet it (0 = use input value, -1 = use the axis extrapolation with its sign flipped)
  - `flux_surface_rtol::Float64` - Relative tolerance of the ODE solves that trace each flux surface; their absolute tolerance
    is `equil_abstol(flux_surface_rtol)`, i.e. `flux_surface_rtol` capped at `EQUIL_ABSTOL_MAX`
  - `imas_cocos::Int` - Expected COCOS convention of an IMAS `dd.equilibrium` input (11 = IMAS standard, 2 = GPEC internal)
  - `kinetic_file::String` - Kinetic-profile file read once into `PlasmaEquilibrium.kinetic` (see [`read_kinetic_file`](@ref));
    empty for none. Every kinetic consumer (NTV, tearing, the layer-overlap scan, grid refinement) derives its view from it.
  - `kinetic_group::String` - HDF5 group of `kinetic_file` holding the profiles
  - `ion_species`, `impurity_charge`, `impurity_mass`, `zeff_default`, `chi_e_default`, `chi_i_default`,
    `chi_phi_default`, `tau_E`, `resistivity_model`, `lnLambda_form` - The plasma composition, profile defaults
    and resistivity closure applied to the profiles of `kinetic_file` (see [`KineticProfileData`](@ref)); `nothing`
    keeps the `KineticProfileData` default. Setting any of them needs `kinetic_file`.
"""
@kwdef struct EquilibriumConfig
    eq_type::String = "efit"
    eq_filename::String = "mypath"
    r0_norm::Float64 = 1.0
    b0_norm::Float64 = 1.0

    jac_type::String = "hamada"
    power_bp::Int = 0
    power_b::Int = 0
    power_r::Int = 0
    power_rc::Int = 0

    jac_custom_power_bp::Int = 0
    jac_custom_power_b::Int = 0
    jac_custom_power_r::Int = 0
    jac_custom_power_rc::Int = 0

    grid_type::String = "auto"
    psilow::Float64 = 1e-2
    psihigh::Float64 = 0.9995
    mpsi::Int = 0
    psi_accuracy::Float64 = 0.001
    mtheta::Int = 512

    newq0::Float64 = 0.0
    flux_surface_rtol::Float64 = 1e-10

    imas_cocos::Int = 11

    kinetic_file::String = ""
    kinetic_group::String = "/"
    ion_species::Union{Nothing,Vector{IonSpecies}} = nothing
    impurity_charge::Union{Nothing,Int} = nothing
    impurity_mass::Union{Nothing,Int} = nothing
    zeff_default::Union{Nothing,Float64} = nothing
    chi_e_default::Union{Nothing,Float64} = nothing
    chi_i_default::Union{Nothing,Float64} = nothing
    chi_phi_default::Union{Nothing,Float64} = nothing
    tau_E::Union{Nothing,Float64} = nothing
    resistivity_model::Union{Nothing,Symbol} = nothing
    lnLambda_form::Union{Nothing,Symbol} = nothing

    """
    Inner constructor that derives the Jacobian exponents from `jac_type` and validates the inputs.
    """
    # The `_` slots are the `power_*` fields, always derived from `jac_type` here.
    function EquilibriumConfig(eq_type, eq_filename, r0_norm, b0_norm, jac_type, _, _, _, _,
        jac_custom_power_bp, jac_custom_power_b, jac_custom_power_r, jac_custom_power_rc,
        grid_type, psilow, psihigh, mpsi, psi_accuracy, mtheta, newq0, flux_surface_rtol,
        imas_cocos, kinetic_file, kinetic_group, composition...)
        if jac_type == "hamada"
            @info "Forcing hamada coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 0
            power_r = 0
            power_rc = 0
        elseif jac_type == "pest"
            @info "Forcing pest coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 0
            power_r = 2
            power_rc = 0
        elseif jac_type == "equal_arc"
            @info "Forcing equal_arc coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 1
            power_r = 0
            power_rc = 0
        elseif jac_type == "boozer"
            @info "Forcing boozer coordinate jacobian exponents: power_*"
            power_b = 2
            power_bp = 0
            power_r = 0
            power_rc = 0
        elseif jac_type == "park"
            @info "Forcing park coordinate jacobian exponents: power_*"
            power_b = 1
            power_bp = 0
            power_r = 0
            power_rc = 0
        elseif jac_type == "custom"
            # Custom Jacobian: the user-facing knobs define the exponents.
            power_bp = jac_custom_power_bp
            power_b = jac_custom_power_b
            power_r = jac_custom_power_r
            power_rc = jac_custom_power_rc
            # Normalize to a named type when the powers match, so fast paths are taken.
            if power_b == 0 && power_bp == 0 && power_r == 0 && power_rc == 0
                jac_type = "hamada"
                @info "Recognized hamada jacobian from power exponents"
            elseif power_b == 0 && power_bp == 0 && power_r == 2 && power_rc == 0
                jac_type = "pest"
                @info "Recognized pest jacobian from power exponents"
            elseif power_b == 0 && power_bp == 1 && power_r == 0 && power_rc == 0
                jac_type = "equal_arc"
                @info "Recognized equal_arc jacobian from power exponents"
            elseif power_b == 2 && power_bp == 0 && power_r == 0 && power_rc == 0
                jac_type = "boozer"
                @info "Recognized boozer jacobian from power exponents"
            elseif power_b == 1 && power_bp == 0 && power_r == 0 && power_rc == 0
                jac_type = "park"
                @info "Recognized park jacobian from power exponents"
            else
                @info "Using manual jacobian exponents: power b, bp, r, rc = $(power_b), $(power_bp), $(power_r), $(power_rc)"
            end
        else
            error("Cannot recognize jac_type = $(jac_type)")
        end
        (isfinite(flux_surface_rtol) && flux_surface_rtol > 0) || error("flux_surface_rtol = $flux_surface_rtol must be a positive, finite relative tolerance")
        if psihigh > 1.0
            @warn "psihigh = $psihigh exceeds 1.0 (separatrix); clamping to 1.0"
        end
        psihigh = min(psihigh, 1.0)
        isempty(kinetic_file) && any(!isnothing, composition) &&
            error("the kinetic composition keys of [Equilibrium] describe the profiles of kinetic_file; set kinetic_file")
        return new(eq_type, eq_filename, r0_norm, b0_norm, jac_type, power_bp, power_b, power_r, power_rc,
            jac_custom_power_bp, jac_custom_power_b, jac_custom_power_r, jac_custom_power_rc,
            grid_type, psilow, psihigh, mpsi, psi_accuracy, mtheta, newq0, flux_surface_rtol,
            imas_cocos, kinetic_file, kinetic_group, composition...)
    end
end

# The kinetic composition keys of a config, the ones it sets, as `KineticProfileData` keywords.
const KINETIC_COMPOSITION_KEYS = (:ion_species, :impurity_charge, :impurity_mass, :zeff_default, :chi_e_default, :chi_i_default,
    :chi_phi_default, :tau_E, :resistivity_model, :lnLambda_form)
kinetic_composition(config::EquilibriumConfig) =
    (; (k => getfield(config, k) for k in KINETIC_COMPOSITION_KEYS if getfield(config, k) !== nothing)...)

"""
    EquilibriumParameters

A mutable struct containing computed equilibrium parameters and diagnostic flags.

## Fields

  - `ro::Union{Nothing,Float64}` - R-coordinate of the magnetic axis [m]
  - `zo::Union{Nothing,Float64}` - Z-coordinate of the magnetic axis [m]
  - `psio::Union{Nothing,Float64}` - Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
  - `psihigh_resolved::Union{Nothing,Float64}` - Outer flux limit the equilibrium was formed on
    (the outermost ψ node); the plasma edge downstream of `PlasmaEquilibrium`.
  - `rsep::Union{Nothing,Vector{Float64}}` - R-coordinates of the plasma boundary [m]
  - `zsep::Union{Nothing,Vector{Float64}}` - Z-coordinates of the plasma boundary [m]
  - `rext::Union{Nothing,Vector{Float64}}` - R-coordinates of the plasma edge [m]
  - `zext::Union{Nothing,Vector{Float64}}` - Z-coordinates of the plasma edge [m]
  - `psi0::Union{Nothing,Float64}` - Normalized poloidal flux at reference location
  - `b0::Union{Nothing,Float64}` - Total magnetic field strength at the axis [T]
  - `q0::Union{Nothing,Float64}` - Safety factor at the axis
  - `qmin::Union{Nothing,Float64}` - Minimum safety factor in the plasma
  - `qmax::Union{Nothing,Float64}` - Maximum safety factor in the plasma
  - `qa::Union{Nothing,Float64}` - Safety factor at the plasma edge
  - `q95::Union{Nothing,Float64}` - Safety factor at 95% flux surface
  - `qextrema_psi::Union{Nothing,Vector{Float64}}` - Normalized flux values at q extrema
  - `qextrema_q::Union{Nothing,Vector{Float64}}` - Safety factor values at extrema
  - `mextrema::Union{Nothing,Int}` - Number of extrema in q-profile
  - `psi_norm::Union{Nothing,Float64}` - Normalized poloidal flux
  - `psi_axis::Union{Nothing,Float64}` - Poloidal flux at the axis
  - `psi_boundary::Union{Nothing,Float64}` - Poloidal flux at the boundary
  - `psi_boundary_norm::Union{Nothing,Float64}` - Normalized boundary flux
  - `psi_boundary_offset::Union{Nothing,Float64}` - Boundary flux offset
  - `psi_axis_offset::Union{Nothing,Float64}` - Axis flux offset
  - `psi_boundary_sign::Union{Nothing,Int}` - Sign of boundary flux
  - `psi_axis_sign::Union{Nothing,Int}` - Sign of axis flux
  - `psi_boundary_zero::Union{Nothing,Bool}` - Whether boundary flux is zero
  - `rmean::Union{Nothing,Float64}` - Mean major radius [m]
  - `amean::Union{Nothing,Float64}` - Mean minor radius [m]
  - `aratio::Union{Nothing,Float64}` - Aspect ratio (R0/a)
  - `kappa::Union{Nothing,Float64}` - Plasma elongation
  - `delta1::Union{Nothing,Float64}` - Upper triangularity
  - `delta2::Union{Nothing,Float64}` - Lower triangularity
  - `bt0::Union{Nothing,Float64}` - Toroidal field at axis [T]
  - `crnt::Union{Nothing,Float64}` - Plasma current [A], always positive
  - `bt_sign::Int` - Sign of the toroidal field (+1 or -1; `bt0` is always positive)
  - `ip_sign::Int` - Sign of the plasma current as the source file states it (+1 or -1)
  - `bwall::Union{Nothing,Float64}` - Toroidal field at wall [T]
  - `verbose::Bool` - Enable verbose output
  - `diagnose_src::Bool` - Enable source data diagnostics
  - `diagnose_maxima::Bool` - Enable extrema diagnostics
  - `volume::Union{Nothing,Float64}` - Plasma volume [m³]
  - `betat::Union{Nothing,Float64}` - Toroidal beta
  - `betan::Union{Nothing,Float64}` - Normalized beta
  - `betaj::Union{Nothing,Float64}` - Total beta
  - `betap1::Union{Nothing,Float64}` - Poloidal beta (definition 1)
  - `betap2::Union{Nothing,Float64}` - Poloidal beta (definition 2)
  - `betap3::Union{Nothing,Float64}` - Poloidal beta (definition 3)
  - `li1::Union{Nothing,Float64}` - Internal inductance (definition 1)
  - `li2::Union{Nothing,Float64}` - Internal inductance (definition 2)
  - `li3::Union{Nothing,Float64}` - Internal inductance (definition 3)
"""
@kwdef mutable struct EquilibriumParameters
    ro::Union{Nothing,Float64} = nothing
    zo::Union{Nothing,Float64} = nothing
    psio::Union{Nothing,Float64} = nothing
    psihigh_resolved::Union{Nothing,Float64} = nothing
    rsep::Union{Nothing,Vector{Float64}} = nothing
    zsep::Union{Nothing,Vector{Float64}} = nothing
    rext::Union{Nothing,Vector{Float64}} = nothing
    zext::Union{Nothing,Vector{Float64}} = nothing
    psi0::Union{Nothing,Float64} = nothing
    b0::Union{Nothing,Float64} = nothing
    q0::Union{Nothing,Float64} = nothing
    qmin::Union{Nothing,Float64} = nothing
    qmax::Union{Nothing,Float64} = nothing
    qa::Union{Nothing,Float64} = nothing
    q95::Union{Nothing,Float64} = nothing
    qextrema_psi::Union{Nothing,Vector{Float64}} = nothing
    qextrema_q::Union{Nothing,Vector{Float64}} = nothing
    mextrema::Union{Nothing,Int} = nothing
    psi_norm::Union{Nothing,Float64} = nothing
    psi_axis::Union{Nothing,Float64} = nothing
    psi_boundary::Union{Nothing,Float64} = nothing
    psi_boundary_norm::Union{Nothing,Float64} = nothing
    psi_boundary_offset::Union{Nothing,Float64} = nothing
    psi_axis_offset::Union{Nothing,Float64} = nothing
    psi_boundary_sign::Union{Nothing,Int} = nothing
    psi_axis_sign::Union{Nothing,Int} = nothing
    psi_boundary_zero::Union{Nothing,Bool} = nothing
    rmean::Union{Nothing,Float64} = nothing
    amean::Union{Nothing,Float64} = nothing
    aratio::Union{Nothing,Float64} = nothing
    kappa::Union{Nothing,Float64} = nothing
    delta1::Union{Nothing,Float64} = nothing
    delta2::Union{Nothing,Float64} = nothing
    bt0::Union{Nothing,Float64} = nothing
    crnt::Union{Nothing,Float64} = nothing
    bt_sign::Int = 1
    ip_sign::Int = 1
    bwall::Union{Nothing,Float64} = nothing
    verbose::Bool = false
    diagnose_src::Bool = false
    diagnose_maxima::Bool = false
    volume::Union{Nothing,Float64} = nothing
    betat::Union{Nothing,Float64} = nothing
    betan::Union{Nothing,Float64} = nothing
    betaj::Union{Nothing,Float64} = nothing
    betap1::Union{Nothing,Float64} = nothing
    betap2::Union{Nothing,Float64} = nothing
    betap3::Union{Nothing,Float64} = nothing
    li1::Union{Nothing,Float64} = nothing
    li2::Union{Nothing,Float64} = nothing
    li3::Union{Nothing,Float64} = nothing
end
