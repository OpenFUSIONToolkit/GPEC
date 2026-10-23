function symbolize_keys(dict::Dict{String,Any})
    return Dict(Symbol(k) => v for (k, v) in dict)
end

"""
    EquilibriumConfig(...)

An immutable struct containing configuration parameters for equilibrium reconstruction
specified in the input.

## Fields

  - `eq_type::String` - Type of equilibrium file ("efit", "solovev", "lar", etc.)
  - `eq_filename::String` - Path to equilibrium input file
  - `jac_type::String` - Jacobian coordinate type ("hamada", "pest", "equal_arc", "boozer", "park", "custom")
  - `power_bp::Int` - Poloidal field power exponent for Jacobian (internal, derived from `jac_type`; deprecated as a TOML input key)
  - `power_b::Int` - Total field power exponent for Jacobian (internal, derived from `jac_type`; deprecated as a TOML input key)
  - `power_r::Int` - Major radius power exponent for Jacobian (internal, derived from `jac_type`; deprecated as a TOML input key)
  - `power_rc::Int` - Minor radius (rfac = √((R-R₀)²+(Z-Z₀)²)) power exponent for Jacobian (internal, derived from `jac_type`; deprecated as a TOML input key)
  - `jac_custom_power_bp::Int` - Poloidal-field exponent for a custom Jacobian `J = Bp^bp · B^b / (R^r · rfac^rc)`; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_b::Int` - Total-field exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_r::Int` - Major-radius exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_rc::Int` - Minor-radius (rfac) exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `r0_norm::Float64` - Major radius normalization for CHEASE/EQDSK [m]
  - `b0_norm::Float64` - On-axis toroidal field normalization for CHEASE/EQDSK [T]
  - `grid_type::String` - Grid type for flux surface discretization ("auto" — two-pass measured-curvature
    refinement when mpsi=0, three-region log layout when mpsi>0; "core_edge_packed" (sin² spacing, dense at axis and edge), "pow1", "uniform";
    inverse equilibria also "rho" (uniform in √ψ from psilow to psihigh) and "original" (the input
    file's own ψ nodes off the axis; ignores psilow and mpsi); "log_asymptotic" is a legacy alias for "auto")
  - `psilow::Float64` - Lower limit of normalized flux coordinate
  - `psihigh::Float64` - Requested upper limit of normalized flux coordinate; the value the
    equilibrium is actually formed on is `EquilibriumParameters.psihigh_resolved`.
  - `mpsi::Int` - Number of radial grid intervals; 0 with grid_type="auto" selects the
    two-pass auto grid: the main driver forms a coarse pass-1 equilibrium, measures its curvature,
    pins knots on rational surfaces, and re-forms on the refined grid. Standalone `PlasmaEquilibrium`
    callers get the coarse pass-1 grid unless they refine via `refined_psi_grid` + `override_psi_nodes`.
  - `psi_accuracy::Float64` - Target relative accuracy τ of splined profile derivatives for the
    two-pass auto grid (knot count scales as τ^(-1/3))
  - `mtheta::Int` - Number of poloidal grid points
  - `newq0::Float64` - Target on-axis safety factor q(0); the q and F profiles are rescaled to
    meet it (0 = use input value, -1 = use the axis extrapolation with its sign flipped)
  - `flux_surface_rtol::Float64` - Relative tolerance of the ODE solves that trace each flux surface; their absolute tolerance
    is `equil_abstol(flux_surface_rtol)`, i.e. `flux_surface_rtol` capped at `EQUIL_ABSTOL_MAX`
  - `force_termination::Bool` - Terminate after equilibrium setup (skip stability calculations)
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

    force_termination::Bool = false

    # IMAS-specific: expected COCOS convention of the input dd.equilibrium (11=IMAS standard, 2=GPEC internal)
    imas_cocos::Int = 11

    """
    Modified internal constructor that enforces self consistency within the inputs
    """
    # The four `_` slots are the `power_bp/power_b/power_r/power_rc` struct fields, which @kwdef
    # forwards positionally. They are always derived below from `jac_type` (or `jac_custom_power_*`),
    # so their incoming values are ignored (hence `_`).
    function EquilibriumConfig(eq_type, eq_filename, r0_norm, b0_norm, jac_type, _, _, _, _,
        jac_custom_power_bp, jac_custom_power_b, jac_custom_power_r, jac_custom_power_rc,
        grid_type, psilow, psihigh, mpsi, psi_accuracy, mtheta, newq0, flux_surface_rtol,
        force_termination, imas_cocos)
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
        return new(eq_type, eq_filename, r0_norm, b0_norm, jac_type, power_bp, power_b, power_r, power_rc,
            jac_custom_power_bp, jac_custom_power_b, jac_custom_power_r, jac_custom_power_rc,
            grid_type, psilow, psihigh, mpsi, psi_accuracy, mtheta, newq0, flux_surface_rtol,
            force_termination, imas_cocos)
    end
end

"""
Outer constructor for EquilibriumConfig from a parsed TOML dictionary
"""
function EquilibriumConfig(equil_dict::Dict{String,Any}, base_path::String="./")
    # `eq_type` is always required.  `eq_filename` is required for file-based
    # equilibria (efit, chease, …) but optional for analytic types whose
    # parameters live in an embedded `[TJ_ANALYTIC_INPUT]` / `[SOL_INPUT]` /
    # `[LAR_INPUT]` section of the parent gpec.toml.
    if !haskey(equil_dict, "eq_type")
        error("Missing required key in [Equilibrium]: eq_type")
    end

    # Filter to only known parameters
    config_fields = Set(String.(fieldnames(EquilibriumConfig)))
    config_data = Dict{String,Any}()

    for (k, v) in equil_dict
        if k in config_fields
            config_data[k] = v
        else
            @warn "Unknown equilibrium parameter: $k"
        end
    end

    # Only resolve `eq_filename` against `base_path` if the user actually supplied one
    # (otherwise leave the kwdef sentinel for the embedded path). The empty string is the
    # rerun path's "no input file" marker and must stay empty, not become `base_path`.
    if haskey(config_data, "eq_filename") && !isempty(config_data["eq_filename"]) && !isabspath(config_data["eq_filename"])
        config_data["eq_filename"] = normpath(joinpath(base_path, config_data["eq_filename"]))
    end

    return EquilibriumConfig(; symbolize_keys(config_data)...)
end

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
  - `b_norm::Union{Nothing,Float64}` - Normalized magnetic field strength
  - `psi_axis::Union{Nothing,Float64}` - Poloidal flux at the axis
  - `psi_boundary::Union{Nothing,Float64}` - Poloidal flux at the boundary
  - `psi_boundary_norm::Union{Nothing,Float64}` - Normalized boundary flux
  - `psi_axis_norm::Union{Nothing,Float64}` - Normalized axis flux
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
  - `crnt::Union{Nothing,Float64}` - Plasma current [A]
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
    ro::Union{Nothing,Float64} = nothing # R-coordinate of the magnetic axis [m]
    zo::Union{Nothing,Float64} = nothing # Z-coordinate of the magnetic axis [m]
    psio::Union{Nothing,Float64} = nothing # Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
    psihigh_resolved::Union{Nothing,Float64} = nothing # Outer flux limit actually formed on (clamped config.psihigh)
    rsep::Union{Nothing,Vector{Float64}} = nothing # R-coordinates of the plasma boundary [m]
    zsep::Union{Nothing,Vector{Float64}} = nothing # Z-coordinates of the plasma boundary [m]
    rext::Union{Nothing,Vector{Float64}} = nothing # R-coordinates of the plasma edge [m]
    zext::Union{Nothing,Vector{Float64}} = nothing # Z-coordinates of the plasma edge [m]
    psi0::Union{Nothing,Float64} = nothing # Normalized poloidal flux
    b0::Union{Nothing,Float64} = nothing # Total magnetic field strength at the axis [T]
    q0::Union{Nothing,Float64} = nothing # Safety factor at the axis
    qmin::Union{Nothing,Float64} = nothing # Minimum safety factor in the plasma
    qmax::Union{Nothing,Float64} = nothing # Maximum safety factor in the plasma
    qa::Union{Nothing,Float64} = nothing # Safety factor at the plasma edge
    q95::Union{Nothing,Float64} = nothing # Safety factor at 95% flux surface
    qextrema_psi::Union{Nothing,Vector{Float64}} = nothing # Normalized poloidal flux values where q has extrema
    qextrema_q::Union{Nothing,Vector{Float64}} = nothing # Safety factor values at the extrema points
    mextrema::Union{Nothing,Int} = nothing # Number of extrema points in the q-profile
    psi_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    b_norm::Union{Nothing,Float64} = nothing # Normalized total magnetic field strength at the axis
    psi_axis::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    psi_boundary::Union{Nothing,Float64} = nothing # Poloidal flux at the boundary
    psi_boundary_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the boundary
    psi_axis_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    psi_boundary_offset::Union{Nothing,Float64} = nothing # Offset for the boundary poloidal flux
    psi_axis_offset::Union{Nothing,Float64} = nothing  # Offset for the axis poloidal flux
    psi_boundary_sign::Union{Nothing,Int} = nothing # Sign of the boundary poloidal flux
    psi_axis_sign::Union{Nothing,Int} = nothing # Sign of the axis poloidal flux
    psi_boundary_zero::Union{Nothing,Bool} = nothing # Whether the boundary poloidal flux is zero
    rmean::Union{Nothing,Float64} = nothing # Mean R-coordinate of the plasma [m]
    amean::Union{Nothing,Float64} = nothing # Mean minor radius of the plasma [m]
    aratio::Union{Nothing,Float64} = nothing # Aspect ratio of the plasma (R0/a)
    kappa::Union{Nothing,Float64} = nothing # Elongation of the plasma cross-section
    delta1::Union{Nothing,Float64} = nothing # Triangularity of the plasma cross-section (upper triangularity)
    delta2::Union{Nothing,Float64} = nothing # Triangularity of the plasma cross-section (lower triangularity)
    bt0::Union{Nothing,Float64} = nothing # Toroidal magnetic field at the axis [T] (always positive; sign in bt_sign)
    crnt::Union{Nothing,Float64} = nothing # Plasma current at the axis [A]
    bt_sign::Int = 1 # Sign of the toroidal field: +1 (positive Bt) or -1 (negative Bt, e.g. DIII-D standard)
    ip_sign::Int = 1 # Sign of the plasma current as the source file states it: +1 or -1 (crnt is always positive)
    bwall::Union{Nothing,Float64} = nothing # Toroidal magnetic field at the wall [T]
    verbose::Bool = false # Whether to print verbose output
    diagnose_src::Bool = false # Whether to diagnose source data
    diagnose_maxima::Bool = false # Whether to diagnose maxima in the equilibrium
    volume::Union{Nothing,Float64} = nothing # Plasma volume [m³]
    betat::Union{Nothing,Float64} = nothing # Toroidal beta (normalized pressure) at the axis
    betan::Union{Nothing,Float64} = nothing # Normalized beta at the axis
    betaj::Union{Nothing,Float64} = nothing # Total beta at the axis
    betap1::Union{Nothing,Float64} = nothing # Pressure beta at the axis
    betap2::Union{Nothing,Float64} = nothing # Toroidal beta at the axis
    betap3::Union{Nothing,Float64} = nothing # Poloidal beta at the axis
    li1::Union{Nothing,Float64} = nothing  # Internal inductance at the axis
    li2::Union{Nothing,Float64} = nothing  # External inductance at the axis
    li3::Union{Nothing,Float64} = nothing  # Total inductance at the axis
end
